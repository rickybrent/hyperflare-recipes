# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Captured Gemma 4 vision tower with canonical blocks and prepared projections."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4VisionRotaryEmbedding,
    apply_multidimensional_rope,
)

from hyperflare_recipes.components.linear import UnquantizedProjection
from hyperflare_recipes.components.varlen_attention import PreparedVarlenAttention
from ignition.preparation.rebuild_manifest import weight_container


HIDDEN = 1152
HEADS = 16
HEAD_DIM = 72
INTERMEDIATE = 4304
PATCH_FEATURES = 3 * 16 * 16
POOLING = 3


@dataclass(frozen=True)
class VisionMetadata:
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    num_patches: torch.Tensor

    @classmethod
    def allocate(cls, patches, device):
        if patches <= 0 or patches % (POOLING * POOLING):
            raise ValueError("Gemma 4 vision requires a positive patch capacity divisible by nine")
        return cls(
            torch.full((patches, 2), -1, dtype=torch.int64, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )


@dataclass(frozen=True)
class VisionScratch:
    patches: torch.Tensor
    patch_input: torch.Tensor
    hidden: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    attention: torch.Tensor
    projected: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    pooled: torch.Tensor
    normed: torch.Tensor
    output: torch.Tensor

    @classmethod
    def allocate(cls, patches, device, *, text_hidden):
        if patches <= 0 or patches % (POOLING * POOLING):
            raise ValueError("Gemma 4 vision patch capacity must be positive and divisible by nine")
        if text_hidden <= 0:
            raise ValueError("Gemma 4 vision text hidden size must be positive")
        def zeros(*shape, dtype=torch.bfloat16):
            return torch.zeros(shape, dtype=dtype, device=device)
        return cls(
            zeros(patches, PATCH_FEATURES, dtype=torch.float32),
            zeros(patches, PATCH_FEATURES),
            zeros(patches, HIDDEN),
            zeros(patches, HEADS, HEAD_DIM),
            zeros(patches, HEADS, HEAD_DIM),
            zeros(patches, HEADS, HEAD_DIM),
            zeros(patches, HIDDEN),
            zeros(patches, HIDDEN),
            zeros(patches, INTERMEDIATE),
            zeros(patches, INTERMEDIATE),
            zeros(patches // (POOLING * POOLING), HIDDEN, dtype=torch.float32),
            zeros(patches // (POOLING * POOLING), HIDDEN),
            zeros(patches // (POOLING * POOLING), text_hidden),
        )

    def rows(self, patches):
        if patches <= 0 or patches % (POOLING * POOLING) or patches > self.patches.shape[0]:
            raise ValueError("Gemma 4 vision view exceeds its patch capacity")
        merged = {"pooled", "normed", "output"}
        return type(self)(
            *(getattr(self, name)[:patches // (POOLING * POOLING) if name in merged else patches]
              for name in self.__dataclass_fields__)
        )


@weight_container(("parameters", "std_bias", "std_scale", "projection"))
@dataclass(frozen=True)
class VisionWeights:
    parameters: dict[str, torch.Tensor]
    std_bias: torch.Tensor
    std_scale: torch.Tensor
    projection: torch.Tensor

    @classmethod
    def load(cls, module, embed, loader):
        text_hidden = embed.embedding_projection.out_features
        parameters = {}
        for path, _ in module.named_parameters():
            tensor = loader.load_tensor(f"model.vision_tower.{path}")
            if tensor.dtype != torch.bfloat16:
                raise ValueError(
                    "Gemma 4 vision requires a BF16 tower; "
                    "the checkpoint stores it at another precision")
            parameters[path] = tensor
        std_bias, std_scale = loader.load_tensors(
            ("model.vision_tower.std_bias", "model.vision_tower.std_scale"))
        if std_bias.dtype != torch.bfloat16 or std_scale.dtype != torch.bfloat16:
            raise ValueError("Gemma 4 vision standardize buffers must be BF16")
        projection = loader.load_tensor("model.embed_vision.embedding_projection.weight")
        if projection.dtype != torch.bfloat16 or projection.shape != (text_hidden, HIDDEN):
            raise ValueError("Gemma 4 vision projection must be BF16 with text-hidden rows")
        return cls(parameters, std_bias, std_scale, projection)

    @classmethod
    def from_checkpoint(cls, loader, vision_config, text_config):
        from transformers.models.gemma4.configuration_gemma4 import (
            Gemma4TextConfig,
            Gemma4VisionConfig,
        )
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4MultimodalEmbedder,
            Gemma4VisionModel,
        )

        vision = Gemma4VisionConfig.from_dict(vision_config)
        text = Gemma4TextConfig.from_dict(text_config)
        with torch.device("meta"):
            module = Gemma4VisionModel(vision)
            embed = Gemma4MultimodalEmbedder(vision, text)
        return cls.load(module, embed, loader)


class _Linear(torch.nn.Module):
    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def forward(self, source):
        return self.projection(source)


class _Attention(torch.nn.Module):
    def __init__(self, module, scratch, metadata, weights, *, session, name, shared=None,
                 workspace=None):
        super().__init__()
        self.scratch = scratch
        self.q_norm, self.k_norm, self.v_norm = module.q_norm, module.k_norm, module.v_norm
        self.q_proj = _Linear(UnquantizedProjection(
            scratch.hidden, weights["q_proj"], scratch.query.view(-1, HIDDEN),
            session=session, name=f"{name}.q_proj"))
        self.k_proj = _Linear(UnquantizedProjection(
            scratch.hidden, weights["k_proj"], scratch.key.view(-1, HIDDEN),
            session=session, name=f"{name}.k_proj"))
        self.v_proj = _Linear(UnquantizedProjection(
            scratch.hidden, weights["v_proj"], scratch.value.view(-1, HIDDEN),
            session=session, name=f"{name}.v_proj"))
        self.o_proj = _Linear(UnquantizedProjection(
            scratch.attention, weights["o_proj"], scratch.projected,
            session=session, name=f"{name}.o_proj"))
        self.attention = shared.attention if shared is not None else PreparedVarlenAttention(
            scratch.query, scratch.key, scratch.value, metadata.cu_seqlens,
            scratch.attention.view_as(scratch.query), session=session, workspace=workspace,
            name=name, softmax_scale=1.0)

    def forward(self, hidden_states, position_embeddings=None, position_ids=None,
                attention_mask=None, **kwargs):
        s = self.scratch
        rows = hidden_states.shape[0]
        query = self.q_norm(self.q_proj(hidden_states).view(rows, HEADS, HEAD_DIM))
        key = self.k_norm(self.k_proj(hidden_states).view(rows, HEADS, HEAD_DIM))
        value = self.v_norm(self.v_proj(hidden_states).view(rows, HEADS, HEAD_DIM))
        cos, sin = position_embeddings
        s.query.copy_(apply_multidimensional_rope(query, cos, sin, position_ids, unsqueeze_dim=1))
        s.key.copy_(apply_multidimensional_rope(key, cos, sin, position_ids, unsqueeze_dim=1))
        s.value.copy_(value)
        self.attention()
        return self.o_proj(s.attention), None


class _MLP(torch.nn.Module):
    def __init__(self, module, scratch, weights, *, session, name):
        super().__init__()
        self.scratch = scratch
        self.act_fn = module.act_fn
        self.gate = _Linear(UnquantizedProjection(
            scratch.hidden, weights["gate_proj"], scratch.gate,
            session=session, name=f"{name}.gate_proj"))
        self.up = _Linear(UnquantizedProjection(
            scratch.hidden, weights["up_proj"], scratch.up,
            session=session, name=f"{name}.up_proj"))
        self.down = _Linear(UnquantizedProjection(
            scratch.gate, weights["down_proj"], scratch.projected,
            session=session, name=f"{name}.down_proj"))

    def forward(self, hidden_states):
        s = self.scratch
        self.gate(hidden_states)
        self.up(hidden_states)
        s.gate.copy_(self.act_fn(s.gate) * s.up)
        return self.down(s.gate)


class PreparedVision:
    """Encode packed patches into text-space Gemma 4 image features.

    The canonical Gemma 4 vision tower retains its patch embedder, RMS norms,
    axial 2-D RoPE, gated MLP, pooling, standardize buffers, and text projection.
    Prepared projections bind each capacity's GEMMs; attention runs through the
    same non-causal varlen primitive as the other vision towers.
    """

    def __init__(self, module, embed, scratch, metadata, weights, *, session, name="vision",
                 workspace=None):
        config = module.config
        if (config.hidden_size, config.num_attention_heads, config.head_dim,
                config.intermediate_size, config.num_hidden_layers,
                config.patch_size, config.pooling_kernel_size) != (
                HIDDEN, HEADS, HEAD_DIM, INTERMEDIATE, 27, 16, POOLING):
            raise ValueError("Gemma 4 vision geometry differs from the checkpoint contract")
        capacity, device = scratch.hidden.shape[0], scratch.hidden.device
        if scratch.output.shape != (capacity // (POOLING * POOLING), weights.projection.shape[0]):
            raise ValueError("Gemma 4 vision output scratch shape differs from text projection")
        if metadata.position_ids.shape != (capacity, 2):
            raise ValueError("Gemma 4 vision metadata and scratch capacities differ")
        if metadata.num_patches.item() or bool(metadata.cu_seqlens.any().item()):
            raise ValueError("Gemma 4 vision preparation requires inactive metadata")
        self.module, self.embed, self.scratch, self.metadata = module, embed, scratch, metadata
        for path, tensor in weights.parameters.items():
            parent, _, attribute = path.rpartition(".")
            setattr(module.get_submodule(parent), attribute, torch.nn.Parameter(tensor, requires_grad=False))
        module.std_bias = torch.nn.Buffer(weights.std_bias)
        module.std_scale = torch.nn.Buffer(weights.std_scale)
        module.encoder.rotary_emb = Gemma4VisionRotaryEmbedding(config).to(device)
        self.patch_linear = _Linear(UnquantizedProjection(
            scratch.patch_input, weights.parameters["patch_embedder.input_proj.weight"],
            scratch.hidden, session=session, name=f"{name}.patch_embed"))
        for index, block in enumerate(module.encoder.layers):
            prefix = f"encoder.layers.{index}"
            block.self_attn = _Attention(
                block.self_attn, scratch, metadata,
                {key: weights.parameters[f"{prefix}.self_attn.{key}.linear.weight"]
                 for key in ("q_proj", "k_proj", "v_proj", "o_proj")},
                session=session, name=f"{name}.{prefix}.self_attn",
                shared=module.encoder.layers[0].self_attn if index else None,
                workspace=workspace)
            block.mlp = _MLP(block.mlp, scratch,
                {key: weights.parameters[f"{prefix}.mlp.{key}.linear.weight"]
                 for key in ("gate_proj", "up_proj", "down_proj")},
                session=session, name=f"{name}.{prefix}.mlp")
        embed.embedding_projection = _Linear(UnquantizedProjection(
            scratch.normed, weights.projection, scratch.output,
            session=session, name=f"{name}.embedding_projection"))
        self.output_rows = torch.arange(capacity // (POOLING * POOLING), device=device)

    def __call__(self, pixels, position_ids):
        module, scratch, metadata = self.module, self.scratch, self.metadata
        scratch.patches.copy_(pixels)
        scratch.patch_input.copy_((scratch.patches - 0.5) * 2)
        padding = (position_ids == -1).all(-1)
        hidden = self.patch_linear(scratch.patch_input)
        hidden += module.patch_embedder._position_embeddings(position_ids, padding)
        cos, sin = module.encoder.rotary_emb(hidden, position_ids[None])
        cos, sin = cos[0], sin[0]
        for block in module.encoder.layers:
            hidden = block(hidden, attention_mask=None, position_embeddings=(cos, sin),
                           position_ids=position_ids)
        scratch.pooled.copy_(module.pooler(
            hidden_states=hidden[None], pixel_position_ids=position_ids[None],
            padding_positions=padding[None],
            output_length=scratch.pooled.shape[0])[0][0])
        hidden = (scratch.pooled - module.std_bias.float()) * module.std_scale.float()
        hidden = hidden.to(scratch.hidden.dtype)
        scratch.normed.copy_(self.embed.embedding_pre_projection_norm(hidden))
        self.embed.embedding_projection(scratch.normed)
        live = self.output_rows < metadata.num_patches // (POOLING * POOLING)
        scratch.output.masked_fill_(~live[:, None], 0)
        return scratch.output
