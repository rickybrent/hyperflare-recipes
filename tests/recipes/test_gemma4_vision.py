# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 vision encoder parity with the checkpoint implementation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from hyperflare_recipes.components.weights import DirectWeightLoader
from ignition.preparation.checkpoint import inspect_indexed_checkpoint
from hyperflare_recipes.recipes.gemma4.vision import (
    PreparedVision,
    VisionMetadata,
    VisionScratch,
    VisionWeights,
)

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
                          reason="requires explicit GPU assignment and FLARE_RUN_CUDA_TESTS=1")


def snapshot() -> Path:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    path = (hub / "models--google--gemma-4-31B-it" / "snapshots"
            / "842da3794eaa0b77d5f08bae87a17459d91ff475")
    if not (path / "config.json").is_file():
        pytest.skip("Gemma 4 31B checkpoint is not cached")
    return path


def _configs(path):
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4TextConfig,
        Gemma4VisionConfig,
    )
    raw = json.loads((path / "config.json").read_text())
    vision = Gemma4VisionConfig.from_dict(raw["vision_config"])
    vision._attn_implementation = "eager"
    text = Gemma4TextConfig.from_dict(raw["text_config"])
    return vision, text


def _modules(path):
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4MultimodalEmbedder,
        Gemma4VisionModel,
    )
    vision, text = _configs(path)
    with torch.device("meta"):
        module = Gemma4VisionModel(vision)
        embed = Gemma4MultimodalEmbedder(vision, text)
    return module.eval().requires_grad_(False), embed.eval().requires_grad_(False)


def _assign(module, embed, weights):
    for path, tensor in weights.parameters.items():
        parent, _, attribute = path.rpartition(".")
        setattr(module.get_submodule(parent), attribute,
                torch.nn.Parameter(tensor, requires_grad=False))
    module.std_bias = torch.nn.Buffer(weights.std_bias)
    module.std_scale = torch.nn.Buffer(weights.std_scale)
    embed.embedding_projection.weight = torch.nn.Parameter(weights.projection, requires_grad=False)
    return module, embed


def _input(path, capacity):
    from PIL import Image
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(path, local_files_only=True)
    width, height = 91, 37
    pixels = bytes(component for y in range(height) for x in range(width)
                   for component in ((x * 7 + 31) % 256, (y * 11) % 256, (x + y * 3) % 256))
    result = processor(images=[Image.frombytes("RGB", (width, height), pixels)], return_tensors="pt")
    positions = result["image_position_ids"][0]
    valid = (positions >= 0).all(-1)
    patches = int(valid.sum())
    padded_pixels = torch.zeros((capacity, 768), dtype=torch.float32)
    padded_positions = torch.full((capacity, 2), -1, dtype=torch.int64)
    padded_pixels[:patches] = result["pixel_values"][0, :patches]
    padded_positions[:patches] = positions[:patches]
    return padded_pixels, padded_positions, patches, int(result["num_soft_tokens_per_image"][0])


@CUDA
@torch.no_grad()
def test_vision_encoder_matches_checkpoint(record_property):
    from b12x.preparation import PreparationSession

    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        pytest.fail("assign a CUDA device explicitly")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    path = snapshot()
    manifest = inspect_indexed_checkpoint(path)
    capacity = 2520
    module, embed = _modules(path)
    with DirectWeightLoader(manifest, device,
                            prefixes=("model.vision_tower.", "model.embed_vision.")) as loader:
        weights = VisionWeights.load(module, embed, loader)
    reference, reference_embed = _assign(*_modules(path), weights)
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionRotaryEmbedding
    reference.encoder.rotary_emb = Gemma4VisionRotaryEmbedding(reference.config).to(device)
    pixels, positions, patches, tokens = _input(path, capacity)
    pixels, positions = pixels.to(device), positions.to(device)
    expected = reference_embed(reference(pixels[None], positions[None]).last_hidden_state).float()

    scratch = VisionScratch.allocate(capacity, device, text_hidden=weights.projection.shape[0])
    metadata = VisionMetadata.allocate(capacity, device)
    with PreparationSession(device=device, autotune=False, compile_workers=0) as session:
        prepared = PreparedVision(module, embed, scratch, metadata, weights, session=session)
        session.freeze()
        metadata.position_ids.copy_(positions)
        metadata.cu_seqlens.copy_(torch.tensor([0, patches], dtype=torch.int32, device=device))
        metadata.num_patches.fill_(patches)
        scratch.output.fill_(float("nan"))
        prepared(pixels, positions)
        actual = scratch.output[:tokens].float()
    # Gemma 4 vision is numerically chaotic in BF16: the checkpoint's own
    # BF16 and FP32 encoders differ by about 13% relative L2 on this image,
    # while both feature norms agree within 0.1%. Compare direction, scale,
    # and magnitude rather than demanding bitwise BF16 agreement.
    relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-12))
    cosine = float(torch.nn.functional.cosine_similarity(
        actual.flatten(), expected.flatten(), dim=0))
    norm_ratio = float(actual.norm() / expected.norm().clamp_min(1e-12))
    record_property("numerical_error", json.dumps({
        "relative_l2": relative, "cosine": cosine, "norm_ratio": norm_ratio}))
    assert relative < 0.15, relative
    assert cosine > 0.99, cosine
    assert abs(norm_ratio - 1) < 0.02, norm_ratio
    assert float((actual - expected).abs().max()) < float(expected.abs().max()) * 0.30
