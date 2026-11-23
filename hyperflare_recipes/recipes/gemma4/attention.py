# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 sliding-window, full-context, and KV-shared paged attention."""

import math
from typing import NamedTuple
import torch

from hyperflare_recipes.components.collectives import PreparedAllReduce
from hyperflare_recipes.components.linear import QuantizedProjection, UnquantizedProjection
from hyperflare_recipes.components.paged_attention import PreparedPagedAttention, PreparedPagedKVWriter
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.tensor_fp8 import TensorFP8Projection, TensorFP8Weight
from hyperflare_recipes.components.window_attention import PreparedWindowAttention, commit_circular_kv
from hyperflare_recipes.components.window_staging import PreparedWindowStaging
from ignition.preparation.examples import declaration_tensor
from ignition.preparation.persistent import persistent_copy


def prepare_projection(source, weight, output, *, session, name, workspace=None):
    if isinstance(weight, torch.Tensor):
        return UnquantizedProjection(source, weight, output, session=session, name=name)
    if isinstance(weight, TensorFP8Weight):
        return TensorFP8Projection(source, weight, output, session=session, name=name)
    return QuantizedProjection(
        source, weight, output, activation_mode="auto", session=session, workspace=workspace, name=name
    )


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).to(x.dtype)


def rms_norm_no_scale(x: torch.Tensor, eps: float) -> torch.Tensor:
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(x.dtype)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    rotated = torch.cat((-right, left), dim=-1)
    return (x * cos[:, None, :]) + (rotated * sin[:, None, :])


class CacheBindings(NamedTuple):
    slots: torch.Tensor
    lengths: torch.Tensor


def cache_bindings(metadata):
    from .config import PAGE_TOKENS

    m = metadata
    columns = (m.positions.clamp_min(0) // PAGE_TOKENS).clamp(max=m.token_page_table.shape[1] - 1)
    rows = torch.arange(m.positions.numel(), device=m.positions.device)
    pages = m.token_page_table[rows, columns]
    slots = torch.where((m.positions >= 0) & (pages >= 0),
        pages.long() * PAGE_TOKENS + m.positions.remainder(PAGE_TOKENS), -1)
    lengths = torch.where(m.lengths > 0, (m.starts + m.lengths).int(), 0)
    return CacheBindings(slots, lengths)


class PreparedAttention:
    def __init__(
        self,
        layer: int,
        weights: dict,
        config,
        source: torch.Tensor,
        metadata,
        pool,
        *,
        shard: TensorParallelShard = TensorParallelShard(0, 1),
        runtime=None,
        coordinator=None,
        mode: str,
        session,
        name: str,
        workspace,
        examples,
        cache_metadata,
        unit_scale: torch.Tensor,
        k_descale: torch.Tensor,
        frozen_kv: bool = False,
    ):
        self.layer = layer
        self.config = config
        self.mode = mode
        self.frozen_kv = frozen_kv
        self.eps = config.rms_norm_eps
        self.is_sliding = config.is_sliding(layer)
        self.shared_kv = config.is_kv_shared(layer)
        self.is_sliding_donor = self.is_sliding and not self.shared_kv and layer == config.sliding_donor_layer
        self.window = config.sliding_window
        self.heads = config.local_heads(shard)
        self.kv_heads = config.local_kv_heads(layer, shard)
        self.dim = config.layer_head_dim(layer)
        tokens = source.shape[0]

        batch = metadata.lengths.numel()
        width = 1 if mode == "prefill" else tokens // batch
        gqa_group = self.heads // self.kv_heads
        q_cap = (
            tokens
            if self.is_sliding or mode == "prefill"
            else batch * max(width, (32 // gqa_group) + 1, 2)
        )

        def zeros(role, shape, dtype=torch.bfloat16):
            return declaration_tensor(examples, "attention." + role, shape, dtype, source.device)

        self.q_norm = weights["q_norm"]
        self.q_raw = zeros("q_raw", (tokens, self.heads * self.dim))
        self.q_full = zeros("q_full", (q_cap, self.heads, self.dim))
        self.q = self.q_full[:tokens]
        self.attended = zeros("attended", (tokens, self.heads * self.dim))
        self.partial_output = zeros("partial_output", (tokens, config.hidden))
        self.output = zeros("output", (tokens, config.hidden))
        self.q_proj = prepare_projection(
            source, weights["q_proj"], self.q_raw, session=session, name=name + ".q_proj", workspace=workspace
        )
        self.o_proj = prepare_projection(
            self.attended,
            weights["o_proj"],
            self.partial_output,
            session=session,
            name=name + ".o_proj",
            workspace=workspace,
        )
        self.reduce = PreparedAllReduce(
            self.partial_output,
            self.output,
            runtime=runtime,
            session=session,
            coordinator=coordinator,
            name=name + ".reduce",
        )

        self.k_proj = None
        self.v_proj = None
        if not self.shared_kv:
            self.k_norm = weights["k_norm"]
            self.k_raw = zeros("k_raw", (tokens, self.kv_heads * self.dim))
            self.key = zeros("key", (tokens, self.kv_heads, self.dim))
            self.k_proj = prepare_projection(
                source, weights["k_proj"], self.k_raw, session=session, name=name + ".k_proj", workspace=workspace
            )
            if "v_proj" in weights:
                self.v_raw = zeros("v_raw", (tokens, self.kv_heads * self.dim))
                self.v_proj = prepare_projection(
                    source, weights["v_proj"], self.v_raw, session=session, name=name + ".v_proj", workspace=workspace
                )
            else:
                self.v_raw = self.k_raw

        self.staging = None
        self.staged_pages = None
        self.history_key = None
        self.history_value = None
        if self.is_sliding:
            history_key, history_value = pool.attention(layer, config, shard=shard)
            self.history_key, self.history_value = history_key, history_value
            self.writer = None
            if mode == "prefill":
                from .config import PAGE_TOKENS

                pages = (tokens + self.window + PAGE_TOKENS - 1) // PAGE_TOKENS
                self.staged_key = zeros("staged_key", (pages, PAGE_TOKENS, self.kv_heads, self.dim), dtype=torch.float8_e4m3fn)
                self.staged_value = zeros("staged_value", (pages, PAGE_TOKENS, self.kv_heads, self.dim), dtype=torch.float8_e4m3fn)
                self.staged_lengths = zeros("staged_lengths", (batch,), dtype=torch.int32)
                self.staged_pages = persistent_copy(
                    torch.arange(pages, dtype=torch.int32, device=source.device).view(batch, -1)
                )
                if not self.shared_kv:
                    self.value = zeros("value", (tokens, self.kv_heads, self.dim))
                    self.staging = PreparedWindowStaging(
                        history_key,
                        history_value,
                        self.key,
                        self.value,
                        self.staged_key,
                        self.staged_value,
                        metadata,
                        session=session,
                        name=name + ".window_stage",
                    )
                self.attention = PreparedPagedAttention(
                    self.q,
                    self.staged_key,
                    self.staged_value,
                    self.staged_pages,
                    self.staged_lengths,
                    metadata.query_start_loc,
                    self.attended.view(tokens, self.heads, self.dim),
                    mode="prefill",
                    window=self.window - 1,
                    width=1,
                    session=session,
                    name=name + ".attention",
                    workspace=workspace,
                    caller_owned_placeholders=True,
                    tuning_page_count=pages,
                    k_descale=k_descale,
                    v_descale=unit_scale,
                )
            else:
                if not self.shared_kv:
                    self.value = zeros("value", (tokens, self.kv_heads, self.dim))
                    key_decl, value_decl = self.key, self.value
                else:
                    self.shared_key = zeros("shared_key", (tokens, self.kv_heads, self.dim))
                    self.shared_value = zeros("shared_value", (tokens, self.kv_heads, self.dim))
                    key_decl, value_decl = self.shared_key, self.shared_value
                self.attention = PreparedWindowAttention(
                    self.q,
                    key_decl,
                    value_decl,
                    history_key,
                    history_value,
                    metadata,
                    self.attended.view(tokens, self.heads, self.dim),
                    softmax_scale=1.0,
                    session=session,
                    workspace=workspace,
                    name=name + ".attention",
                )
        else:
            key_cache, value_lo_cache, value_hi_cache = pool.attention(layer, config, shard=shard)
            half_dim = self.dim // 2
            self.writer_v = None
            self.writer_k = None
            if not self.shared_kv:
                self.value_lo = zeros("value_lo", (tokens, self.kv_heads, half_dim))
                self.value_hi = zeros("value_hi", (tokens, self.kv_heads, half_dim))
                self.writer_v = PreparedPagedKVWriter(
                    self.value_lo,
                    self.value_hi,
                    value_lo_cache,
                    value_hi_cache,
                    cache_metadata.slots,
                    session=session,
                    name=name + ".cache_write_v",
                    k_scale=unit_scale,
                    v_scale=unit_scale,
                    caller_owned_placeholders=True,
                )
                self.writer_k = PreparedPagedKVWriter(
                    self.key,
                    self.value_hi,
                    key_cache,
                    value_hi_cache,
                    cache_metadata.slots,
                    session=session,
                    name=name + ".cache_write_k",
                    k_scale=unit_scale,
                    v_scale=unit_scale,
                    caller_owned_placeholders=True,
                )
            self.attended_lo_full = zeros("attended_lo_full", (q_cap, self.heads, half_dim))
            self.attended_hi_full = zeros("attended_hi_full", (q_cap, self.heads, half_dim))
            self.attended_lo = self.attended_lo_full[:tokens]
            self.attended_hi = self.attended_hi_full[:tokens]
            self.attention_lo = PreparedPagedAttention(
                self.q_full,
                key_cache,
                value_lo_cache,
                metadata.page_table,
                cache_metadata.lengths,
                metadata.query_start_loc,
                self.attended_lo_full,
                mode="prefill",
                window=-1,
                width=1,
                session=session,
                name=name + ".attention_lo",
                workspace=workspace,
                caller_owned_placeholders=True,
                tuning_page_count=int(pool.memory_capacity[1].item()),
                k_descale=k_descale,
                v_descale=unit_scale,
            )
            self.attention_hi = PreparedPagedAttention(
                self.q_full,
                key_cache,
                value_hi_cache,
                metadata.page_table,
                cache_metadata.lengths,
                metadata.query_start_loc,
                self.attended_hi_full,
                mode="prefill",
                window=-1,
                width=1,
                session=session,
                name=name + ".attention_hi",
                workspace=workspace,
                caller_owned_placeholders=True,
                tuning_page_count=int(pool.memory_capacity[1].item()),
                k_descale=k_descale,
                v_descale=unit_scale,
            )

        self.q_cap = q_cap
        for projection in (self.q_proj, self.o_proj, self.k_proj, self.v_proj):
            if projection is not None:
                projection.functional(session=session)
        self.reduce.functional(session=session)
        if self.staging is not None:
            self.staging = self.staging.functional(session=session)
        for field in ("writer", "writer_v", "writer_k"):
            operation = getattr(self, field, None)
            if operation is not None:
                setattr(self, field, operation.functional())
        if getattr(self, "attention", None) is not None:
            if isinstance(self.attention, PreparedWindowAttention):
                self.attention = self.attention.functional(session=session)
            else:
                self.attention = self.attention.functional(
                    session=session,
                    k_cache=None,
                    v_cache=None,
                    page_table=None,
                    cache_seqlens=None,
                    cu_seqlens_q=None,
                )
        for field in ("attention_lo", "attention_hi"):
            operation = getattr(self, field, None)
            if operation is not None:
                setattr(self, field, operation.functional(session=session, page_table=None,
                    cache_seqlens=None, cu_seqlens_q=None))
        for field in ("q_raw", "q_full", "q", "attended", "partial_output", "output", "k_raw", "key",
                      "v_raw", "value", "value_lo", "value_hi", "attended_lo_full", "attended_hi_full",
                      "attended_lo", "attended_hi", "staged_key", "staged_value", "staged_lengths",
                      "shared_key", "shared_value"):
            if hasattr(self, field):
                setattr(self, field, None)

    def __call__(self, source, cos, sin, *, metadata, cache_metadata, sliding_donor=None, deferred_sliding=None):
        tokens = source.shape[0]
        raw = self.q_proj(source)
        q = rms_norm(raw.view(tokens, self.heads, self.dim), self.q_norm, self.eps)
        query = apply_rotary_pos_emb(q, cos, sin)
        if self.is_sliding:
            if self.mode == "prefill":
                if not self.shared_kv:
                    k_raw = self.k_proj(source)
                    v_raw = k_raw if self.v_proj is None else self.v_proj(source)
                    k = rms_norm(k_raw.view(tokens, self.kv_heads, self.dim), self.k_norm, self.eps)
                    key = apply_rotary_pos_emb(k, cos, sin)
                    value = rms_norm_no_scale(v_raw.view(tokens, self.kv_heads, self.dim), self.eps)
                    staged_key, staged_value = self.staging(key, value, metadata=metadata)
                else:
                    staged_key, staged_value = sliding_donor[0], sliding_donor[1]
                staged_lengths = torch.where(
                    metadata.lengths > 0, metadata.starts.clamp(max=self.window).int() + metadata.lengths, 0
                )
                attended = self.attention(
                    query,
                    k_cache=staged_key,
                    v_cache=staged_value,
                    page_table=self.staged_pages,
                    cache_seqlens=staged_lengths,
                    cu_seqlens_q=metadata.query_start_loc,
                )
                if not self.shared_kv:
                    commit_circular_kv(key, value, self.history_key, self.history_value, metadata, metadata.lengths)
                if self.is_sliding_donor and sliding_donor is not None:
                    sliding_donor[:] = [staged_key, staged_value]
            else:
                if not self.shared_kv:
                    k_raw = self.k_proj(source)
                    v_raw = k_raw if self.v_proj is None else self.v_proj(source)
                    k = rms_norm(k_raw.view(tokens, self.kv_heads, self.dim), self.k_norm, self.eps)
                    key = apply_rotary_pos_emb(k, cos, sin)
                    value = rms_norm_no_scale(v_raw.view(tokens, self.kv_heads, self.dim), self.eps)
                elif self.frozen_kv:
                    key = torch.zeros((tokens, self.kv_heads, self.dim), dtype=query.dtype, device=query.device)
                    value = key
                else:
                    key, value = sliding_donor[0], sliding_donor[1]
                attended = self.attention(query, key, value, metadata)
                if not self.shared_kv:
                    if self.mode == "verify" and deferred_sliding is not None:
                        deferred_sliding.append((self.history_key, self.history_value, key, value))
                    else:
                        commit_circular_kv(key, value, self.history_key, self.history_value, metadata, metadata.lengths)
                if self.is_sliding_donor and sliding_donor is not None:
                    sliding_donor[:] = [key, value]
            return self.reduce(self.o_proj(attended.reshape(tokens, -1)))
        if not self.shared_kv:
            k_raw = self.k_proj(source)
            v_raw = k_raw if self.v_proj is None else self.v_proj(source)
            k = rms_norm(k_raw.view(tokens, self.kv_heads, self.dim), self.k_norm, self.eps)
            key = apply_rotary_pos_emb(k, cos, sin)
            value = rms_norm_no_scale(v_raw.view(tokens, self.kv_heads, self.dim), self.eps)
            half_dim = self.dim // 2
            lo, hi = value[:, :, :half_dim].contiguous(), value[:, :, half_dim:].contiguous()
            self.writer_v(lo, hi, cache_metadata.slots)
            self.writer_k(key, hi, cache_metadata.slots)
        arguments = dict(page_table=metadata.page_table, cache_seqlens=cache_metadata.lengths,
                         cu_seqlens_q=metadata.query_start_loc)
        full_query = torch.zeros((self.q_cap, self.heads, self.dim), dtype=query.dtype, device=query.device)
        full_query[:tokens].copy_(query)
        lo = self.attention_lo(full_query, **arguments)[:tokens]
        hi = self.attention_hi(full_query, **arguments)[:tokens]
        attended = torch.cat((lo, hi), dim=-1)
        return self.reduce(self.o_proj(attended.reshape(tokens, -1)))


def attention_descales(config, device):
    unit = torch.ones(1, dtype=torch.float32, device=device)
    sliding_descale = torch.tensor([math.sqrt(config.text["head_dim"])], dtype=torch.float32, device=device)
    full_descale = torch.tensor([math.sqrt(config.text["global_head_dim"])], dtype=torch.float32, device=device)
    return tuple(persistent_copy(value) for value in (unit, sliding_descale, full_descale))
