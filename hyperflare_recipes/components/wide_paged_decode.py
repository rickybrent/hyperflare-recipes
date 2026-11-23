# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Split-KV paged decode and verify attention for 512-dim global heads."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import (
    attach_programs,
    compile_only_launches,
    launch_triton,
    load_programs,
)
from b12x._lib.compile_pool import CompileJob
from b12x._lib.program_cache import program_cache
from b12x._lib.scratch import scratch_buffer_spec
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.persistent import persistent_empty
from ignition.preparation.scratch import planned_scratch, prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import native_contract, native_plan
from .selected_attention import metadata_tensor, tensor_facts

_HALF_DIM = 256
_FULL_DIM = 512


@triton.jit
def _wide_paged_decode_split_kernel(
    q_ptr,
    k_cache_ptr,
    v_lo_cache_ptr,
    v_hi_cache_ptr,
    page_table_ptr,
    cache_seqlens_ptr,
    cu_seqlens_q_ptr,
    k_descale_ptr,
    partial_out_ptr,
    partial_lse_ptr,
    stride_q_t,
    stride_q_h,
    stride_k_page,
    stride_k_tok,
    stride_k_h,
    stride_vlo_page,
    stride_vlo_tok,
    stride_vlo_h,
    stride_vhi_page,
    stride_vhi_tok,
    stride_vhi_h,
    stride_pt_b,
    stride_po_t,
    stride_po_s,
    stride_po_h,
    stride_lse_t,
    stride_lse_s,
    softmax_scale,
    NUM_KV_HEADS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    Q_TILES: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    MIN_TILES_PER_SPLIT: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    pid_work = tl.program_id(0)
    split_idx = tl.program_id(1)

    q_tile_idx = pid_work % Q_TILES
    rem = pid_work // Q_TILES
    kv_head_idx = rem % NUM_KV_HEADS
    req_idx = rem // NUM_KV_HEADS

    q_start = tl.load(cu_seqlens_q_ptr + req_idx)
    q_end = tl.load(cu_seqlens_q_ptr + req_idx + 1)
    q_len = q_end - q_start
    seq_len = tl.load(cache_seqlens_ptr + req_idx)

    packed_q = q_tile_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_local = packed_q // GQA_GROUP
    gqa_local = packed_q % GQA_GROUP
    q_head = kv_head_idx * GQA_GROUP + gqa_local
    q_row = q_start + q_local
    q_valid = (q_local < q_len) & (seq_len > 0)

    if q_len <= 0 or q_tile_idx * BLOCK_Q >= q_len * GQA_GROUP:
        return
    if seq_len <= 0:
        lse_offs = (
            q_row.to(tl.int64) * stride_lse_t
            + split_idx.to(tl.int64) * stride_lse_s
            + q_head.to(tl.int64)
        )
        tl.store(partial_lse_ptr + lse_offs, -float("inf"), mask=(q_local < q_len))
        return

    total_tiles = tl.cdiv(seq_len, BLOCK_N)
    active_splits = tl.minimum(
        NUM_SPLITS,
        tl.maximum(1, tl.cdiv(total_tiles, MIN_TILES_PER_SPLIT)),
    )
    if split_idx >= active_splits:
        lse_offs = (
            q_row.to(tl.int64) * stride_lse_t
            + split_idx.to(tl.int64) * stride_lse_s
            + q_head.to(tl.int64)
        )
        tl.store(partial_lse_ptr + lse_offs, -float("inf"), mask=q_valid)
        return

    tiles_per_split = tl.cdiv(total_tiles, active_splits)
    tile_start = split_idx * tiles_per_split
    tile_end = tl.minimum(tile_start + tiles_per_split, total_tiles)
    if tile_start >= tile_end:
        lse_offs = (
            q_row.to(tl.int64) * stride_lse_t
            + split_idx.to(tl.int64) * stride_lse_s
            + q_head.to(tl.int64)
        )
        tl.store(partial_lse_ptr + lse_offs, -float("inf"), mask=q_valid)
        return

    context_len = seq_len - q_len
    causal_hi = context_len + q_local
    causal_hi_min = context_len + (q_tile_idx * BLOCK_Q) // GQA_GROUP
    causal_hi_max = tl.minimum(
        seq_len - 1,
        context_len + ((q_tile_idx + 1) * BLOCK_Q - 1) // GQA_GROUP,
    )
    max_needed_tiles = tl.cdiv(causal_hi_max + 1, BLOCK_N)
    tile_end = tl.minimum(tile_end, max_needed_tiles)
    if tile_start >= tile_end:
        lse_offs = (
            q_row.to(tl.int64) * stride_lse_t
            + split_idx.to(tl.int64) * stride_lse_s
            + q_head.to(tl.int64)
        )
        tl.store(partial_lse_ptr + lse_offs, -float("inf"), mask=q_valid)
        return

    offs_d = tl.arange(0, HALF_DIM)
    q_base = q_row.to(tl.int64)[:, None] * stride_q_t + q_head.to(tl.int64)[:, None] * stride_q_h
    q_lo = tl.load(q_ptr + q_base + offs_d[None, :], mask=q_valid[:, None], other=0.0)
    q_hi = tl.load(q_ptr + q_base + HALF_DIM + offs_d[None, :], mask=q_valid[:, None], other=0.0)

    k_scale = tl.load(k_descale_ptr).to(tl.float32)
    scale_log2 = softmax_scale * k_scale * 1.4426950408889634

    m_i = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc_lo = tl.zeros([BLOCK_Q, HALF_DIM], dtype=tl.float32)
    acc_hi = tl.zeros([BLOCK_Q, HALF_DIM], dtype=tl.float32)

    offs_n = tl.arange(0, BLOCK_N)
    pt_row_ptr = page_table_ptr + req_idx.to(tl.int64) * stride_pt_b
    k_head_off = kv_head_idx.to(tl.int64) * stride_k_h
    vlo_head_off = kv_head_idx.to(tl.int64) * stride_vlo_h
    vhi_head_off = kv_head_idx.to(tl.int64) * stride_vhi_h

    for tile_idx in range(tile_start, tile_end):
        kv_start = tile_idx * BLOCK_N
        page_idx = kv_start // PAGE_SIZE
        page_off = kv_start % PAGE_SIZE
        page_id = tl.load(pt_row_ptr + page_idx).to(tl.int64)

        kv_pos = kv_start + offs_n
        kv_in_bounds = kv_pos < seq_len
        tok_offs = (page_off + offs_n).to(tl.int64)

        k_tile_base = page_id * stride_k_page + tok_offs[:, None] * stride_k_tok + k_head_off
        k_lo = tl.load(
            k_cache_ptr + k_tile_base + offs_d[None, :],
            mask=kv_in_bounds[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        k_hi = tl.load(
            k_cache_ptr + k_tile_base + HALF_DIM + offs_d[None, :],
            mask=kv_in_bounds[:, None],
            other=0.0,
        ).to(tl.bfloat16)

        qk = tl.dot(q_lo, tl.trans(k_lo))
        qk = tl.dot(q_hi, tl.trans(k_hi), acc=qk)
        qk = qk * scale_log2

        if kv_start + BLOCK_N - 1 > causal_hi_min:
            valid_kv = kv_in_bounds[None, :] & (kv_pos[None, :] <= causal_hi[:, None])
            qk = tl.where(valid_kv, qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        has_new = m_new > -float("inf")
        m_new_safe = tl.where(has_new, m_new, 0.0)
        alpha = tl.where(m_i > -float("inf"), tl.math.exp2(m_i - m_new_safe), 0.0)
        p = tl.where(qk > -float("inf"), tl.math.exp2(qk - m_new_safe[:, None]), 0.0)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc_lo = acc_lo * alpha[:, None]
        acc_hi = acc_hi * alpha[:, None]
        p_bf16 = p.to(tl.bfloat16)

        vlo_tile_base = page_id * stride_vlo_page + tok_offs[:, None] * stride_vlo_tok + vlo_head_off
        v_lo = tl.load(
            v_lo_cache_ptr + vlo_tile_base + offs_d[None, :],
            mask=kv_in_bounds[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        acc_lo = tl.dot(p_bf16, v_lo, acc=acc_lo)

        vhi_tile_base = page_id * stride_vhi_page + tok_offs[:, None] * stride_vhi_tok + vhi_head_off
        v_hi = tl.load(
            v_hi_cache_ptr + vhi_tile_base + offs_d[None, :],
            mask=kv_in_bounds[:, None],
            other=0.0,
        ).to(tl.bfloat16)
        acc_hi = tl.dot(p_bf16, v_hi, acc=acc_hi)

        m_i = m_new

    has_mass = l_i > 0.0
    inv_l = tl.where(has_mass, 1.0 / l_i, 0.0)

    po_base = (
        q_row.to(tl.int64)[:, None] * stride_po_t
        + split_idx.to(tl.int64) * stride_po_s
        + q_head.to(tl.int64)[:, None] * stride_po_h
    )
    tl.store(
        partial_out_ptr + po_base + offs_d[None, :],
        acc_lo * inv_l[:, None],
        mask=q_valid[:, None],
    )
    tl.store(
        partial_out_ptr + po_base + HALF_DIM + offs_d[None, :],
        acc_hi * inv_l[:, None],
        mask=q_valid[:, None],
    )
    lse = tl.where(
        has_mass,
        m_i + tl.math.log2(tl.where(has_mass, l_i, 1.0)),
        -float("inf"),
    )
    lse_offs = (
        q_row.to(tl.int64) * stride_lse_t
        + split_idx.to(tl.int64) * stride_lse_s
        + q_head.to(tl.int64)
    )
    tl.store(partial_lse_ptr + lse_offs, lse, mask=q_valid)


@triton.jit
def _wide_paged_decode_merge_kernel(
    partial_out_ptr,
    partial_lse_ptr,
    cu_seqlens_q_ptr,
    v_descale_ptr,
    output_ptr,
    stride_po_t,
    stride_po_s,
    stride_po_h,
    stride_lse_t,
    stride_lse_s,
    stride_o_t,
    stride_o_h,
    BATCH: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    SPLIT_PAD: tl.constexpr,
    HALF_DIM: tl.constexpr,
):
    row_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    half_idx = tl.program_id(2)

    offs_d = tl.arange(0, HALF_DIM)
    d_offset = half_idx * HALF_DIM
    o_base = row_idx.to(tl.int64) * stride_o_t + head_idx.to(tl.int64) * stride_o_h + d_offset

    total_live_q = tl.load(cu_seqlens_q_ptr + BATCH)
    if row_idx >= total_live_q:
        tl.store(output_ptr + o_base + offs_d, tl.zeros([HALF_DIM], dtype=tl.bfloat16))
        return

    offs_s = tl.arange(0, SPLIT_PAD)
    lse_base = row_idx.to(tl.int64) * stride_lse_t + head_idx.to(tl.int64)
    lse_vec = tl.load(
        partial_lse_ptr + lse_base + offs_s.to(tl.int64) * stride_lse_s,
        mask=offs_s < NUM_SPLITS,
        other=-float("inf"),
    )
    m_max = tl.max(lse_vec, axis=0)
    if m_max == -float("inf"):
        tl.store(output_ptr + o_base + offs_d, tl.zeros([HALF_DIM], dtype=tl.bfloat16))
        return

    acc = tl.zeros([HALF_DIM], dtype=tl.float32)
    w_sum = tl.zeros([], dtype=tl.float32)
    po_base = row_idx.to(tl.int64) * stride_po_t + head_idx.to(tl.int64) * stride_po_h + d_offset

    for s in range(NUM_SPLITS):
        lse_s = tl.load(partial_lse_ptr + lse_base + tl.cast(s, tl.int64) * stride_lse_s)
        if lse_s > -float("inf"):
            w_s = tl.math.exp2(lse_s - m_max)
            val = tl.load(
                partial_out_ptr + po_base + tl.cast(s, tl.int64) * stride_po_s + offs_d
            )
            acc += w_s * val
            w_sum += w_s

    v_scale = tl.load(v_descale_ptr).to(tl.float32)
    inv_w = tl.where(w_sum > 0.0, v_scale / w_sum, 0.0)
    tl.store(output_ptr + o_base + offs_d, (acc * inv_w).to(tl.bfloat16))


@dataclass(frozen=True)
class Query:
    operands: tuple
    width: int
    softmax_scale: float

    @property
    def tokens(self) -> int:
        return self.operands[0][0][0]

    @property
    def num_q_heads(self) -> int:
        return self.operands[0][0][1]

    @property
    def num_kv_heads(self) -> int:
        return self.operands[1][0][2]

    @property
    def page_size(self) -> int:
        return self.operands[1][0][1]

    @property
    def batch(self) -> int:
        return self.operands[4][0][0]

    @property
    def gqa_group(self) -> int:
        return self.num_q_heads // self.num_kv_heads

    def scratch_bytes(self, tactic: tuple[int, int, int, int]) -> int:
        num_splits = tactic[0]
        if num_splits < 2:
            raise ValueError("wide paged decode requires at least two KV splits")
        # FP32 partial_out [tokens, num_splits, heads, 512] + partial_lse [tokens, num_splits, heads].
        return self.tokens * num_splits * self.num_q_heads * (_FULL_DIM + 1) * 4


def _tactics(query: Query) -> tuple[tuple[int, int, int, int], ...]:
    """Candidate (num_splits, block_q, block_n, num_warps) tactics."""
    max_packed_q = query.width * query.gqa_group
    is_bf16_kv = query.operands[1][2] == "bfloat16"
    # BLOCK_Q=32 with BLOCK_N=64 (FP8) or any BLOCK_N=64 (BF16) exceeds SM120's 99 KiB shared memory
    # at num_stages=2; keep BF16 KV tactics at BLOCK_N<=32.
    if is_bf16_kv:
        block_configs = (
            ((16, 32, 4), (16, 16, 4), (16, 32, 8))
            if max_packed_q <= 16
            else ((32, 32, 8), (16, 32, 4), (32, 32, 4), (16, 16, 4))
        )
    else:
        block_configs = (
            ((16, 64, 4), (16, 32, 4), (16, 64, 8))
            if max_packed_q <= 16
            else ((32, 32, 8), (16, 64, 4), (16, 32, 4), (32, 32, 4))
        )
    candidates = []
    for block_q, block_n, warps in block_configs:
        if query.page_size % block_n != 0:
            continue
        q_tiles = (max_packed_q + block_q - 1) // block_q
        base_ctas = query.batch * query.num_kv_heads * q_tiles
        raw_splits = (
            (64, 32, 96, 16)
            if base_ctas <= 4
            else (32, 16, 64, 8)
            if base_ctas <= 16
            else (16, 8, 32, 4)
        )
        for splits in raw_splits:
            candidates.append((splits, block_q, block_n, warps))
    return tuple(candidates)


CONTRACT = native_contract(
    component_id="flare.wide_paged_decode",
    query_type=Query,
    tactics=_tactics,
    semantic_version=1,
)


def _split_views(scratch: torch.Tensor, tokens: int, num_splits: int, heads: int):
    if num_splits < 2:
        raise ValueError("wide paged decode requires at least two KV splits")
    values = scratch.view(torch.float32)
    po_elems = tokens * num_splits * heads * _FULL_DIM
    lse_elems = tokens * num_splits * heads
    partial_out = values[:po_elems].view(tokens, num_splits, heads, _FULL_DIM)
    partial_lse = values[po_elems : po_elems + lse_elems].view(tokens, num_splits, heads)
    return partial_out, partial_lse


def _split_args(tensors, partial_out, partial_lse, query: Query, tactic: tuple[int, int, int, int]):
    q, k_cache, v_lo_cache, v_hi_cache, page_table, cache_seqlens, cu_seqlens_q, k_descale, _v_descale, _output = tensors
    num_splits, block_q, block_n, _ = tactic
    q_tiles = (query.width * query.gqa_group + block_q - 1) // block_q
    min_tiles = max(1, query.page_size // block_n)
    args = (
        q,
        k_cache,
        v_lo_cache,
        v_hi_cache,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        k_descale,
        partial_out,
        partial_lse,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_lo_cache.stride(0),
        v_lo_cache.stride(1),
        v_lo_cache.stride(2),
        v_hi_cache.stride(0),
        v_hi_cache.stride(1),
        v_hi_cache.stride(2),
        page_table.stride(0),
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        query.softmax_scale,
    )
    constexprs = dict(
        NUM_KV_HEADS=query.num_kv_heads,
        GQA_GROUP=query.gqa_group,
        Q_TILES=q_tiles,
        PAGE_SIZE=query.page_size,
        NUM_SPLITS=num_splits,
        MIN_TILES_PER_SPLIT=min_tiles,
        BLOCK_Q=block_q,
        BLOCK_N=block_n,
        HALF_DIM=_HALF_DIM,
    )
    grid = (query.batch * query.num_kv_heads * q_tiles, num_splits, 1)
    return grid, args, constexprs


def _merge_args(tensors, partial_out, partial_lse, query: Query, num_splits: int):
    _, _, _, _, _, _, cu_seqlens_q, _, v_descale, output = tensors
    args = (
        partial_out,
        partial_lse,
        cu_seqlens_q,
        v_descale,
        output,
        partial_out.stride(0),
        partial_out.stride(1),
        partial_out.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        output.stride(0),
        output.stride(1),
    )
    constexprs = dict(
        BATCH=query.batch,
        NUM_SPLITS=num_splits,
        SPLIT_PAD=triton.next_power_of_2(num_splits),
        HALF_DIM=_HALF_DIM,
    )
    grid = (query.tokens, query.num_q_heads, 2)
    return grid, args, constexprs


@program_cache(scope="preparation")
def compile_attention(payload, tactic, ordinal):
    query = Query(**dict(payload))
    tensors = tuple(metadata_tensor(facts) for facts in query.operands)
    num_splits, _block_q, _block_n, num_warps = tactic
    scratch = torch.empty((max(1, query.scratch_bytes(tactic)),), dtype=torch.uint8, device="meta")
    partial_out, partial_lse = _split_views(scratch, query.tokens, num_splits, query.num_q_heads)
    with torch.cuda.device(ordinal), compile_only_launches():
        s_grid, s_args, s_const = _split_args(tensors, partial_out, partial_lse, query, tactic)
        split_prog = launch_triton(
            _wide_paged_decode_split_kernel,
            s_grid[:2],
            *s_args,
            **s_const,
            num_warps=num_warps,
            num_stages=2,
        )
        programs = {"split": split_prog}
        m_grid, m_args, m_const = _merge_args(tensors, partial_out, partial_lse, query, num_splits)
        merge_prog = launch_triton(
            _wide_paged_decode_merge_kernel,
            m_grid,
            *m_args,
            **m_const,
            num_warps=4,
        )
        programs["merge"] = merge_prog
        return programs


@dataclass(frozen=True)
class State:
    query: Query
    tactic: tuple[int, int, int, int]
    programs: dict

    def run(self, tensors, scratch: torch.Tensor):
        num_splits = self.tactic[0]
        partial_out, partial_lse = _split_views(
            scratch, self.query.tokens, num_splits, self.query.num_q_heads
        )
        s_grid, s_args, _ = _split_args(tensors, partial_out, partial_lse, self.query, self.tactic)
        self.programs["split"][s_grid](*s_args)
        m_grid, m_args, _ = _merge_args(tensors, partial_out, partial_lse, self.query, num_splits)
        self.programs["merge"][m_grid](*m_args)


@torch.library.custom_op("flare::wide_paged_decode", mutates_args=("output", "scratch"))
def _execute(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_lo_cache: torch.Tensor,
    v_hi_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    output: torch.Tensor,
    scratch: torch.Tensor,
    handle: int,
) -> None:
    plan = plan_from_handle(handle)
    state = require_prepared(plan, plan.component_id, q.device)
    state.run(
        (q, k_cache, v_lo_cache, v_hi_cache, page_table, cache_seqlens, cu_seqlens_q, k_descale, v_descale, output),
        scratch,
    )


@_execute.register_fake
def _fake(
    q, k_cache, v_lo_cache, v_hi_cache, page_table, cache_seqlens, cu_seqlens_q,
    k_descale, v_descale, output, scratch, handle,
):
    pass


class PreparedWidePagedDecode:
    """Prepared split-KV paged decode/verify attention for 512-dim K and split 256+256 V caches."""

    def __init__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_lo_cache: torch.Tensor,
        v_hi_cache: torch.Tensor,
        page_table: torch.Tensor,
        cache_seqlens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        output: torch.Tensor,
        *,
        width: int,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        session,
        workspace,
        name: str,
        tuning_page_count: int | None = None,
    ):
        if (
            q.ndim != 3
            or output.shape != q.shape
            or q.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
            or q.shape[-1] != _FULL_DIM
            or not q.is_contiguous()
            or not output.is_contiguous()
        ):
            raise ValueError("wide paged decode requires contiguous BF16 query and output with head_dim=512")
        if (
            k_cache.ndim != 4
            or v_lo_cache.ndim != 4
            or v_hi_cache.ndim != 4
            or k_cache.shape[:3] != v_lo_cache.shape[:3]
            or k_cache.shape[:3] != v_hi_cache.shape[:3]
            or k_cache.shape[3] != _FULL_DIM
            or v_lo_cache.shape[3] != _HALF_DIM
            or v_hi_cache.shape[3] != _HALF_DIM
            or k_cache.dtype not in (torch.float8_e4m3fn, torch.bfloat16)
            or v_lo_cache.dtype != k_cache.dtype
            or v_hi_cache.dtype != k_cache.dtype
            or k_cache.stride(3) != 1
            or v_lo_cache.stride(3) != 1
            or v_hi_cache.stride(3) != 1
        ):
            raise ValueError("wide paged decode requires matching FP8/BF16 K(512) and split V(256+256) paged caches")
        if q.shape[1] % k_cache.shape[2] != 0:
            raise ValueError("query heads must be divisible by KV heads")
        if page_table.ndim != 2 or cache_seqlens.ndim != 1 or cu_seqlens_q.ndim != 1:
            raise ValueError("invalid paged metadata ranks")
        if page_table.shape[0] != cache_seqlens.shape[0] or cu_seqlens_q.shape[0] != page_table.shape[0] + 1:
            raise ValueError("mismatched batch dimensions in paged metadata")
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        if tuning_page_count is None:
            tuning_page_count = min(k_cache.shape[0], v_lo_cache.shape[0], v_hi_cache.shape[0])
        if type(tuning_page_count) is not int or not 0 < tuning_page_count <= k_cache.shape[0]:
            raise ValueError("tuning_page_count must be positive and within cache capacity")
        self.tuning_page_count = tuning_page_count

        if k_descale is None:
            k_descale = persistent_empty((1,), torch.float32, q.device).fill_(1.0)
        if v_descale is None:
            v_descale = persistent_empty((1,), torch.float32, q.device).fill_(1.0)
        if softmax_scale is None:
            softmax_scale = _FULL_DIM ** -0.5

        self.arguments = (
            q,
            k_cache,
            v_lo_cache,
            v_hi_cache,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            k_descale,
            v_descale,
            output,
        )
        self.workspace = workspace
        query = Query(
            operands=tuple(tensor_facts(t) for t in self.arguments),
            width=width,
            softmax_scale=float(softmax_scale),
        )
        payload = CONTRACT.encode_query(query)
        device = q.device

        def materialize(tactic):
            programs = compile_attention(payload, tactic, device.index)
            load_programs(programs)
            return attach_programs(State(query, tactic, programs), *programs.values())

        self.plan = native_plan(
            contract=CONTRACT,
            query=query,
            device=device,
            scratch_bytes=query.scratch_bytes,
            materialize=materialize,
            compile_jobs=lambda t, d: (
                CompileJob.create(
                    "hyperflare_recipes.components.wide_paged_decode:compile_attention",
                    payload,
                    t,
                    d.ordinal,
                ),
            ),
        )

        declaration = self.arguments

        def call(state, benchmark=False):
            args = declaration
            produce = None
            if benchmark:
                def inputs():
                    dq, kc, vlo, vhi, pt, lens, cu, kd, vd, out = declaration
                    bq = torch.empty_like(dq).normal_(std=0.2)
                    bpt = torch.empty_like(pt)
                    blens = torch.empty_like(lens)
                    bcu = torch.empty_like(cu)
                    bout = torch.zeros_like(out)
                    # Tune at a realistic long context (up to 32K) wrapping over backed pages.
                    extent = min(32768, pt.shape[1] * kc.shape[1])
                    pages_per_row = triton.cdiv(extent, kc.shape[1])
                    page_ids = (
                        torch.arange(pt.shape[0], device=pt.device, dtype=torch.int64)[:, None] * pages_per_row
                        + torch.arange(pt.shape[1], device=pt.device, dtype=torch.int64)
                    )
                    bpt.copy_((page_ids % self.tuning_page_count).to(pt.dtype))
                    blens.fill_(extent)
                    bcu.copy_(
                        (torch.arange(cu.numel(), device=cu.device, dtype=torch.int64) * width).clamp_max(
                            dq.shape[0]
                        ).to(cu.dtype)
                    )
                    return (bq, kc, vlo, vhi, bpt, blens, bcu, kd, vd, bout), bq.clone()

                args, seed = benchmark_resource(session, ("wide_paged_decode", name), inputs)
                produce = lambda: args[0].copy_(seed)

            nbytes = query.scratch_bytes(state.tactic)
            scratch = planned_scratch(
                (scratch_buffer_spec(CONTRACT.component_id, nbytes=max(1, nbytes), device=device),),
                workspace,
            )
            return PreparedCall(
                run=lambda: state.run(args, scratch),
                output=args[-1],
                produce=produce,
                owners=prime_owners(workspace, scratch, args),
            )

        declare(
            session,
            (
                self.plan.request(
                    name=name,
                    prepare_call=lambda state: replace(call(state, True), owners=()),
                    benchmark_call=lambda state: call(state, True),
                ),
            ),
        )
        register_finalizer(session, lambda: self._bind(workspace), workspace=workspace)

    def _bind(self, workspace):
        specs = self.plan.scratch_specs()
        self.scratch = planned_scratch(specs, workspace) if specs else torch.empty(1, dtype=torch.uint8, device=self.arguments[0].device)

    def functional(self, *, session):
        _, kc, vlo, vhi, _, _, _, kd, vd, out = self.arguments
        self.caches = (kc, vlo, vhi)
        self.descales = (kd, vd)
        self.output_shape = tuple(out.shape)
        self.output_dtype = out.dtype
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        specs = self.plan.scratch_specs()
        self.scratch_layout = (
            tuple((tuple(spec.shape), spec.dtype) for spec in specs)
            if specs
            else (((1,), torch.uint8),)
        )
        self.arguments = self.scratch = None

    def __call__(self, q=None, *, page_table=None, cache_seqlens=None, cu_seqlens_q=None):
        if self.arguments is not None:
            _execute(*self.arguments, self.scratch, self.plan.handle)
            return self.arguments[-1]
        output = torch.empty(self.output_shape, dtype=self.output_dtype, device=q.device)
        scratch, = (torch.empty(shape, dtype=dtype, device=q.device) for shape, dtype in self.scratch_layout)
        _execute(
            q,
            *self.caches,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            *self.descales,
            output,
            scratch,
            self.plan.handle,
        )
        return output
