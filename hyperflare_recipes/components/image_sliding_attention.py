# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Sparse paged-cache sliding-window attention for bidirectional image spans."""

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
from b12x.preparation import PreparedCall
from b12x.preparation.types import plan_from_handle, require_prepared

from ignition.preparation.persistent import persistent_empty
from ignition.preparation.scratch import prime_owners
from ignition.preparation.tuning import benchmark_resource, declare, register_finalizer

from .native_operator import native_contract, native_plan
from .selected_attention import metadata_tensor, tensor_facts


def compute_image_sliding_spans(
    token_ids: torch.Tensor,
    positions: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    image_token: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-row image block end positions and owning request indices.

    Returns ``(image_span_end, row_request)``, each int32 with shape ``(tokens,)``.
    Non-image and padded rows receive ``-1`` in ``image_span_end``; each image
    row receives the exclusive sequence position of the end of its contiguous
    same-request image run. Pure tensor operations keep this CUDA-graph safe.
    """
    tokens = token_ids.shape[0]
    device = token_ids.device
    rows = torch.arange(tokens, device=device, dtype=torch.int32)
    req = (rows[:, None] >= cu_seqlens_q[None, 1:]).sum(dim=1, dtype=torch.int32)
    live = (rows < cu_seqlens_q[-1]) & (positions >= 0)
    row_request = torch.where(live, req, -1)

    is_img = live & (token_ids == image_token)
    if tokens == 1:
        span_end = torch.where(is_img, positions.to(torch.int32) + 1, -1)
        return span_end, row_request

    same_next = (
        is_img[:-1]
        & is_img[1:]
        & (row_request[:-1] == row_request[1:])
        & (positions[1:] == positions[:-1] + 1)
    )
    tail_true = torch.ones((1,), dtype=torch.bool, device=device)
    breaks_after = torch.cat((~same_next, tail_true), dim=0)
    end_idx = torch.where(breaks_after, rows, tokens)
    r_end = torch.flip(torch.flip(end_idx, dims=(0,)).cummin(dim=0).values, dims=(0,))
    r_end_safe = r_end.clamp_max(tokens - 1).to(torch.int64)
    end_pos = positions[r_end_safe].to(torch.int32) + 1
    image_span_end = torch.where(is_img, end_pos, -1)
    return image_span_end, row_request


@triton.jit
def _image_sliding_attention_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    page_table_ptr,
    positions_ptr,
    image_span_end_ptr,
    row_request_ptr,
    k_descale_ptr,
    v_descale_ptr,
    output_ptr,
    stride_q_t,
    stride_q_h,
    stride_k_page,
    stride_k_tok,
    stride_k_h,
    stride_v_page,
    stride_v_tok,
    stride_v_h,
    stride_pt_b,
    stride_o_t,
    stride_o_h,
    softmax_scale,
    TOKENS: tl.constexpr,
    GQA_GROUP: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    WINDOW: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    q_tile_idx = tl.program_id(0)
    q_head_idx = tl.program_id(1)
    kv_head_idx = q_head_idx // GQA_GROUP

    offs_q = q_tile_idx * BLOCK_Q + tl.arange(0, BLOCK_Q)
    q_in_bounds = offs_q < TOKENS

    span_end = tl.load(image_span_end_ptr + offs_q, mask=q_in_bounds, other=-1)
    # Early-exit immediately on text-only chunks and non-image query tiles.
    if tl.max(span_end, axis=0) < 0:
        return

    is_img = q_in_bounds & (span_end >= 0)
    req_ids = tl.load(row_request_ptr + offs_q, mask=is_img, other=-1)
    q_pos = tl.load(positions_ptr + offs_q, mask=is_img, other=0).to(tl.int32)

    min_req = tl.min(tl.where(is_img, req_ids, 2147483647), axis=0)
    max_req = tl.max(tl.where(is_img, req_ids, -1), axis=0)
    if min_req > max_req:
        return

    offs_d = tl.arange(0, HEAD_DIM)
    q_base = offs_q.to(tl.int64)[:, None] * stride_q_t + q_head_idx.to(tl.int64) * stride_q_h
    q_tile = tl.load(q_ptr + q_base + offs_d[None, :], mask=is_img[:, None], other=0.0)

    k_scale = tl.load(k_descale_ptr).to(tl.float32)
    v_scale = tl.load(v_descale_ptr).to(tl.float32)
    scale_log2 = softmax_scale * k_scale * 1.4426950408889634

    offs_n = tl.arange(0, BLOCK_N)
    k_head_off = kv_head_idx.to(tl.int64) * stride_k_h
    v_head_off = kv_head_idx.to(tl.int64) * stride_v_h
    o_base = offs_q.to(tl.int64)[:, None] * stride_o_t + q_head_idx.to(tl.int64) * stride_o_h

    for req_id in range(min_req, max_req + 1):
        req_mask = is_img & (req_ids == req_id)
        if tl.max(req_mask.to(tl.int32), axis=0) > 0:
            kv_lo = tl.maximum(
                0,
                tl.min(tl.where(req_mask, q_pos - WINDOW + 1, 2147483647), axis=0),
            )
            kv_hi = tl.max(tl.where(req_mask, span_end, 0), axis=0)
            tile_start = kv_lo // BLOCK_N
            tile_end = tl.cdiv(kv_hi, BLOCK_N)

            m_i = tl.full([BLOCK_Q], -float("inf"), dtype=tl.float32)
            l_i = tl.zeros([BLOCK_Q], dtype=tl.float32)
            acc = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

            pt_row_ptr = page_table_ptr + tl.cast(req_id, tl.int64) * stride_pt_b
            for tile_idx in range(tile_start, tile_end):
                kv_start = tile_idx * BLOCK_N
                page_idx = kv_start // PAGE_SIZE
                page_off = kv_start % PAGE_SIZE
                page_id = tl.load(pt_row_ptr + page_idx).to(tl.int64)

                kv_pos = kv_start + offs_n
                kv_in_bounds = (kv_pos >= kv_lo) & (kv_pos < kv_hi)
                tok_offs = (page_off + offs_n).to(tl.int64)

                k_base = page_id * stride_k_page + tok_offs[:, None] * stride_k_tok + k_head_off
                k_tile = tl.load(
                    k_cache_ptr + k_base + offs_d[None, :],
                    mask=kv_in_bounds[:, None],
                    other=0.0,
                ).to(tl.bfloat16)

                qk = tl.dot(q_tile, tl.trans(k_tile)) * scale_log2
                valid_kv = (
                    kv_in_bounds[None, :]
                    & (kv_pos[None, :] > (q_pos[:, None] - WINDOW))
                    & (kv_pos[None, :] < span_end[:, None])
                )
                qk = tl.where(valid_kv, qk, -float("inf"))

                m_ij = tl.max(qk, axis=1)
                m_new = tl.maximum(m_i, m_ij)
                has_new = m_new > -float("inf")
                m_new_safe = tl.where(has_new, m_new, 0.0)
                alpha = tl.where(m_i > -float("inf"), tl.math.exp2(m_i - m_new_safe), 0.0)
                p = tl.where(qk > -float("inf"), tl.math.exp2(qk - m_new_safe[:, None]), 0.0)
                l_i = l_i * alpha + tl.sum(p, axis=1)
                acc = acc * alpha[:, None]

                v_base = page_id * stride_v_page + tok_offs[:, None] * stride_v_tok + v_head_off
                v_tile = tl.load(
                    v_cache_ptr + v_base + offs_d[None, :],
                    mask=kv_in_bounds[:, None],
                    other=0.0,
                ).to(tl.bfloat16)
                acc = tl.dot(p.to(tl.bfloat16), v_tile, acc=acc)
                m_i = m_new

            out_scale = tl.where(l_i > 0.0, v_scale / l_i, 0.0)
            tl.store(
                output_ptr + o_base + offs_d[None, :],
                (acc * out_scale[:, None]).to(tl.bfloat16),
                mask=req_mask[:, None],
            )


@dataclass(frozen=True)
class Query:
    operands: tuple
    window: int
    softmax_scale: float

    @property
    def tokens(self) -> int:
        return self.operands[0][0][0]

    @property
    def num_q_heads(self) -> int:
        return self.operands[0][0][1]

    @property
    def head_dim(self) -> int:
        return self.operands[0][0][2]

    @property
    def page_size(self) -> int:
        return self.operands[1][0][1]

    @property
    def num_kv_heads(self) -> int:
        return self.operands[1][0][2]

    @property
    def gqa_group(self) -> int:
        return self.num_q_heads // self.num_kv_heads


def _tactics(query: Query) -> tuple[tuple[int, int, int, int], ...]:
    """Candidate (block_q, block_n, num_warps, num_stages) tactics within SM120 SMEM."""
    raw = (
        (32, 32, 4, 2),
        (64, 32, 4, 2),
        (32, 32, 8, 2),
        (64, 32, 8, 2),
        (32, 64, 4, 1),
        (16, 64, 4, 1),
    )
    return tuple(t for t in raw if query.page_size % t[1] == 0)


CONTRACT = native_contract(
    component_id="flare.image_sliding_attention",
    query_type=Query,
    tactics=_tactics,
    semantic_version=1,
)


def _kernel_args(tensors, query: Query, tactic: tuple[int, int, int, int]):
    q, k_cache, v_cache, page_table, positions, image_span_end, row_request, k_descale, v_descale, output = tensors
    block_q, block_n, _, _ = tactic
    args = (
        q,
        k_cache,
        v_cache,
        page_table,
        positions,
        image_span_end,
        row_request,
        k_descale,
        v_descale,
        output,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        page_table.stride(0),
        output.stride(0),
        output.stride(1),
        query.softmax_scale,
    )
    constexprs = dict(
        TOKENS=query.tokens,
        GQA_GROUP=query.gqa_group,
        PAGE_SIZE=query.page_size,
        WINDOW=query.window,
        HEAD_DIM=query.head_dim,
        BLOCK_Q=block_q,
        BLOCK_N=block_n,
    )
    grid = (triton.cdiv(query.tokens, block_q), query.num_q_heads, 1)
    return grid, args, constexprs


@program_cache(scope="preparation")
def compile_attention(payload, tactic, ordinal):
    query = Query(**dict(payload))
    tensors = tuple(metadata_tensor(facts) for facts in query.operands)
    _, _, num_warps, num_stages = tactic
    with torch.cuda.device(ordinal), compile_only_launches():
        grid, args, constexprs = _kernel_args(tensors, query, tactic)
        prog = launch_triton(
            _image_sliding_attention_kernel,
            grid[:2],
            *args,
            **constexprs,
            num_warps=num_warps,
            num_stages=num_stages,
        )
        return {"attend": prog}


@dataclass(frozen=True)
class State:
    query: Query
    tactic: tuple[int, int, int, int]
    programs: dict

    def run(self, tensors):
        grid, args, _ = _kernel_args(tensors, self.query, self.tactic)
        self.programs["attend"][grid](*args)


@torch.library.custom_op("flare::image_sliding_attention", mutates_args=("output",))
def _execute(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    positions: torch.Tensor,
    image_span_end: torch.Tensor,
    row_request: torch.Tensor,
    k_descale: torch.Tensor,
    v_descale: torch.Tensor,
    output: torch.Tensor,
    handle: int,
) -> None:
    plan = plan_from_handle(handle)
    state = require_prepared(plan, plan.component_id, q.device)
    state.run((q, k_cache, v_cache, page_table, positions, image_span_end, row_request, k_descale, v_descale, output))


@_execute.register_fake
def _fake(
    q, k_cache, v_cache, page_table, positions, image_span_end, row_request,
    k_descale, v_descale, output, handle,
):
    pass


class PreparedImageSlidingAttention:
    """Prepared sparse paged-cache sliding attention overwriting image rows in place."""

    def __init__(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        positions: torch.Tensor,
        image_span_end: torch.Tensor,
        row_request: torch.Tensor,
        output: torch.Tensor,
        *,
        window: int,
        k_descale: torch.Tensor | None = None,
        v_descale: torch.Tensor | None = None,
        softmax_scale: float | None = None,
        session,
        workspace,
        name: str,
        tuning_page_count: int | None = None,
    ):
        head_dim = q.shape[-1]
        if (
            q.ndim != 3
            or output.shape != q.shape
            or q.dtype != torch.bfloat16
            or output.dtype != torch.bfloat16
            or head_dim <= 0
            or (head_dim & (head_dim - 1)) != 0
            or not q.is_contiguous()
            or not output.is_contiguous()
        ):
            raise ValueError("image sliding attention requires contiguous BF16 query and output with power-of-two head_dim")
        if (
            k_cache.ndim != 4
            or v_cache.ndim != 4
            or k_cache.shape != v_cache.shape
            or k_cache.shape[3] != head_dim
            or k_cache.dtype not in (torch.float8_e4m3fn, torch.bfloat16)
            or v_cache.dtype != k_cache.dtype
            or k_cache.stride(3) != 1
            or v_cache.stride(3) != 1
        ):
            raise ValueError("image sliding attention requires matching FP8/BF16 paged K/V caches")
        if q.shape[1] % k_cache.shape[2] != 0:
            raise ValueError("query heads must be divisible by KV heads")
        if page_table.ndim != 2 or positions.shape != (q.shape[0],) or image_span_end.shape != (q.shape[0],) or row_request.shape != (q.shape[0],):
            raise ValueError("invalid metadata shapes for image sliding attention")
        if type(window) is not int or window < 1:
            raise ValueError("window must be a positive integer")
        if tuning_page_count is None:
            tuning_page_count = k_cache.shape[0]
        if type(tuning_page_count) is not int or not 0 < tuning_page_count <= k_cache.shape[0]:
            raise ValueError("tuning_page_count must be positive and within cache capacity")
        self.tuning_page_count = tuning_page_count

        if k_descale is None:
            k_descale = persistent_empty((1,), torch.float32, q.device).fill_(1.0)
        if v_descale is None:
            v_descale = persistent_empty((1,), torch.float32, q.device).fill_(1.0)
        if softmax_scale is None:
            softmax_scale = head_dim ** -0.5

        self.arguments = (
            q,
            k_cache,
            v_cache,
            page_table,
            positions,
            image_span_end,
            row_request,
            k_descale,
            v_descale,
            output,
        )
        query = Query(
            operands=tuple(tensor_facts(t) for t in self.arguments),
            window=window,
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
            scratch_bytes=lambda _t: 0,
            materialize=materialize,
            compile_jobs=lambda t, d: (
                CompileJob.create(
                    "hyperflare_recipes.components.image_sliding_attention:compile_attention",
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
                    dq, kc, vc, pt, pos, span_end, req, kd, vd, out = declaration
                    tokens = dq.shape[0]
                    bq = torch.empty_like(dq).normal_(std=0.2)
                    bpt = torch.empty_like(pt)
                    bpos = torch.arange(tokens, device=pos.device, dtype=pos.dtype)
                    bspan = torch.full_like(span_end, -1)
                    breq = torch.zeros_like(req)
                    bout = torch.zeros_like(out)
                    img_len = min(280, tokens)
                    img_start = max(0, min(tokens - img_len, 128))
                    bspan[img_start : img_start + img_len] = img_start + img_len
                    extent = min(max(tokens, window + img_len), pt.shape[1] * kc.shape[1])
                    pages_per_row = triton.cdiv(extent, kc.shape[1])
                    page_ids = (
                        torch.arange(pt.shape[0], device=pt.device, dtype=torch.int64)[:, None] * pages_per_row
                        + torch.arange(pt.shape[1], device=pt.device, dtype=torch.int64)
                    )
                    bpt.copy_((page_ids % self.tuning_page_count).to(pt.dtype))
                    return (bq, kc, vc, bpt, bpos, bspan, breq, kd, vd, bout), bq.clone()

                args, seed = benchmark_resource(session, ("image_sliding_attention", name), inputs)
                produce = lambda: args[0].copy_(seed)

            return PreparedCall(
                run=lambda: state.run(args),
                output=args[-1],
                produce=produce,
                owners=prime_owners(workspace, args),
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

    def functional(self, *, session, k_cache="BOUND", v_cache="BOUND"):
        _, kc, vc, _, _, _, _, kd, vd, _ = self.arguments
        self.caches = (
            kc if k_cache == "BOUND" else k_cache,
            vc if v_cache == "BOUND" else v_cache,
        )
        self.descales = (kd, vd)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.arguments = None

    def __call__(
        self,
        q=None,
        output=None,
        *,
        k_cache=None,
        v_cache=None,
        page_table=None,
        positions=None,
        image_span_end=None,
        row_request=None,
    ):
        if self.arguments is not None:
            _execute(*self.arguments, self.plan.handle)
            return self.arguments[-1]
        kc = self.caches[0] if k_cache is None else k_cache
        vc = self.caches[1] if v_cache is None else v_cache
        if kc is None or vc is None:
            raise ValueError("functional image sliding attention requires k_cache and v_cache")
        _execute(
            q,
            kc,
            vc,
            page_table,
            positions,
            image_span_end,
            row_request,
            *self.descales,
            output,
            self.plan.handle,
        )
        return output
