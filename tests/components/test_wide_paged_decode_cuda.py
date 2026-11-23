# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Numerical parity, CUDA graph replay, and NaN-safety tests for PreparedWidePagedDecode."""

from __future__ import annotations

import math
import os

import pytest
import torch

from b12x.preparation import PreparationSession
from hyperflare_recipes.components.wide_paged_decode import PreparedWidePagedDecode
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires explicit GPU assignment via FLARE_RUN_CUDA_TESTS=1",
)


def _reference_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_lo_cache: torch.Tensor,
    v_hi_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    softmax_scale: float,
    k_descale: float = 1.0,
    v_descale: float = 1.0,
) -> torch.Tensor:
    tokens, num_q_heads, dim = q.shape
    page_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    gqa = num_q_heads // num_kv_heads
    batch = cache_seqlens.numel()
    out = torch.zeros_like(q, dtype=torch.float32)
    scale = softmax_scale * k_descale
    for b in range(batch):
        q_start = int(cu_seqlens_q[b].item())
        q_end = int(cu_seqlens_q[b + 1].item())
        q_len = q_end - q_start
        seq_len = int(cache_seqlens[b].item())
        if q_len <= 0 or seq_len <= 0:
            continue
        num_pages = (seq_len + page_size - 1) // page_size
        pages = page_table[b, :num_pages].long()
        k_seq = k_cache[pages].reshape(num_pages * page_size, num_kv_heads, dim)[:seq_len].float()
        v_lo_seq = v_lo_cache[pages].reshape(num_pages * page_size, num_kv_heads, dim // 2)[:seq_len].float()
        v_hi_seq = v_hi_cache[pages].reshape(num_pages * page_size, num_kv_heads, dim // 2)[:seq_len].float()
        v_seq = torch.cat((v_lo_seq, v_hi_seq), dim=-1)
        k_rep = k_seq.repeat_interleave(gqa, dim=1)  # [S, Hq, D]
        v_rep = v_seq.repeat_interleave(gqa, dim=1)  # [S, Hq, D]
        q_req = q[q_start:q_end].float()  # [Q, Hq, D]
        scores = torch.einsum("qhd,khd->hqk", q_req, k_rep) * scale
        ctx = seq_len - q_len
        q_pos = torch.arange(q_len, device=q.device)[:, None] + ctx
        k_pos = torch.arange(seq_len, device=q.device)[None, :]
        scores = scores.masked_fill(k_pos > q_pos, float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        req_out = torch.einsum("hqk,khd->qhd", probs, v_rep) * v_descale
        out[q_start:q_end] = req_out
    return out.to(q.dtype)


def _make_caches(num_pages, page_size, num_kv_heads, device, *, poison_nans=False):
    gen = torch.Generator(device=device).manual_seed(12345)
    k = (torch.randn((num_pages, page_size, num_kv_heads, 512), device=device, generator=gen) * 0.25).to(
        torch.float8_e4m3fn
    )
    v_lo = (torch.randn((num_pages, page_size, num_kv_heads, 256), device=device, generator=gen) * 0.25).to(
        torch.float8_e4m3fn
    )
    v_hi = (torch.randn((num_pages, page_size, num_kv_heads, 256), device=device, generator=gen) * 0.25).to(
        torch.float8_e4m3fn
    )
    if poison_nans:
        # Fill the upper half of every page with NaN bit patterns (0x7f in FP8 E4M3).
        k.view(torch.uint8)[:, page_size // 2 :, :, :] = 0x7F
        v_lo.view(torch.uint8)[:, page_size // 2 :, :, :] = 0x7F
        v_hi.view(torch.uint8)[:, page_size // 2 :, :, :] = 0x7F
    return k, v_lo, v_hi


@pytest.mark.parametrize(
    "num_q_heads,num_kv_heads,batch,width,seq_lens,q_lens",
    [
        (16, 2, 1, 1, [1], [1]),
        (16, 2, 1, 1, [129], [1]),
        (16, 2, 1, 1, [4097], [1]),
        (32, 4, 1, 6, [2050], [6]),
        (16, 2, 4, 1, [513, 0, 2048, 17], [1, 0, 1, 1]),
        (16, 2, 4, 6, [1030, 0, 256, 3000], [6, 0, 3, 2]),
        (16, 1, 2, 1, [300, 1500], [1, 1]),
        (8, 2, 2, 4, [200, 500], [4, 2]),
        (2, 1, 2, 1, [100, 300], [1, 1]),
        (16, 2, 2, 1, [0, 0], [0, 0]),
    ],
)
@torch.no_grad()
def test_wide_paged_decode_matches_reference(num_q_heads, num_kv_heads, batch, width, seq_lens, q_lens):
    device = torch.device("cuda", 0)
    page_size = 128
    max_pages = 64
    num_pages = 96
    tokens = batch * width
    k_cache, v_lo, v_hi = _make_caches(num_pages, page_size, num_kv_heads, device)

    q = torch.randn((tokens, num_q_heads, 512), dtype=torch.bfloat16, device=device) * 0.2
    output = torch.empty_like(q)
    # Shuffled non-contiguous page assignments.
    perm = torch.randperm(num_pages, dtype=torch.int32, device=device)
    page_table = torch.full((batch, max_pages), -1, dtype=torch.int32, device=device)
    cursor = 0
    for b, sl in enumerate(seq_lens):
        np = (sl + page_size - 1) // page_size
        if np > 0:
            page_table[b, :np] = perm[cursor : cursor + np]
            cursor += np
    cache_seqlens = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    cu = [0]
    for ql in q_lens:
        cu.append(cu[-1] + ql)
    cu_seqlens_q = torch.tensor(cu, dtype=torch.int32, device=device)
    k_descale = torch.tensor([math.sqrt(512.0)], dtype=torch.float32, device=device)
    v_descale = torch.tensor([1.0], dtype=torch.float32, device=device)

    arena = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint="test-wide-decode", workload_schema="test", numerical_policy="default",
        compile_workers=1, autotune_effort="off",
    ) as session:
        session.register_scratch(arena)
        op = PreparedWidePagedDecode(
            q,
            k_cache,
            v_lo,
            v_hi,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            output,
            width=width,
            k_descale=k_descale,
            v_descale=v_descale,
            session=session,
            workspace=arena,
            name="wide_decode_test",
        )
        fn = op.functional(session=session)
        session.tune(show_progress=False)
        session.freeze()

    actual = fn(q, page_table=page_table, cache_seqlens=cache_seqlens, cu_seqlens_q=cu_seqlens_q)
    expected = _reference_attention(
        q,
        k_cache,
        v_lo,
        v_hi,
        page_table,
        cache_seqlens,
        cu_seqlens_q,
        softmax_scale=512 ** -0.5,
        k_descale=math.sqrt(512.0),
        v_descale=1.0,
    )
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)


@torch.no_grad()
def test_wide_paged_decode_ignores_nan_tail_and_replays_in_cuda_graph():
    device = torch.device("cuda", 0)
    page_size = 128
    num_kv_heads = 2
    num_q_heads = 16
    batch = 1
    width = 1
    max_pages = 160
    num_pages = 160
    # Poison upper half of page 0 with NaNs, then test seq_len=64 (strictly lower half).
    k_cache, v_lo, v_hi = _make_caches(num_pages, page_size, num_kv_heads, device, poison_nans=False)
    k_cache.view(torch.uint8)[0, 64:, :, :] = 0x7F
    v_lo.view(torch.uint8)[0, 64:, :, :] = 0x7F
    v_hi.view(torch.uint8)[0, 64:, :, :] = 0x7F

    q = torch.randn((1, num_q_heads, 512), dtype=torch.bfloat16, device=device) * 0.2
    output = torch.zeros_like(q)
    page_table = torch.arange(max_pages, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([64], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, 1], dtype=torch.int32, device=device)
    k_descale = torch.tensor([math.sqrt(512.0)], dtype=torch.float32, device=device)
    v_descale = torch.tensor([1.0], dtype=torch.float32, device=device)

    arena = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint="test-wide-graph", workload_schema="test", numerical_policy="default",
        compile_workers=1, autotune_effort="off",
    ) as session:
        session.register_scratch(arena)
        op = PreparedWidePagedDecode(
            q,
            k_cache,
            v_lo,
            v_hi,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            output,
            width=width,
            k_descale=k_descale,
            v_descale=v_descale,
            session=session,
            workspace=arena,
            name="wide_decode_graph",
        )
        session.tune(show_progress=False)
        session.freeze()

    # Warmup and capture CUDA graph at seq_len=64 (with poisoned NaN tail in [64..127]).
    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        op()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op()

    # 1. Replay at seq_len=64 (NaN tail check).
    graph.replay()
    torch.cuda.synchronize()
    expected_64 = _reference_attention(
        q, k_cache, v_lo, v_hi, page_table, cache_seqlens, cu_seqlens_q,
        softmax_scale=512 ** -0.5, k_descale=math.sqrt(512.0),
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected_64.float(), rtol=2e-2, atol=2e-2)

    # 2. Replay the SAME captured graph at seq_len=16384 (using clean pages 1..128).
    page_table.copy_(torch.arange(1, max_pages + 1, dtype=torch.int32, device=device).clamp_max(num_pages - 1).unsqueeze(0))
    cache_seqlens.fill_(16384)
    graph.replay()
    torch.cuda.synchronize()
    expected_16k = _reference_attention(
        q, k_cache, v_lo, v_hi, page_table, cache_seqlens, cu_seqlens_q,
        softmax_scale=512 ** -0.5, k_descale=math.sqrt(512.0),
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected_16k.float(), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("width", [1, 6])
def test_bf16_kv_parity_and_cuda_graph_replay(width):
    """Verify BF16 K(512) + split V(256+256) parity and CUDA graph replay with unit scale."""
    device = torch.device("cuda")
    page_size = 128
    num_q_heads, num_kv_heads = 16, 4
    batch = 2
    num_pages = 32
    max_pages = 16

    gen = torch.Generator(device=device).manual_seed(777 + width)
    q = (torch.randn((batch * width, num_q_heads, 512), device=device, generator=gen) * 0.25).to(torch.bfloat16)
    k_cache = (torch.randn((num_pages, page_size, num_kv_heads, 512), device=device, generator=gen) * 0.25).to(torch.bfloat16)
    v_lo = (torch.randn((num_pages, page_size, num_kv_heads, 256), device=device, generator=gen) * 0.25).to(torch.bfloat16)
    v_hi = (torch.randn((num_pages, page_size, num_kv_heads, 256), device=device, generator=gen) * 0.25).to(torch.bfloat16)
    # Poison upper half of page 0 with NaN to ensure out-of-bounds masking on BF16 caches too.
    k_cache[0, page_size // 2 :] = float("nan")
    v_lo[0, page_size // 2 :] = float("nan")
    v_hi[0, page_size // 2 :] = float("nan")

    page_table = torch.zeros((batch, max_pages), dtype=torch.int32, device=device)
    page_table[0, 0] = 0
    page_table[1, :10] = torch.arange(1, 11, dtype=torch.int32, device=device)
    cache_seqlens = torch.tensor([64, 1200], dtype=torch.int32, device=device)
    cu_seqlens_q = torch.tensor([0, width, 2 * width], dtype=torch.int32, device=device)
    output = torch.empty_like(q)

    arena = ScratchArena(device)
    with RecordedPreparationSession(
        device=device, checkpoint=f"test-wide-bf16-{width}", workload_schema="test", numerical_policy="default",
        compile_workers=1, autotune_effort="off",
    ) as session:
        session.register_scratch(arena)
        op = PreparedWidePagedDecode(
            q,
            k_cache,
            v_lo,
            v_hi,
            page_table,
            cache_seqlens,
            cu_seqlens_q,
            output,
            width=width,
            softmax_scale=1.0,
            session=session,
            workspace=arena,
            name=f"wide_decode_bf16_{width}",
        )
        session.tune(show_progress=False)
        session.freeze()

    stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(stream):
        op()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        op()
    graph.replay()
    torch.cuda.synchronize()

    expected = _reference_attention(
        q, k_cache, v_lo, v_hi, page_table, cache_seqlens, cu_seqlens_q,
        softmax_scale=1.0,
    )
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output.float(), expected.float(), rtol=1e-2, atol=1e-2)
