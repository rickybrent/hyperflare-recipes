# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity, prefix-cache, and zero-overhead tests for PreparedImageSlidingAttention."""

from __future__ import annotations

import math
import os

import pytest
import torch

from hyperflare_recipes.components.image_sliding_attention import (
    PreparedImageSlidingAttention,
    compute_image_sliding_spans,
)
from ignition.preparation.scratch import ScratchArena
from b12x.preparation import PreparationSession

CUDA = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires explicit GPU assignment and FLARE_RUN_CUDA_TESTS=1",
)

IMAGE_TOKEN = 258880
PAGE_SIZE = 128
WINDOW = 1024


def _reference_image_sliding(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    page_table: torch.Tensor,
    positions: torch.Tensor,
    image_span_end: torch.Tensor,
    row_request: torch.Tensor,
    k_descale: float,
    v_descale: float,
    window: int,
    softmax_scale: float,
    base_output: torch.Tensor,
) -> torch.Tensor:
    tokens, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[2]
    gqa = num_q_heads // num_kv_heads
    out = base_output.clone().float()

    for r in range(tokens):
        end_pos = int(image_span_end[r].item())
        if end_pos < 0:
            continue
        req = int(row_request[r].item())
        pos = int(positions[r].item())
        lo = max(0, pos - window + 1)
        hi = end_pos
        kv_positions = list(range(lo, hi))
        ks = []
        vs = []
        for p in kv_positions:
            page_id = int(page_table[req, p // PAGE_SIZE].item())
            off = p % PAGE_SIZE
            ks.append(k_cache[page_id, off].float())
            vs.append(v_cache[page_id, off].float())
        k_seq = torch.stack(ks, dim=0)  # [L, Hkv, D]
        v_seq = torch.stack(vs, dim=0)  # [L, Hkv, D]
        k_rep = k_seq.repeat_interleave(gqa, dim=1) * k_descale
        v_rep = v_seq.repeat_interleave(gqa, dim=1) * v_descale
        qr = q[r].float()  # [Hq, D]
        scores = torch.einsum("hd,lhd->hl", qr, k_rep) * softmax_scale
        probs = torch.softmax(scores, dim=-1)
        out[r] = torch.einsum("hl,lhd->hd", probs, v_rep)
    return out


def _fill_fp8_cache(
    device: torch.device,
    seq_lens: list[int],
    num_kv_heads: int,
    head_dim: int,
    *,
    poison_unused: bool = False,
):
    total_pages = sum((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens) + 4
    k_cache = torch.zeros((total_pages, PAGE_SIZE, num_kv_heads, head_dim), dtype=torch.float8_e4m3fn, device=device)
    v_cache = torch.zeros((total_pages, PAGE_SIZE, num_kv_heads, head_dim), dtype=torch.float8_e4m3fn, device=device)
    if poison_unused:
        k_cache.view(torch.uint8).fill_(0x7F)
        v_cache.view(torch.uint8).fill_(0x7F)

    perm = torch.randperm(total_pages, device=device, dtype=torch.int32)
    max_pages = max((s + PAGE_SIZE - 1) // PAGE_SIZE for s in seq_lens) + 1
    page_table = torch.zeros((len(seq_lens), max_pages), dtype=torch.int32, device=device)
    cursor = 0
    for b, s in enumerate(seq_lens):
        n_pages = (s + PAGE_SIZE - 1) // PAGE_SIZE
        for p in range(n_pages):
            pid = int(perm[cursor].item())
            cursor += 1
            page_table[b, p] = pid
            valid = min(PAGE_SIZE, s - p * PAGE_SIZE)
            k_cache[pid, :valid].copy_(torch.randn(valid, num_kv_heads, head_dim, device=device).clamp(-2, 2).to(torch.float8_e4m3fn))
            v_cache[pid, :valid].copy_(torch.randn(valid, num_kv_heads, head_dim, device=device).clamp(-2, 2).to(torch.float8_e4m3fn))
    return k_cache, v_cache, page_table


@CUDA
@torch.no_grad()
def test_single_image_and_prefix_cache_parity():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(11)

    # Simulate Turn 1 cached prefix of 1500 tokens + Turn 2 chunk of 512 tokens:
    # [30 text, 280 image, 50 text, 152 padded]
    prefix_len = 1500
    chunk_tokens = 512
    live_tokens = 360
    total_seq = prefix_len + live_tokens
    num_q_heads, num_kv_heads, head_dim = 16, 8, 256

    k_cache, v_cache, page_table = _fill_fp8_cache(
        device, [total_seq], num_kv_heads, head_dim, poison_unused=True
    )
    q = torch.randn((chunk_tokens, num_q_heads, head_dim), dtype=torch.bfloat16, device=device) * 0.25
    token_ids = torch.ones((chunk_tokens,), dtype=torch.int32, device=device)
    token_ids[30 : 30 + 280] = IMAGE_TOKEN
    positions = torch.full((chunk_tokens,), -1, dtype=torch.int64, device=device)
    positions[:live_tokens] = torch.arange(prefix_len, total_seq, dtype=torch.int64, device=device)
    cu_seqlens_q = torch.tensor([0, live_tokens], dtype=torch.int32, device=device)
    k_descale = torch.tensor([math.sqrt(head_dim)], dtype=torch.float32, device=device)
    v_descale = torch.tensor([1.0], dtype=torch.float32, device=device)

    sentinel = torch.randn((chunk_tokens, num_q_heads, head_dim), dtype=torch.bfloat16, device=device)
    output = sentinel.clone()
    image_span_end, row_request = compute_image_sliding_spans(token_ids, positions, cu_seqlens_q, IMAGE_TOKEN)

    ws = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        op = PreparedImageSlidingAttention(
            q, k_cache, v_cache, page_table, positions, image_span_end, row_request, output,
            window=WINDOW, k_descale=k_descale, v_descale=v_descale,
            session=session, workspace=ws, name="img_slide_prefix",
        )
        session.freeze()
        op()

    expected = _reference_image_sliding(
        q, k_cache, v_cache, page_table, positions, image_span_end, row_request,
        float(k_descale.item()), float(v_descale.item()), WINDOW, head_dim ** -0.5, sentinel,
    )
    # Non-image and padded rows must remain bit-for-bit equal to sentinel.
    non_img = image_span_end < 0
    assert torch.equal(output[non_img], sentinel[non_img])
    # Image rows must match FP32 reference closely.
    img = image_span_end >= 0
    assert int(img.sum().item()) == 280
    torch.testing.assert_close(output[img].float(), expected[img], rtol=0.02, atol=0.02)


@CUDA
@torch.no_grad()
def test_1120_token_window_clipped_image_matches_hf_mask():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(22)

    chunk_tokens = 1280
    num_q_heads, num_kv_heads, head_dim = 8, 4, 256
    k_cache, v_cache, page_table = _fill_fp8_cache(device, [chunk_tokens], num_kv_heads, head_dim)
    q = torch.randn((chunk_tokens, num_q_heads, head_dim), dtype=torch.bfloat16, device=device) * 0.25
    token_ids = torch.ones((chunk_tokens,), dtype=torch.int32, device=device)
    token_ids[50 : 50 + 1120] = IMAGE_TOKEN
    positions = torch.arange(chunk_tokens, dtype=torch.int64, device=device)
    cu_seqlens_q = torch.tensor([0, chunk_tokens], dtype=torch.int32, device=device)
    k_descale = torch.tensor([math.sqrt(head_dim)], dtype=torch.float32, device=device)
    v_descale = torch.tensor([1.0], dtype=torch.float32, device=device)
    sentinel = torch.full((chunk_tokens, num_q_heads, head_dim), 3.25, dtype=torch.bfloat16, device=device)
    output = sentinel.clone()
    image_span_end, row_request = compute_image_sliding_spans(token_ids, positions, cu_seqlens_q, IMAGE_TOKEN)

    assert int(image_span_end[50].item()) == 50 + 1120
    assert int(image_span_end[50 + 1119].item()) == 50 + 1120
    assert int(image_span_end[49].item()) == -1
    assert int(image_span_end[50 + 1120].item()) == -1

    ws = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        op = PreparedImageSlidingAttention(
            q, k_cache, v_cache, page_table, positions, image_span_end, row_request, output,
            window=WINDOW, k_descale=k_descale, v_descale=v_descale,
            session=session, workspace=ws, name="img_slide_1120",
        )
        session.freeze()
        op()

    expected = _reference_image_sliding(
        q, k_cache, v_cache, page_table, positions, image_span_end, row_request,
        float(k_descale.item()), float(v_descale.item()), WINDOW, head_dim ** -0.5, sentinel,
    )
    assert torch.equal(output[image_span_end < 0], sentinel[image_span_end < 0])
    torch.testing.assert_close(output[image_span_end >= 0].float(), expected[image_span_end >= 0], rtol=0.02, atol=0.02)


@CUDA
@torch.no_grad()
def test_multi_image_multi_request_and_cuda_graph_text_early_exit():
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.manual_seed(33)

    # Batch of 2 requests sharing one 512-token chunk, where request 0 ends with an image
    # right at row 128 and request 1 starts immediately with an image at row 128 (crossing a tile).
    chunk_tokens = 512
    num_q_heads, num_kv_heads, head_dim = 16, 8, 256
    k_cache, v_cache, page_table = _fill_fp8_cache(
        device, [200 + 128, 100 + 256], num_kv_heads, head_dim, poison_unused=True
    )
    q = torch.randn((chunk_tokens, num_q_heads, head_dim), dtype=torch.bfloat16, device=device) * 0.25
    k_descale = torch.tensor([math.sqrt(head_dim)], dtype=torch.float32, device=device)
    v_descale = torch.tensor([1.0], dtype=torch.float32, device=device)

    positions = torch.full((chunk_tokens,), -1, dtype=torch.int64, device=device)
    image_span_end = torch.full((chunk_tokens,), -1, dtype=torch.int32, device=device)
    row_request = torch.full((chunk_tokens,), -1, dtype=torch.int32, device=device)
    sentinel = torch.randn((chunk_tokens, num_q_heads, head_dim), dtype=torch.bfloat16, device=device)
    output = sentinel.clone()

    ws = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        op = PreparedImageSlidingAttention(
            q, k_cache, v_cache, page_table, positions, image_span_end, row_request, output,
            window=WINDOW, k_descale=k_descale, v_descale=v_descale,
            session=session, workspace=ws, name="img_slide_graph",
        )
        session.freeze()

        # Capture CUDA graph on a text-only chunk (all -1 spans).
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            op()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            op()
        torch.cuda.synchronize(device)

        # Replay 1: text-only chunk must leave sentinel untouched.
        output.copy_(sentinel)
        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.synchronize(device)
        assert torch.equal(output, sentinel)

        # Replay 2: multi-request, multi-image with adjacent request-boundary images.
        token_ids = torch.ones((chunk_tokens,), dtype=torch.int32, device=device)
        token_ids[10:50] = IMAGE_TOKEN     # req 0 image 1
        token_ids[90:128] = IMAGE_TOKEN    # req 0 image 2 (ends right at req boundary 128)
        token_ids[128:180] = IMAGE_TOKEN   # req 1 image 1 (starts right at req boundary 128)
        positions[:128] = torch.arange(200, 328, dtype=torch.int64, device=device)
        positions[128:384] = torch.arange(100, 356, dtype=torch.int64, device=device)
        positions[384:] = -1
        cu_seqlens_q = torch.tensor([0, 128, 384], dtype=torch.int32, device=device)
        se, rr = compute_image_sliding_spans(token_ids, positions, cu_seqlens_q, IMAGE_TOKEN)
        image_span_end.copy_(se)
        row_request.copy_(rr)
        output.copy_(sentinel)

        with torch.cuda.stream(stream):
            graph.replay()
        torch.cuda.synchronize(device)

    expected = _reference_image_sliding(
        q, k_cache, v_cache, page_table, positions, image_span_end, row_request,
        float(k_descale.item()), float(v_descale.item()), WINDOW, head_dim ** -0.5, sentinel,
    )
    assert torch.equal(output[image_span_end < 0], sentinel[image_span_end < 0])
    torch.testing.assert_close(output[image_span_end >= 0].float(), expected[image_span_end >= 0], rtol=0.02, atol=0.02)


@CUDA
@pytest.mark.parametrize("chunk_start,img_len", [(0, 256), (500, 256), (1500, 1120)])
@torch.no_grad()
def test_staged_circular_window_image_attention_parity(chunk_start, img_len):
    """Verify functional unbound staged circular-window image attention across early, short-prefix,
    poisoned stale ring slots, and >window wraparound (including 1120-token window-clipped image)."""
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.components.window_staging import PreparedWindowStaging

    torch.manual_seed(700 + chunk_start)
    device = torch.device("cuda:0")
    num_heads, num_kv_heads, head_dim = 32, 16, 256
    chunk_tokens = img_len + 128

    # Poison full circular ring with 0x7F (FP8 NaN) before populating only valid prefix slots [first..chunk_start).
    hk = torch.empty((3, WINDOW, num_kv_heads, head_dim), dtype=torch.float8_e4m3fn, device=device)
    hv = torch.empty_like(hk)
    hk.view(torch.uint8).fill_(0x7F)
    hv.view(torch.uint8).fill_(0x7F)

    first = max(0, chunk_start - WINDOW)
    hist_len = chunk_start - first
    hist_k = (torch.randn(hist_len, num_kv_heads, head_dim, device=device) * 0.1).bfloat16().to(torch.float8_e4m3fn)
    hist_v = (torch.randn(hist_len, num_kv_heads, head_dim, device=device) * 0.2).bfloat16().to(torch.float8_e4m3fn)
    for i in range(hist_len):
        pos = first + i
        hk[1, pos % WINDOW].copy_(hist_k[i])
        hv[1, pos % WINDOW].copy_(hist_v[i])

    q = (torch.randn(chunk_tokens, num_heads, head_dim, device=device) * 0.1).bfloat16()
    k_chunk = (torch.randn(chunk_tokens, num_kv_heads, head_dim, device=device) * 0.1).bfloat16()
    v_chunk = (torch.randn(chunk_tokens, num_kv_heads, head_dim, device=device) * 0.2).bfloat16()

    m = Metadata.allocate(chunk_tokens, 1, 16, device, samples=chunk_tokens)
    m.set(
        torch.tensor([chunk_start], dtype=torch.int64, device=device),
        torch.tensor([chunk_tokens], dtype=torch.int32, device=device),
        torch.tensor([1], dtype=torch.int64, device=device),
        m.page_table,
        torch.zeros(1, chunk_tokens, dtype=torch.int64, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
    )

    pages = (chunk_tokens + WINDOW + PAGE_SIZE - 1) // PAGE_SIZE
    staged_k_decl = torch.empty((pages, PAGE_SIZE, num_kv_heads, head_dim), dtype=torch.float8_e4m3fn, device=device)
    staged_v_decl = torch.empty_like(staged_k_decl)
    staged_pages = torch.arange(pages, dtype=torch.int32, device=device).view(1, -1)

    token_ids = torch.ones((chunk_tokens,), dtype=torch.int32, device=device)
    token_ids[32 : 32 + img_len] = IMAGE_TOKEN
    image_span_end, row_request = compute_image_sliding_spans(token_ids, m.positions, m.query_start_loc, IMAGE_TOKEN)
    first_t = (m.starts - WINDOW).clamp_min(0)
    row_first = first_t[row_request.clamp_min(0).long()]
    staged_positions = torch.where(row_request >= 0, m.positions - row_first, m.positions)
    staged_span_end = torch.where(image_span_end >= 0, (image_span_end.long() - row_first).to(torch.int32), -1)

    k_descale = torch.tensor([math.sqrt(head_dim)], dtype=torch.float32, device=device)
    v_descale = torch.ones(1, dtype=torch.float32, device=device)
    sentinel = torch.randn((chunk_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    output = sentinel.clone()

    ws = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        st_op = PreparedWindowStaging(
            hk, hv, k_chunk, v_chunk, staged_k_decl, staged_v_decl, m, session=session, name="stage_img"
        ).functional(session=session)
        img_op = PreparedImageSlidingAttention(
            q, staged_k_decl, staged_v_decl, staged_pages, staged_positions, staged_span_end, row_request, output,
            window=WINDOW, k_descale=k_descale, v_descale=v_descale,
            session=session, workspace=ws, name="img_slide_staged", tuning_page_count=pages,
        ).functional(session=session, k_cache=None, v_cache=None)
        session.freeze()
        sk, sv = st_op(k_chunk, v_chunk, metadata=m)
        actual = img_op(
            q, sentinel.clone(),
            k_cache=sk, v_cache=sv, page_table=staged_pages,
            positions=staged_positions, image_span_end=staged_span_end, row_request=row_request,
        )
        torch.cuda.synchronize()

    assert not torch.isnan(actual.float()).any()
    assert torch.equal(actual[staged_span_end < 0], sentinel[staged_span_end < 0])
    expected = _reference_image_sliding(
        q, sk, sv, staged_pages, staged_positions, staged_span_end, row_request,
        float(k_descale.item()), float(v_descale.item()), WINDOW, head_dim ** -0.5, sentinel,
    )
    torch.testing.assert_close(actual[staged_span_end >= 0].float(), expected[staged_span_end >= 0], rtol=0.02, atol=0.02)


@CUDA
@torch.no_grad()
def test_staged_circular_window_image_attention_multi_request_parity():
    """Verify multi-request staged circular-window image attention with heterogeneous starts (pre-wrap & post-wrap)."""
    from hyperflare_recipes.components.paged_batch import Metadata
    from hyperflare_recipes.components.window_staging import PreparedWindowStaging

    torch.manual_seed(911)
    device = torch.device("cuda:0")
    num_heads, num_kv_heads, head_dim = 16, 8, 256
    starts = [500, 1500]
    lens = [192, 320]
    chunk_tokens = sum(lens)

    hk = torch.empty((4, WINDOW, num_kv_heads, head_dim), dtype=torch.float8_e4m3fn, device=device)
    hv = torch.empty_like(hk)
    hk.view(torch.uint8).fill_(0x7F)
    hv.view(torch.uint8).fill_(0x7F)
    state_rows = [1, 3]
    for row, start in zip(state_rows, starts):
        first = max(0, start - WINDOW)
        hist_len = start - first
        hist_k = (torch.randn(hist_len, num_kv_heads, head_dim, device=device) * 0.1).bfloat16().to(torch.float8_e4m3fn)
        hist_v = (torch.randn(hist_len, num_kv_heads, head_dim, device=device) * 0.2).bfloat16().to(torch.float8_e4m3fn)
        for i in range(hist_len):
            pos = first + i
            hk[row, pos % WINDOW].copy_(hist_k[i])
            hv[row, pos % WINDOW].copy_(hist_v[i])

    q = (torch.randn(chunk_tokens, num_heads, head_dim, device=device) * 0.1).bfloat16()
    k_chunk = (torch.randn(chunk_tokens, num_kv_heads, head_dim, device=device) * 0.1).bfloat16()
    v_chunk = (torch.randn(chunk_tokens, num_kv_heads, head_dim, device=device) * 0.2).bfloat16()

    m = Metadata.allocate(chunk_tokens, 2, 16, device, samples=chunk_tokens)
    m.set(
        torch.tensor(starts, dtype=torch.int64, device=device),
        torch.tensor(lens, dtype=torch.int32, device=device),
        torch.tensor(state_rows, dtype=torch.int64, device=device),
        m.page_table,
        torch.zeros(2, chunk_tokens, dtype=torch.int64, device=device),
        torch.tensor([2], dtype=torch.int32, device=device),
    )

    pages_per_row = (max(lens) + WINDOW + PAGE_SIZE - 1) // PAGE_SIZE
    total_pages = 2 * pages_per_row
    staged_k_decl = torch.empty((total_pages, PAGE_SIZE, num_kv_heads, head_dim), dtype=torch.float8_e4m3fn, device=device)
    staged_v_decl = torch.empty_like(staged_k_decl)
    staged_pages = torch.arange(total_pages, dtype=torch.int32, device=device).view(2, pages_per_row)

    token_ids = torch.ones((chunk_tokens,), dtype=torch.int32, device=device)
    token_ids[24:152] = IMAGE_TOKEN     # req 0 image (128 tokens)
    token_ids[192 + 16 : 192 + 272] = IMAGE_TOKEN  # req 1 image (256 tokens, post-wrap)
    image_span_end, row_request = compute_image_sliding_spans(token_ids, m.positions, m.query_start_loc, IMAGE_TOKEN)
    first_t = (m.starts - WINDOW).clamp_min(0)
    row_first = first_t[row_request.clamp_min(0).long()]
    staged_positions = torch.where(row_request >= 0, m.positions - row_first, m.positions)
    staged_span_end = torch.where(image_span_end >= 0, (image_span_end.long() - row_first).to(torch.int32), -1)

    k_descale = torch.tensor([math.sqrt(head_dim)], dtype=torch.float32, device=device)
    v_descale = torch.ones(1, dtype=torch.float32, device=device)
    sentinel = torch.randn((chunk_tokens, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    output = sentinel.clone()

    ws = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        st_op = PreparedWindowStaging(
            hk, hv, k_chunk, v_chunk, staged_k_decl, staged_v_decl, m, session=session, name="stage_img_b2"
        ).functional(session=session)
        img_op = PreparedImageSlidingAttention(
            q, staged_k_decl, staged_v_decl, staged_pages, staged_positions, staged_span_end, row_request, output,
            window=WINDOW, k_descale=k_descale, v_descale=v_descale,
            session=session, workspace=ws, name="img_slide_staged_b2", tuning_page_count=total_pages,
        ).functional(session=session, k_cache=None, v_cache=None)
        session.freeze()
        sk, sv = st_op(k_chunk, v_chunk, metadata=m)
        actual = img_op(
            q, sentinel.clone(),
            k_cache=sk, v_cache=sv, page_table=staged_pages,
            positions=staged_positions, image_span_end=staged_span_end, row_request=row_request,
        )
        torch.cuda.synchronize()

    assert not torch.isnan(actual.float()).any()
    assert torch.equal(actual[staged_span_end < 0], sentinel[staged_span_end < 0])
    expected = _reference_image_sliding(
        q, sk, sv, staged_pages, staged_positions, staged_span_end, row_request,
        float(k_descale.item()), float(v_descale.item()), WINDOW, head_dim ** -0.5, sentinel,
    )
    torch.testing.assert_close(actual[staged_span_end >= 0].float(), expected[staged_span_end >= 0], rtol=0.02, atol=0.02)


@CUDA
@torch.no_grad()
def test_bf16_cache_image_sliding_attention_parity():
    """Verify PreparedImageSlidingAttention matches FP32 reference with BF16 paged KV caches."""
    torch.manual_seed(922)
    device = torch.device("cuda:0")
    num_heads, num_kv_heads, head_dim = 16, 8, 256
    tokens = 384
    num_pages = 8
    q = (torch.randn(tokens, num_heads, head_dim, device=device) * 0.1).bfloat16()
    k_cache = (torch.randn(num_pages, PAGE_SIZE, num_kv_heads, head_dim, device=device) * 0.1).bfloat16()
    v_cache = (torch.randn(num_pages, PAGE_SIZE, num_kv_heads, head_dim, device=device) * 0.2).bfloat16()
    page_table = torch.arange(num_pages, dtype=torch.int32, device=device).view(1, num_pages)
    positions = torch.arange(tokens, dtype=torch.int64, device=device)
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    token_ids = torch.ones((tokens,), dtype=torch.int32, device=device)
    token_ids[32:288] = IMAGE_TOKEN
    image_span_end, row_request = compute_image_sliding_spans(token_ids, positions, cu, IMAGE_TOKEN)
    sentinel = torch.randn((tokens, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    output = sentinel.clone()

    ws = ScratchArena(device)
    with PreparationSession(device=device, autotune=False, compile_workers=1) as session:
        op = PreparedImageSlidingAttention(
            q, k_cache, v_cache, page_table, positions, image_span_end, row_request, output,
            window=WINDOW, softmax_scale=1.0, session=session, workspace=ws,
            name="img_slide_bf16", tuning_page_count=num_pages,
        ).functional(session=session, k_cache=None, v_cache=None)
        session.freeze()
        actual = op(
            q, sentinel.clone(), k_cache=k_cache, v_cache=v_cache, page_table=page_table,
            positions=positions, image_span_end=image_span_end, row_request=row_request,
        )
        torch.cuda.synchronize()

    assert torch.equal(actual[image_span_end < 0], sentinel[image_span_end < 0])
    expected = _reference_image_sliding(
        q, k_cache, v_cache, page_table, positions, image_span_end, row_request,
        1.0, 1.0, WINDOW, 1.0, sentinel,
    )
    torch.testing.assert_close(actual[image_span_end >= 0].float(), expected[image_span_end >= 0], rtol=1e-2, atol=1e-2)
