# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""CUDA parity, rollback, frozen-KV Assistant, and RankCosts tests for Gemma 4 circular sliding KV."""

from dataclasses import replace
import math
import os

import pytest
import torch

from hyperflare_recipes.components.paged_attention import PreparedPagedAttention
from hyperflare_recipes.components.paged_batch import Metadata
from hyperflare_recipes.components.parallel import TensorParallelShard
from hyperflare_recipes.components.window_attention import PreparedWindowAttention, commit_circular_kv
from hyperflare_recipes.components.window_staging import PreparedWindowStaging
from ignition.preparation.compilation import compile_forward
from ignition.preparation.scratch import ScratchArena
from ignition.preparation.tuning import RecordedPreparationSession
from hyperflare_recipes.recipes.gemma4.config import PAGE_TOKENS, ModelConfig
from hyperflare_recipes.recipes.gemma4.pool import ContinuationPool, continuation_layout, page_bytes
from hyperflare_recipes.recipes.gemma4.recipe import RECIPE
from tests.recipes.test_gemma4 import _dense_31b_raw, _e2b_raw, _e4b_raw, _gemma4_12b_raw, _moe_26b_raw

CUDA = pytest.mark.skipif(os.environ.get("FLARE_RUN_CUDA_TESTS") != "1", reason="requires an assigned GPU")


def _ref_sliding_unit_scale(q, k_full, v_full, positions, window):
    """Exact FP32 unit-scale causal sliding-window reference over contiguous history [0..pos]."""
    heads, groups = q.shape[1], k_full.shape[1]
    gqa = heads // groups
    out = torch.zeros_like(q, dtype=torch.float32)
    for i in range(q.shape[0]):
        pos = int(positions[i])
        if pos < 0:
            continue
        lo = max(0, pos - window + 1)
        ks = k_full[lo : pos + 1].float()
        vs = v_full[lo : pos + 1].float()
        for h in range(heads):
            g = h // gqa
            logits = q[i, h].float() @ ks[:, g].T  # unit attention scaling (1.0)
            out[i, h] = logits.softmax(-1) @ vs[:, g]
    return out


def test_rank_costs_match_circular_pool_for_all_variants_and_tp():
    factories = {
        "gemma-4-e2b": _e2b_raw,
        "gemma-4-e4b": _e4b_raw,
        "gemma-4-12b": _gemma4_12b_raw,
        "gemma-4-26b-a4b": _moe_26b_raw,
        "gemma-4-31b": _dense_31b_raw,
    }
    by_alias = {cp.alias: cp for cp in RECIPE.checkpoints}
    for alias, factory in factories.items():
        config = ModelConfig.from_dict(factory())
        cp = by_alias[alias]
        for ws in (1, 2, 4):
            shard = TensorParallelShard(0, ws)
            costs = RECIPE.costs(ws, None, cp)
            assert costs.state_row_bytes == continuation_layout(config, shard).row_bytes, f"{alias} TP{ws}"
            assert costs.page_bytes == page_bytes(config, shard), f"{alias} TP{ws}"


@CUDA
@torch.no_grad()
def test_batched_commit_circular_kv_decode_verify_rollback_and_fp8_clamp():
    torch.manual_seed(101)
    device = torch.device("cuda:0")
    window, groups, dim, batch, width = 1024, 16, 256, 4, 6
    tokens = batch * width

    # Allocate non-contiguous continuation-style history with stride(0) > window * groups * dim.
    row_pad = 4096
    backing_k = torch.full((batch + 2, window * groups * dim + row_pad), 7.0, device=device, dtype=torch.float8_e4m3fn)
    backing_v = torch.full((batch + 2, window * groups * dim + row_pad), -3.0, device=device, dtype=torch.float8_e4m3fn)
    hk = backing_k[:, : window * groups * dim].view(batch + 2, window, groups, dim)
    hv = backing_v[:, : window * groups * dim].view(batch + 2, window, groups, dim)
    orig_k, orig_v = hk.clone(), hv.clone()

    starts = torch.tensor([0, 510, 1020, 1030], dtype=torch.int64, device=device)
    lengths = torch.tensor([width, 0, width, width], dtype=torch.int32, device=device)
    counts = torch.tensor([6, 0, 4, 2], dtype=torch.int32, device=device)
    state_rows = torch.tensor([1, 2, 3, 4], dtype=torch.int64, device=device)

    m = Metadata.allocate(tokens, batch, 1, device, samples=tokens)
    m.set(starts, lengths, state_rows, m.page_table,
          torch.zeros(batch, width, dtype=torch.int64, device=device), torch.tensor([batch], device=device))

    key = (torch.randn(tokens, groups, dim, device=device) * 10.0).bfloat16()
    val = (torch.randn(tokens, groups, dim, device=device) * 10.0).bfloat16()
    # Inject >448 outliers to confirm FP8 saturation clamp prevents NaN.
    key[0, 0, 0] = 1000.0
    val[0, 0, 1] = -900.0

    commit_fn = compile_forward(lambda: commit_circular_kv(key, val, hk, hv, m, counts))
    commit_fn()
    torch.cuda.synchronize()

    assert not torch.isnan(hk.float()).any()
    assert not torch.isnan(hv.float()).any()
    assert float(hk[1, 0, 0, 0]) == 448.0
    assert float(hv[1, 0, 0, 1]) == -448.0

    for b in range(batch):
        row = int(state_rows[b])
        st = int(starts[b])
        cnt = int(counts[b])
        begin = int(m.query_start_loc[b])
        for d in range(cnt):
            slot = (st + d) % window
            ek = key[begin + d].clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            ev = val[begin + d].clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
            torch.testing.assert_close(hk[row, slot].float(), ek.float(), atol=0, rtol=0)
            torch.testing.assert_close(hv[row, slot].float(), ev.float(), atol=0, rtol=0)
        # Rejected draft slots must remain untouched (preserving history from 1024 tokens ago).
        for d in range(cnt, width):
            slot = (st + d) % window
            torch.testing.assert_close(hk[row, slot].float(), orig_k[row, slot].float(), atol=0, rtol=0)
            torch.testing.assert_close(hv[row, slot].float(), orig_v[row, slot].float(), atol=0, rtol=0)
    # Row 0 (null) and inactive row 2 must be completely untouched.
    torch.testing.assert_close(hk[0].float(), orig_k[0].float(), atol=0, rtol=0)
    torch.testing.assert_close(hk[2].float(), orig_k[2].float(), atol=0, rtol=0)


@CUDA
@torch.no_grad()
def test_prefill_staging_and_decode_verify_wraparound_parity():
    torch.manual_seed(202)
    device = torch.device("cuda:0")
    heads, groups, dim, window = 32, 16, 256, 1024
    chunk1, chunk2 = 1200, 320  # chunk1 wraps ring (1200 > 1024); chunk2 continues across wrap
    total_prefill = chunk1 + chunk2

    k_all = (torch.randn(total_prefill + 8, groups, dim, device=device) * 0.1).bfloat16()
    v_all = (torch.randn(total_prefill + 8, groups, dim, device=device) * 0.2).bfloat16()
    q_all = (torch.randn(total_prefill + 8, heads, dim, device=device) * 0.1).bfloat16()
    # FP8-quantized history for exact reference comparison of historical positions.
    k_fp8 = k_all.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).bfloat16()
    v_fp8 = v_all.clamp(-448.0, 448.0).to(torch.float8_e4m3fn).bfloat16()

    hk = torch.full((3, window, groups, dim), float("nan"), device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    hv = torch.full_like(hk, 0)

    k_descale = torch.tensor([math.sqrt(dim)], dtype=torch.float32, device=device)
    v_descale = torch.ones(1, dtype=torch.float32, device=device)

    # Chunk 1 prefill (start=0, len=1200) -> commit -> Chunk 2 prefill (start=1200, len=320) -> commit
    for start, length in ((0, chunk1), (chunk1, chunk2)):
        pages = (length + window + PAGE_TOKENS - 1) // PAGE_TOKENS
        staged_k = torch.empty((pages, PAGE_TOKENS, groups, dim), dtype=torch.float8_e4m3fn, device=device)
        staged_v = torch.empty((pages, PAGE_TOKENS, groups, dim), dtype=torch.float8_e4m3fn, device=device)
        staged_pages = torch.arange(pages, dtype=torch.int32, device=device).view(1, -1)
        m = Metadata.allocate(length, 1, 16, device, samples=length)
        m.set(torch.tensor([start], dtype=torch.int64, device=device),
              torch.tensor([length], dtype=torch.int32, device=device),
              torch.tensor([1], dtype=torch.int64, device=device),
              m.page_table, torch.zeros(1, length, dtype=torch.int64, device=device),
              torch.tensor([1], dtype=torch.int32, device=device))
        staged_lengths = torch.tensor([min(start, window) + length], dtype=torch.int32, device=device)
        out = torch.empty((length, heads, dim), dtype=torch.bfloat16, device=device)
        q_c, k_c, v_c = q_all[start : start + length].contiguous(), k_all[start : start + length].contiguous(), v_all[start : start + length].contiguous()
        arena = ScratchArena(device)
        with RecordedPreparationSession(device=device, checkpoint=f"g4-pf-{start}", workload_schema="g4-pf", numerical_policy="default") as session:
            session.register_scratch(arena)
            st_op = PreparedWindowStaging(hk, hv, k_c, v_c, staged_k, staged_v, m, session=session, name="stage").functional(session=session)
            at_op = PreparedPagedAttention(
                q_c, staged_k, staged_v, staged_pages, staged_lengths, m.query_start_loc, out,
                mode="prefill", window=window - 1, width=1, k_descale=k_descale, v_descale=v_descale,
                session=session, workspace=arena, name="attn", tuning_page_count=pages,
            ).functional(session=session, k_cache=None, v_cache=None, page_table=None, cache_seqlens=None, cu_seqlens_q=None)
            session.tune()
            arena.release_preparation()
            sk, sv = st_op(k_c, v_c, metadata=m)
            actual = at_op(q_c, k_cache=sk, v_cache=sv, page_table=staged_pages, cache_seqlens=staged_lengths, cu_seqlens_q=m.query_start_loc)
            commit_circular_kv(k_c, v_c, hk, hv, m, m.lengths)
            torch.cuda.synchronize()
            # Check selected rows across chunk against exact FP32 reference.
            ref = _ref_sliding_unit_scale(q_c, k_fp8, v_fp8, m.positions, window)
            torch.testing.assert_close(actual.float(), ref, atol=0.02, rtol=0.03)

    # Now test Assistant frozen-KV draft read at before=total_prefill (1520, wrapped!)
    # sliding_m = replace(m, starts=before), positions = before - 1 -> 100% history reads.
    before = torch.tensor([total_prefill, 0], dtype=torch.int64, device=device)
    active = torch.tensor([True, False], dtype=torch.bool, device=device)
    m_asst = Metadata.pack(
        (before - 1).clamp_min(0), active.int(),
        torch.tensor([1, 2], dtype=torch.int64, device=device),
        torch.full((2, 16), -1, dtype=torch.int32, device=device),
        torch.zeros((2, 1), dtype=torch.int64, device=device),
        torch.tensor([1], dtype=torch.int32, device=device),
        token_capacity=2, sample_count=2,
    )
    sliding_m = replace(m_asst, starts=before)
    q_asst = torch.cat((q_all[total_prefill : total_prefill + 1], torch.randn(1, heads, dim, device=device, dtype=torch.bfloat16)), dim=0)
    # Poison dummy candidate tensors with huge values to prove frozen-KV NEVER reads candidates!
    poison = torch.full((2, groups, dim), 100.0, dtype=torch.bfloat16, device=device)
    out_asst = torch.empty_like(q_asst)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="g4-asst", workload_schema="g4-asst", numerical_policy="default") as session:
        session.register_scratch(arena)
        op_asst = PreparedWindowAttention(
            q_asst, poison, poison, hk, hv, sliding_m, out_asst, softmax_scale=1.0,
            session=session, workspace=arena, name="asst_win",
        ).functional(session=session)
        session.tune()
        arena.release_preparation()
        actual_asst = op_asst(q_asst, poison, poison, sliding_m)
        torch.cuda.synchronize()
        ref_asst = _ref_sliding_unit_scale(q_asst[:1], k_fp8, v_fp8, torch.tensor([total_prefill - 1], device=device), window)
        torch.testing.assert_close(actual_asst[:1].float(), ref_asst, atol=0.02, rtol=0.03)
        assert not actual_asst[1].any()

    # Now test verify width=6 with rollback at accepted=2, then follow-up decode at start=total_prefill+2!
    w = 6
    m_ver = Metadata.allocate(w, 1, 16, device, samples=w)
    m_ver.set(torch.tensor([total_prefill], dtype=torch.int64, device=device),
              torch.tensor([w], dtype=torch.int32, device=device),
              torch.tensor([1], dtype=torch.int64, device=device),
              m_ver.page_table, torch.zeros(1, w, dtype=torch.int64, device=device),
              torch.tensor([1], dtype=torch.int32, device=device))
    q_ver = q_all[total_prefill : total_prefill + w].contiguous()
    k_ver = k_all[total_prefill : total_prefill + w].contiguous()
    v_ver = v_all[total_prefill : total_prefill + w].contiguous()
    out_ver = torch.empty_like(q_ver)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="g4-ver", workload_schema="g4-ver", numerical_policy="default") as session:
        session.register_scratch(arena)
        op_ver = PreparedWindowAttention(
            q_ver, k_ver, v_ver, hk, hv, m_ver, out_ver, softmax_scale=1.0,
            session=session, workspace=arena, name="ver_win",
        ).functional(session=session)
        session.tune()
        arena.release_preparation()
        actual_ver = op_ver(q_ver, k_ver, v_ver, m_ver)
        # Accept only 2 tokens (rejecting tokens 2..5).
        commit_circular_kv(k_ver, v_ver, hk, hv, m_ver, torch.tensor([2], dtype=torch.int32, device=device))
        torch.cuda.synchronize()
        # Verify mixed FP8 history (< total_prefill) + BF16 candidate (>= total_prefill) parity.
        k_mix = torch.cat((k_fp8[:total_prefill], k_ver), dim=0)
        v_mix = torch.cat((v_fp8[:total_prefill], v_ver), dim=0)
        ref_ver = _ref_sliding_unit_scale(q_ver, k_mix, v_mix, m_ver.positions, window)
        torch.testing.assert_close(actual_ver.float(), ref_ver, atol=0.02, rtol=0.03)

    # Follow-up decode at total_prefill + 2 must see preserved history at (total_prefill+2..5) - 1024!
    pos_next = total_prefill + 2
    m_dec = Metadata.allocate(1, 1, 16, device, samples=1)
    m_dec.set(torch.tensor([pos_next], dtype=torch.int64, device=device),
              torch.tensor([1], dtype=torch.int32, device=device),
              torch.tensor([1], dtype=torch.int64, device=device),
              m_dec.page_table, torch.zeros(1, 1, dtype=torch.int64, device=device),
              torch.tensor([1], dtype=torch.int32, device=device))
    q_dec = q_all[pos_next : pos_next + 1].contiguous()
    k_dec = k_all[pos_next : pos_next + 1].contiguous()
    v_dec = v_all[pos_next : pos_next + 1].contiguous()
    out_dec = torch.empty_like(q_dec)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="g4-dec", workload_schema="g4-dec", numerical_policy="default") as session:
        session.register_scratch(arena)
        op_dec = PreparedWindowAttention(
            q_dec, k_dec, v_dec, hk, hv, m_dec, out_dec, softmax_scale=1.0,
            session=session, workspace=arena, name="dec_win",
        ).functional(session=session)
        session.tune()
        arena.release_preparation()
        actual_dec = op_dec(q_dec, k_dec, v_dec, m_dec)
        torch.cuda.synchronize()
        k_mix2 = torch.cat((k_fp8[:pos_next], k_dec), dim=0)
        v_mix2 = torch.cat((v_fp8[:pos_next], v_dec), dim=0)
        ref_dec = _ref_sliding_unit_scale(q_dec, k_mix2, v_mix2, m_dec.positions, window)
        torch.testing.assert_close(actual_dec.float(), ref_dec, atol=0.02, rtol=0.03)


@CUDA
@torch.no_grad()
def test_kv_shared_donor_sliding_layer_parity_across_wraparound():
    """Verify E4B-style KV-shared sliding layers reusing donor history + current K/V across wraparound."""
    torch.manual_seed(303)
    device = torch.device("cuda:0")
    heads, groups, dim, window, batch, width = 8, 2, 256, 512, 2, 4
    tokens = batch * width
    hk = (torch.randn(batch + 1, window, groups, dim, device=device) * 0.1).bfloat16().to(torch.float8_e4m3fn)
    hv = (torch.randn(batch + 1, window, groups, dim, device=device) * 0.2).bfloat16().to(torch.float8_e4m3fn)
    q_donor = (torch.randn(tokens, heads, dim, device=device) * 0.1).bfloat16()
    q_shared = (torch.randn(tokens, heads, dim, device=device) * 0.1).bfloat16()
    k_donor = (torch.randn(tokens, groups, dim, device=device) * 0.1).bfloat16()
    v_donor = (torch.randn(tokens, groups, dim, device=device) * 0.2).bfloat16()
    m = Metadata.allocate(tokens, batch, 4, device, samples=tokens)
    starts = torch.tensor([600, 510], dtype=torch.int64, device=device)
    m.set(starts, torch.full((batch,), width, dtype=torch.int32, device=device),
          torch.arange(1, batch + 1, dtype=torch.int64, device=device), m.page_table,
          torch.zeros(batch, width, dtype=torch.int64, device=device), torch.tensor([batch], dtype=torch.int32, device=device))
    out_d, out_s = torch.empty_like(q_donor), torch.empty_like(q_shared)
    arena = ScratchArena(device)
    with RecordedPreparationSession(device=device, checkpoint="g4-shared", workload_schema="g4-shared", numerical_policy="default") as session:
        session.register_scratch(arena)
        op_d = PreparedWindowAttention(q_donor, k_donor, v_donor, hk, hv, m, out_d, softmax_scale=1.0,
                                       session=session, workspace=arena, name="donor").functional(session=session)
        op_s = PreparedWindowAttention(q_shared, k_donor, v_donor, hk, hv, m, out_s, softmax_scale=1.0,
                                       session=session, workspace=arena, name="shared").functional(session=session)
        session.tune()
        arena.release_preparation()
        # Donor runs first (without committing yet, as in verify mode), shared layer consumes donor (k_donor, v_donor).
        actual_d = op_d(q_donor, k_donor, v_donor, m)
        actual_s = op_s(q_shared, k_donor, v_donor, m)
        torch.cuda.synchronize()
        gqa = heads // groups
        for row in range(batch):
            st = int(starts[row])
            for col in range(width):
                tok = row * width + col
                pos = int(m.positions[tok])
                positions = torch.arange(max(0, pos - window + 1), pos + 1, device=device)
                old = positions < st
                ks = torch.cat((hk[row + 1, positions[old] % window].float(), k_donor[row * width + (positions[~old] - st)].float()), dim=0)
                vs = torch.cat((hv[row + 1, positions[old] % window].float(), v_donor[row * width + (positions[~old] - st)].float()), dim=0)
                for q_t, act in ((q_donor, actual_d), (q_shared, actual_s)):
                    for h in range(heads):
                        g = h // gqa
                        exp = (q_t[tok, h].float() @ ks[:, g].T).softmax(-1) @ vs[:, g]
                        torch.testing.assert_close(act[tok, h].float(), exp, atol=0.02, rtol=0.03)
