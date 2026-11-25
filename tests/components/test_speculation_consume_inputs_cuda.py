# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Bit-exact parity for fused Triton probability validation and in-place masking."""

import os

import pytest
import torch

from ignition.components.sampling import DeviceRequestState, RequestAdmission
from ignition.preparation.compilation import compile_forward
from hyperflare_recipes.components.speculation import (
    PreparedStochasticVerification,
    SpeculativeRngState,
)
from tests.components.probability_mask_reference import FunctionalVerificationReference

pytestmark = pytest.mark.skipif(
    os.environ.get("FLARE_RUN_CUDA_TESTS") != "1",
    reason="requires an explicitly assigned CUDA device and FLARE_RUN_CUDA_TESTS=1",
)


@pytest.mark.parametrize("limited", [False, True])
@torch.no_grad()
def test_consume_inputs_matches_reference_compiled_and_captured(limited):
    batch, width, vocab = 4, 4, 256
    device = torch.device("cuda:0")
    state = DeviceRequestState.allocate(device, vocab_size=vocab, capacity=batch)
    for row in range(batch):
        state.admit(
            row,
            RequestAdmission(generation=7, seed=101 + row, max_new_tokens=20, eos_token_ids=()),
            [1, 2],
        )
    generator = torch.Generator(device=device).manual_seed(73)
    q = torch.rand((batch, width - 1, vocab), generator=generator, device=device)
    q[..., 0] = -0.0
    q /= q.sum(-1, keepdim=True)
    p = torch.rand((batch, width, vocab), generator=generator, device=device)
    p[..., 1] = -0.0
    p /= p.sum(-1, keepdim=True)
    q[1, 1, 7] = float("nan")
    p[2, 0, 9] = -1e-5
    if limited:
        q[3, 1:] = float("nan")
        p[3, 2:] = float("inf")
    ids = torch.randint(0, vocab, (batch, width - 1), device=device, dtype=torch.int64)
    slots = torch.arange(batch, device=device, dtype=torch.int32)
    epochs = torch.full((batch,), 7, device=device, dtype=torch.int64)
    live = torch.tensor([batch], device=device, dtype=torch.int32)
    limits = torch.tensor([width, width, width, 2], device=device, dtype=torch.int32) if limited else None
    rng = SpeculativeRngState.allocate(device, capacity=batch)
    rng.generation.fill_(7)
    rng.target_offset.fill_(44)

    ref = FunctionalVerificationReference(
        q.clone(), ids, p.clone(), state, slots, epochs, live, rng=rng, token_limits=limits
    )
    ref.prepare()
    torch._dynamo.reset()

    def ref_forward():
        ref()

    ref_call = compile_forward(ref_forward)
    ref_call()
    torch.cuda.synchronize()
    expected = {name: tensor.cpu().clone() for name, tensor in vars(ref.output).items()}

    op = PreparedStochasticVerification(
        q.clone(), ids, p.clone(), state, slots, epochs, live, rng=rng, token_limits=limits
    )
    op.prepare()
    op.functional()
    scratch = {
        name: torch.empty(shape, device=device, dtype=dtype)
        for name, (shape, dtype) in PreparedStochasticVerification.scratch_layout(
            batch, width, vocab, consume_inputs=True
        ).items()
    }
    q_in = q.clone()
    p_in = p.clone()
    q_valid = torch.ones((batch, width - 1), dtype=torch.bool, device=device)
    p_valid = torch.ones((batch, width), dtype=torch.bool, device=device)
    out_buf = {
        name: torch.empty_like(tensor, device=device)
        for name, tensor in expected.items()
    }

    def forward():
        res = op.run(
            scratch,
            q_in,
            p_in,
            p_valid,
            drafts=ids,
            draft_valid=q_valid,
            disjoint=True,
            consume_inputs=True,
        )
        for name, tensor in out_buf.items():
            tensor.copy_(getattr(res, name))

    torch._dynamo.reset()
    call = compile_forward(forward)
    q_in.copy_(q)
    p_in.copy_(p)
    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    q_in.copy_(q)
    p_in.copy_(p)
    with torch.cuda.graph(graph):
        call()
    q_in.copy_(q)
    p_in.copy_(p)
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_stats()["allocation.all.allocated"]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == alloc
    for name, expected_tensor in expected.items():
        assert torch.equal(
            out_buf[name].cpu().contiguous().view(torch.uint8),
            expected_tensor.contiguous().view(torch.uint8),
        ), (limited, name)
    graph.reset()
