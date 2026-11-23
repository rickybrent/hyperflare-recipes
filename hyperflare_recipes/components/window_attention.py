# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Paged attention over circular continuations and private candidate sequences."""

from dataclasses import dataclass, replace
import math

import torch
import triton
import triton.language as tl
from b12x._lib.compile_plan import attach_programs, compile_only_launches, load_programs
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


def stage_windows(history, values, output, metadata, *, counts=None, first_position=0):
    """Linearize circular prefixes and append packed candidates into caller storage."""
    m = metadata
    rows, window = m.starts.numel(), history.shape[1]
    capacity = output.shape[0] * output.shape[1] // rows
    offsets = torch.arange(capacity, device=values.device)[None, :]
    first = (m.starts - window).clamp_min(first_position)[:, None]
    positions = first + offsets
    previous = history[m.state_rows[:, None], positions.remainder(window)].to(values.dtype)
    indices = (m.query_start_loc[:-1, None] + positions - m.starts[:, None]).clamp(0, values.shape[0] - 1)
    current = values[indices]
    counts = m.lengths if counts is None else counts
    valid = (m.lengths[:, None] > 0) & (positions < (m.starts + counts)[:, None])
    staged = torch.where(valid[:, :, None, None],
        torch.where((positions < m.starts[:, None])[:, :, None, None], previous, current), 0)
    if output.dtype == torch.float8_e4m3fn:
        staged = staged.clamp(-448, 448)
    output.view(rows, capacity, *values.shape[1:]).copy_(staged.to(output.dtype))


def commit_window(history, values, row, start, length):
    """Retain the final window with token positions selecting their circular slots."""
    offsets = torch.arange(history.shape[1], device=values.device)
    end = start + length
    positions = end - 1 - (end - 1 - offsets).remainder(history.shape[1])
    current = values[(positions - start).clamp(0, values.shape[0] - 1)]
    rows = row.expand_as(offsets)
    previous = history[rows, offsets].to(values.dtype)
    committed = torch.where(((positions >= start) & (positions < end))[:, None, None], current, previous)
    if history.dtype == torch.float8_e4m3fn:
        committed = committed.clamp(-448, 448)
    history[rows, offsets] = committed.to(history.dtype)


@triton.jit
def _commit_circular_kv(
    Keys,
    Values,
    HistoryKeys,
    HistoryValues,
    Starts,
    Begins,
    Counts,
    StateRows,
    HKS: tl.constexpr,
    HVS: tl.constexpr,
    KS: tl.constexpr,
    VS: tl.constexpr,
    WIDTH: tl.constexpr,
    WINDOW: tl.constexpr,
    CLAMP_FP8: tl.constexpr,
    BLOCK: tl.constexpr,
):
    request, distance = tl.program_id(0), tl.program_id(1)
    dim = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    start = tl.load(Starts + request).to(tl.int64)
    count = tl.load(Counts + request).to(tl.int64)
    begin = tl.load(Begins + request).to(tl.int64)
    state = tl.load(StateRows + request).to(tl.int64)
    end = start + count
    absolute = end - 1 - distance
    ring_slot = absolute % WINDOW
    valid = (state > 0) & (count > 0) & (absolute >= start) & (dim < WIDTH)
    src_row = begin + absolute - start
    k_val = tl.load(Keys + src_row * KS + dim, valid, other=0.0)
    v_val = tl.load(Values + src_row * VS + dim, valid, other=0.0)
    if CLAMP_FP8:
        k_val = tl.clamp(k_val.to(tl.float32), -448.0, 448.0)
        v_val = tl.clamp(v_val.to(tl.float32), -448.0, 448.0)
    tl.store(HistoryKeys + state * HKS + ring_slot * WIDTH + dim, k_val, valid)
    tl.store(HistoryValues + state * HVS + ring_slot * WIDTH + dim, v_val, valid)


def commit_circular_kv(key, value, history_key, history_value, metadata, counts=None):
    """Batched circular KV commit for accepted prefixes across all active requests."""
    if key.ndim != 3 or value.shape != key.shape or history_key.ndim != 4 or history_value.shape != history_key.shape:
        raise ValueError("circular KV commit requires 3D candidate and 4D circular history tensors with matching shapes")
    if history_key.shape[2:] != key.shape[1:] or history_key.dtype != history_value.dtype:
        raise ValueError("circular KV commit requires matching head geometry and history dtypes")
    counts = metadata.lengths if counts is None else counts
    flat_k, flat_v = key.reshape(key.shape[0], -1), value.reshape(value.shape[0], -1)
    width = flat_k.shape[1]
    window = history_key.shape[1]
    clamp_fp8 = history_key.dtype == torch.float8_e4m3fn
    torch.library.wrap_triton(_commit_circular_kv)[
        (counts.numel(), min(window, key.shape[0]), triton.cdiv(width, 256))
    ](
        flat_k,
        flat_v,
        history_key,
        history_value,
        metadata.starts,
        metadata.query_start_loc,
        counts,
        metadata.state_rows,
        history_key.stride(0),
        history_value.stride(0),
        flat_k.stride(0),
        flat_v.stride(0),
        width,
        window,
        clamp_fp8,
        256,
    )


@dataclass(frozen=True)
class Query:
    operands: tuple
    first_position: int
    output_gate: tuple | None = None

    @property
    def rows(self):
        return self.operands[0][0][0]

    def scratch_bytes(self, tactic):
        splits, _ = tactic
        _, heads, dim = self.operands[0][0]
        return 0 if splits == 1 else self.rows * splits * heads * (dim + 1) * 4


def _tactics(query):
    direct = tuple((1, warps) for warps in (1, 2, 4))
    if query.rows > 64:
        return direct
    window = query.operands[3][0][1]
    return direct + tuple((splits, 2) for splits in (2, 4, 8, 16) if splits <= (window + 31) // 32)


CONTRACT = native_contract(component_id="flare.window_gqa", query_type=Query, tactics=_tactics)


@program_cache(scope="preparation")
def compile_attention(payload, tactic, ordinal):
    from b12x.attention.paged._selected_forward import _compile, launch_sparse_gqa_merge

    q = Query(**dict(payload))
    query, _key, _value, history_key, history_value, requests, _positions, _starts, _begins, _states, output = map(metadata_tensor, q.operands)
    splits, warps = tactic
    gate = metadata_tensor(q.output_gate) if q.output_gate is not None else None
    with torch.cuda.device(ordinal), compile_only_launches():
        _, raw = _compile(query=query, key_cache=history_key, value_cache=history_value,
            request_ids=requests, direct_output=splits == 1, return_lse=splits != 1,
            kv_warps=warps, selection_width=history_key.shape[1], first_position=q.first_position,
            output_gate=gate if splits == 1 else None)
        programs = {"attention": raw}
        if splits > 1:
            partial = torch.empty((query.shape[0], splits, *query.shape[1:]), device="meta", dtype=torch.float32)
            lse = torch.empty(partial.shape[:-1], device="meta", dtype=torch.float32)
            programs["merge"] = launch_sparse_gqa_merge(partial_output=partial, partial_lse=lse,
                output=output, output_lse=None, rows=q.rows, splits=splits, output_gate=gate)
        return programs


@dataclass(frozen=True)
class State:
    query: Query
    tactic: tuple
    programs: dict


def _run(query, key, value, history_key, history_value, requests, positions, starts, begins, states, output, output_gate, descale, scratch, softmax_scale, state):
    from b12x.attention.paged._selected_forward import (
        BFloat16,
        Float32,
        Float8E4M3FN,
        Int32,
        Int64,
        _fake_pointer,
        _pointer,
        current_cuda_stream,
        run_compiled,
    )

    splits, _ = state.tactic
    rows, heads, dim = query.shape
    partial, lse = _fake_pointer(Float32), _fake_pointer(Float32)
    if splits > 1:
        size = rows * splits * heads * dim
        storage = scratch.view(torch.float32)
        partial, lse = _pointer(storage[:size], Float32), _pointer(storage[size:], Float32)
    kv_type = Float8E4M3FN if history_key.dtype == torch.float8_e4m3fn else BFloat16
    continuation = tuple(_pointer(t, dtype) for t, dtype in zip((key, value, starts, begins, states),
        (BFloat16, BFloat16, Int64, Int32, Int64), strict=True))
    run_compiled(state.programs["attention"], (
        _pointer(query, BFloat16), _pointer(history_key, kv_type), _pointer(history_value, kv_type),
        _pointer(descale, Float32), _pointer(descale, Float32), _fake_pointer(Int32),
        _pointer(requests, Int64), _fake_pointer(Int32), _pointer(positions, Int64), partial, lse,
        _pointer(output, BFloat16),
        _pointer(output_gate, BFloat16) if output_gate is not None else _fake_pointer(BFloat16),
        continuation, history_key.shape[0], starts.numel(), 0,
        float(softmax_scale), rows, splits, current_cuda_stream()))
    if splits > 1:
        run_compiled(state.programs["merge"], (partial, lse, _pointer(output, BFloat16),
            _fake_pointer(Float32),
            _pointer(output_gate, BFloat16) if output_gate is not None else _fake_pointer(BFloat16),
            rows, splits, current_cuda_stream()))


@torch.library.custom_op("flare::window_gqa", mutates_args=("output", "scratch"))
def _execute(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, history_key: torch.Tensor,
             history_value: torch.Tensor, requests: torch.Tensor, positions: torch.Tensor,
             starts: torch.Tensor, begins: torch.Tensor, states: torch.Tensor, output: torch.Tensor,
             output_gate: torch.Tensor | None, descale: torch.Tensor, scratch: torch.Tensor,
             softmax_scale: float, handle: int) -> None:
    plan = plan_from_handle(handle)
    _run(query, key, value, history_key, history_value, requests, positions, starts, begins, states,
         output, output_gate, descale, scratch, softmax_scale, require_prepared(plan, plan.component_id, query.device))


@_execute.register_fake
def _fake(query, key, value, history_key, history_value, requests, positions, starts, begins, states, output, output_gate, descale, scratch, softmax_scale, handle):
    pass


class PreparedWindowAttention:
    """Read circular KV and packed candidates through b12x's paged-forward engine."""

    def __init__(self, query, key, value, history_key, history_value, metadata, output, *,
                 first_position=0, proposal=None, output_gate=None, softmax_scale=None,
                 session, workspace, name):
        self.key, self.value, self.proposal = key, value, proposal
        if (query.ndim != 3 or key.ndim != 3 or history_key.ndim != 4
                or query.dtype != torch.bfloat16 or query.shape[-1] not in (192, 256)
                or key.shape != value.shape or history_key.shape != history_value.shape
                or output.shape != query.shape or history_key.shape[2:] != key.shape[1:]
                or key.shape[-1] != query.shape[-1] or key.shape[1] <= 0 or history_key.shape[1] <= 0
                or query.shape[1] % key.shape[1] or query.shape[0] != key.shape[0]
                or key.dtype != torch.bfloat16 or value.dtype != key.dtype
                or history_key.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
                or output.dtype != query.dtype
                or any(not t.is_contiguous() for t in (query, key, value, output))
                or any(t.dtype != history_key.dtype or t.stride(-1) != 1 for t in (history_key, history_value))):
            raise ValueError("circular GQA requires BF16 queries, BF16 candidates, matching BF16/FP8 history, and 192/256-dimensional heads")
        if softmax_scale is not None:
            if isinstance(softmax_scale, bool) or not isinstance(softmax_scale, (int, float)) or not math.isfinite(softmax_scale) or softmax_scale <= 0:
                raise ValueError("circular GQA softmax_scale must be a positive finite float")
            self.softmax_scale = float(softmax_scale)
        else:
            self.softmax_scale = float(query.shape[-1] ** -0.5)
        m = metadata
        if proposal is not None:
            key, value = (t.flatten(0, 1) for t in proposal[:2])
        self.arguments = query, key, value, history_key, history_value, m.requests, m.positions, m.starts, m.query_start_loc, m.state_rows, output
        from b12x.attention.paged._selected_forward import _output_gate_stride

        _output_gate_stride(output_gate, query)
        self.output_gate = output_gate
        self.descale = persistent_empty((1,), torch.float32, query.device).fill_(1)
        q = Query(tuple(tensor_facts(x) for x in self.arguments), first_position,
                  tensor_facts(output_gate) if output_gate is not None else None)
        payload = CONTRACT.encode_query(q)

        device = query.device

        def materialize(tactic):
            programs = compile_attention(payload, tactic, device.index)
            load_programs(programs)
            return attach_programs(State(q, tactic, programs), *programs.values())

        self.plan = native_plan(contract=CONTRACT, query=q, device=query.device,
            scratch_bytes=q.scratch_bytes, materialize=materialize,
            compile_jobs=lambda t, d: (CompileJob.create(
                "hyperflare_recipes.components.window_attention:compile_attention", payload, t, d.ordinal),))

        declaration, declaration_gate, scale = self.arguments, self.output_gate, self.softmax_scale

        def call(state, benchmark=False):
            args, produce, gate = declaration, None, declaration_gate
            if benchmark:
                def inputs():
                    a, k, v, hk, hv, r, pos, start, begin, rows, out = declaration
                    a, k, v, r, pos, start, begin, rows, out = (
                        torch.empty_like(t) for t in (a, k, v, r, pos, start, begin, rows, out))
                    a.normal_(std=.2)
                    k.normal_(std=.2)
                    v.normal_(std=.2)
                    width = a.shape[0] // start.numel()
                    r.copy_(torch.arange(a.shape[0], device=a.device) // width)
                    pos.copy_(hk.shape[1] + torch.arange(a.shape[0], device=a.device) % width)
                    start.fill_(hk.shape[1])
                    begin.copy_(torch.arange(begin.numel(), device=a.device) * (k.shape[0] // start.numel()))
                    rows.fill_(1)
                    return (a, k, v, hk, hv, r, pos, start, begin, rows, out), a.clone()
                args, seed = benchmark_resource(session, ("window_gqa", name), inputs)
                produce = lambda: args[0].copy_(seed)
                if gate is not None:
                    gate = benchmark_resource(session, ("window_gqa_gate", name),
                        lambda: torch.empty_strided(output_gate.shape, output_gate.stride(),
                            device=output_gate.device, dtype=output_gate.dtype).normal_())
            scratch = planned_scratch((scratch_buffer_spec(CONTRACT.component_id,
                nbytes=q.scratch_bytes(state.tactic), device=query.device),), workspace)
            return PreparedCall(run=lambda: _run(*args, gate, self.descale, scratch, scale, state), output=args[-1], produce=produce,
                                owners=prime_owners(workspace, scratch, (args, gate)))
        declare(session, (self.plan.request(name=name, prepare_call=lambda state: replace(call(state, True), owners=()),
            benchmark_call=lambda state: call(state, True)),))
        register_finalizer(session, lambda: self._bind(workspace), workspace=workspace)

    def _bind(self, workspace):
        self.scratch = planned_scratch(self.plan.scratch_specs(), workspace)

    def functional(self, *, session):
        self.history = self.arguments[3:5]
        self.output_shape = tuple(self.arguments[-1].shape)
        register_finalizer(session, self._finalize_functional)
        return self

    def _finalize_functional(self):
        self.scratch_layout = tuple((tuple(spec.shape), spec.dtype) for spec in self.plan.scratch_specs())
        self.arguments = self.output_gate = self.scratch = None
        self.key = self.value = self.proposal = None

    def __call__(self, query=None, key=None, value=None, metadata=None, *, output_gate=None, proposal=None):
        if self.arguments is not None:
            return self._bound_call()
        if proposal is not None:
            keys, values, step = proposal
            keys[:, step].copy_(key)
            values[:, step].copy_(value)
            key, value = keys.flatten(0, 1), values.flatten(0, 1)
        m = metadata
        output = torch.zeros(self.output_shape, dtype=query.dtype, device=query.device)
        scratch, = (torch.empty(shape, dtype=dtype, device=query.device) for shape, dtype in self.scratch_layout)
        scale = getattr(self, "softmax_scale", float(query.shape[-1] ** -0.5))
        _execute(query, key, value, *self.history, m.requests, m.positions, m.starts,
                 m.query_start_loc, m.state_rows, output, output_gate, self.descale, scratch, scale, self.plan.handle)
        return output

    def _bound_call(self):
        if self.proposal is not None:
            key, value, step = self.proposal
            key[:, step].copy_(self.key)
            value[:, step].copy_(self.value)
        scale = getattr(self, "softmax_scale", float(self.arguments[0].shape[-1] ** -0.5))
        _execute(*self.arguments, self.output_gate, self.descale, self.scratch, scale, self.plan.handle)
        return self.arguments[-1]

    def scratch_tensors(self):
        return ()
