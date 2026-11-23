# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 text inputs, continuation state, and paged attention storage."""

from ignition.preparation import contract as c
from ignition.preparation import runtime_core_pb2 as p

from .config import PAGE_TOKENS


FRONTEND = c.frontend(
    tools=p.GEMMA_CALL,
    assistant_marker="<|turn>model\n",
    message_end_markers=["<turn|>", "<|tool_response>", "<eos>"],
    prefix_boundary=c.prefix_boundary(
        ["system", "developer"],
        append_role="user",
        minimum_messages=0,
        required_start="<bos><|turn>system\n",
        required_marker="<turn|>",
    ),
)
FRONTEND["reasoning"] = dict(
    kind=p.CHANNEL_THOUGHT,
    open="<|channel>thought",
    close="<channel|>",
    wrapping_newlines=1,
    terminal_markers=["<|tool_call>"],
)


VISION = dict(
    features_per_request=4480,
    max_images=4,
    patches_per_feature=9,
    patch_capacities=[630, 1260, 2520, 5040, 10080],
)
VISION_FRONTEND = {
    **FRONTEND,
    "image_preparation": p.GEMMA4_IMAGE_PREPARATION,
    "image_layout": p.BRACKETED_IMAGE_LAYOUT,
    "image_tokens": dict(
        feature_token=258880,
        span=dict(
            start_token=255999,
            end_token=258882,
            max_tokens=1122,
            interior_tokens=[],
            unscoped_tokens=[],
        ),
    ),
}


def prepare(
    signatures,
    buffers,
    pool,
    *,
    vocabulary,
    world_size: int = 1,
    mtp: bool = False,
    vision: bool = False,
    image_slots: int = 0,
):
    inputs = {
        kind: c.text_inputs(kind)
        for kind in {s.kind for s in signatures.values()}
        if kind != "vision"
    }
    vision_fields = {field for s in signatures.values() if s.kind == "vision" for field in s.inputs}
    inputs["vision"] = c.vision_inputs(vision_fields) if vision_fields else {}
    if world_size == 1 and "decode" in inputs:
        inputs["decode"].pop("token_ids", None)
    continuation = c.state_view("state_pool", pool.layout.row_bytes)
    consumed = pool.layout.region("consumed_tokens")
    cursor = (
        dict(
            state=c.state_view(
                "state_pool", pool.layout.row_bytes, offset=consumed.offset, nbytes=consumed.nbytes
            ),
            lag_tokens=0,
        )
        if mtp
        else None
    )
    pending = c.state_view("request_pending_token", 8)
    regions = [
        c.history_region(buffers, f"cache.{name}", tensor, p.PAGED_KEY_VALUE)
        for name, tensor in pool.pages.items()
    ]
    return c.model_contract(
        vocabulary=vocabulary,
        page_tokens=PAGE_TOKENS,
        page_tables=[dict(buffer="page_tables", page_offset=0)],
        regions=[
            dict(kind=p.CONTINUATION_STATE, storage=continuation, tokens_per_record=1, window_tokens=0),
            *regions,
        ],
        graphs=c.graph_contracts(signatures, inputs, continuation=continuation, pending=pending, cursor=cursor),
        vision={**VISION, "feature_slots": image_slots} if vision else None,
        frontend=VISION_FRONTEND if vision else FRONTEND,
    )
