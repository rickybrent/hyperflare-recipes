# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Device-resident continuation state and paged KV storage for Gemma 4."""

import torch

from hyperflare_recipes.components.paged_state import PagedStateStorage
from hyperflare_recipes.components.parallel import TensorParallelShard
from ignition.components.state import StateLayout

from .config import PAGE_TOKENS

_KV_DTYPES = {
    "fp8": torch.float8_e4m3fn,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
}


def resolve_kv_dtype(kv_dtype: torch.dtype | str = torch.float8_e4m3fn) -> torch.dtype:
    if isinstance(kv_dtype, str):
        resolved = _KV_DTYPES.get(kv_dtype.lower())
        if resolved is not None:
            return resolved
    elif kv_dtype in (torch.float8_e4m3fn, torch.bfloat16):
        return kv_dtype
    raise ValueError(f"Gemma 4 KV cache dtype must be fp8/float8_e4m3fn or bf16/bfloat16, got {kv_dtype!r}")


def page_specs(
    config,
    shard: TensorParallelShard = TensorParallelShard(0, 1),
    *,
    kv_dtype: torch.dtype | str = torch.float8_e4m3fn,
):
    dtype = resolve_kv_dtype(kv_dtype)
    elem_bytes = torch.empty((), dtype=dtype).element_size()
    for index in range(config.first_kv_shared_layer):
        if config.is_sliding(index):
            continue
        kv_heads = config.local_kv_heads(index, shard)
        dim = config.layer_head_dim(index)
        key_size = PAGE_TOKENS * kv_heads * dim * elem_bytes
        half_value_size = PAGE_TOKENS * kv_heads * (dim // 2) * elem_bytes
        yield f"layer{index}.key", (key_size,)
        yield f"layer{index}.value_lo", (half_value_size,)
        yield f"layer{index}.value_hi", (half_value_size,)


def page_bytes(
    config,
    shard: TensorParallelShard = TensorParallelShard(0, 1),
    *,
    kv_dtype: torch.dtype | str = torch.float8_e4m3fn,
):
    return sum(shape[0] for _, shape in page_specs(config, shard, kv_dtype=kv_dtype))


def continuation_layout(
    config,
    shard: TensorParallelShard = TensorParallelShard(0, 1),
    *,
    kv_dtype: torch.dtype | str = torch.float8_e4m3fn,
):
    dtype = resolve_kv_dtype(kv_dtype)
    regions = []
    for index in range(config.first_kv_shared_layer):
        if config.is_sliding(index):
            kv_heads = config.local_kv_heads(index, shard)
            dim = config.layer_head_dim(index)
            for field in ("key", "value"):
                regions.append(
                    (f"layer{index}.{field}", (config.sliding_window, kv_heads, dim), dtype)
                )
    regions.extend((name, shape, torch.uint8) for name, shape in page_specs(config, shard, kv_dtype=dtype))
    regions.extend((
        ("last_hidden", (config.hidden,), torch.bfloat16),
        ("consumed_tokens", (1,), torch.int64),
    ))
    return StateLayout.from_specs(regions)


class ContinuationPool(PagedStateStorage):
    @classmethod
    def for_model(
        cls,
        config,
        slots,
        pages,
        device,
        *,
        shard: TensorParallelShard = TensorParallelShard(0, 1),
        kv_dtype: torch.dtype | str = torch.float8_e4m3fn,
        **kwargs,
    ):
        dtype = resolve_kv_dtype(kv_dtype)
        pool = cls.allocate(
            continuation_layout(config, shard, kv_dtype=dtype),
            page_specs(config, shard, kv_dtype=dtype),
            slots,
            pages,
            device,
            **kwargs,
        )
        pool.kv_dtype = dtype
        return pool

    def attention(self, layer, config, shard: TensorParallelShard = TensorParallelShard(0, 1)):
        donor = config.kv_source_layer(layer)
        if config.is_sliding(layer):
            return (
                self.reserved_region(f"layer{donor}.key"),
                self.reserved_region(f"layer{donor}.value"),
            )
        kv_dtype = getattr(self, "kv_dtype", torch.float8_e4m3fn)
        kv_heads = config.local_kv_heads(layer, shard)
        dim = config.layer_head_dim(layer)
        key_shape = (-1, PAGE_TOKENS, kv_heads, dim)
        value_shape = (-1, PAGE_TOKENS, kv_heads, dim // 2)
        return (
            self.pages[f"layer{donor}.key"].view(kv_dtype).view(key_shape),
            self.pages[f"layer{donor}.value_lo"].view(kv_dtype).view(value_shape),
            self.pages[f"layer{donor}.value_hi"].view(kv_dtype).view(value_shape),
        )
