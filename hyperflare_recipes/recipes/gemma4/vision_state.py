# SPDX-FileCopyrightText: Copyright (c) 2026 Flare community contributors
# SPDX-License-Identifier: Apache-2.0

"""Gemma 4 patch uploads and generation-scoped image feature publication."""

from dataclasses import dataclass

import torch

from ignition.components.vision_delivery import PreparedFeaturePublication

FEATURES = 4480
PATCH_CAPACITIES = (630, 1260, 2520, 5040, 10080)
PATCH_FEATURES = 3 * 16 * 16
POOLING = 3


@dataclass(frozen=True)
class VisionInputs:
    vision_pixel_values: torch.Tensor
    vision_position_ids: torch.Tensor
    vision_cu_seqlens: torch.Tensor
    vision_num_patches: torch.Tensor
    vision_slot: torch.Tensor
    vision_generation: torch.Tensor
    vision_feature_offset: torch.Tensor

    @classmethod
    def allocate(cls, device):
        patches = max(PATCH_CAPACITIES)
        return cls(
            torch.zeros((patches, PATCH_FEATURES), dtype=torch.float32, device=device),
            torch.full((patches, 2), -1, dtype=torch.int64, device=device),
            torch.zeros(2, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.full((1,), -1, dtype=torch.int32, device=device),
            torch.zeros(1, dtype=torch.int64, device=device),
            torch.zeros(1, dtype=torch.int32, device=device),
        )

    def buffers(self):
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


class PreparedVisionGeneration:
    def __init__(self, encoder, inputs, state, completion, frame, sequence):
        self.encoder, self.inputs, self.state = encoder, inputs, state
        self.errors = torch.zeros(1, dtype=torch.int32, device=completion.records.device)
        self.publish = PreparedFeaturePublication(state, inputs, completion, frame, sequence)

    def __call__(self):
        e, i, s = self.encoder, self.inputs, self.state
        r, m = s.requests, e.metadata
        capacity = e.scratch.patches.shape[0]
        slot = i.vision_slot.long().clamp(0, r.capacity - 1)
        count, offset = i.vision_num_patches, i.vision_feature_offset
        features = count // (POOLING * POOLING)
        current = (i.vision_slot >= 0) & (i.vision_slot < r.capacity)
        current &= (i.vision_generation == r.generation[slot]) & (r.status[slot] == 1)
        valid = current & (count > 0) & (count <= capacity) & (count % (POOLING * POOLING) == 0)
        valid &= (offset >= 0) & (offset + features <= s.arena.shape[1])
        valid &= (offset == 0) | ((s.generation[slot] == i.vision_generation) & (s.count[slot] == offset))
        valid &= (i.vision_cu_seqlens[0] == 0) & (i.vision_cu_seqlens[1] == count)
        valid &= (s.feature_slot >= 0) & (s.feature_slot < s.arena.shape[0])
        live = torch.arange(capacity, device=count.device) < count
        pixels, positions = i.vision_pixel_values[:capacity], i.vision_position_ids[:capacity]
        valid &= (~live[:, None] | torch.isfinite(pixels)).all()
        valid &= (~live[:, None] | ((positions >= 0) & (positions < capacity))).all()
        self.errors.copy_(~valid)
        e.scratch.patches.copy_(torch.where(valid & live[:, None], pixels, 0))
        m.position_ids.copy_(torch.where(valid & live[:, None], positions, -1))
        m.cu_seqlens.copy_(torch.where(valid, i.vision_cu_seqlens, 0))
        m.num_patches.copy_(torch.where(valid, count, 0))
        output = e(e.scratch.patches, m.position_ids)
        feature_live = torch.arange(capacity // (POOLING * POOLING), device=count.device) < features
        finite = (~feature_live[:, None] | torch.isfinite(output)).all()
        self.errors.copy_(self.errors | (~finite).int() * 64)
        self.publish(output, m.num_patches // (POOLING * POOLING), self.errors)
