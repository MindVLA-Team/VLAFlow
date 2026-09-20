# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any

import torch
from pydantic import Field

from .base import ModalityTransform


class StateActionZeroPadTransform(ModalityTransform):
    """Pads action (and optionally state) tensors to target_dim with zeros along the last axis.

    Used to normalize single-arm action representations to a fixed-width dual-arm format,
    e.g. padding 7-dim single-arm EEF actions to 14-dim.

    The transform pads the last dimension of each matched key from its current width
    to ``target_dim`` by appending zeros. Keys whose last dimension already equals
    ``target_dim`` are left unchanged.  Keys whose last dimension exceeds ``target_dim``
    will raise a ValueError.

    Example::

        # Single-arm 7-D (pos[3]+quat[4]) -> 14-D (right-arm zeros)
        pad = StateActionZeroPadTransform(
            apply_to=["action"],
            target_dim=14,
        )
    """

    apply_to: list[str] = Field(
        default_factory=lambda: ["action"],
        description="Modality keys (or key prefixes) to pad. "
        "Matched against the top-level key of the data dict (e.g. 'action', 'state').",
    )
    target_dim: int = Field(
        default=14,
        description="Target dimension for the last axis of matched tensors.",
        gt=0,
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in list(data.keys()):
            # Match against apply_to: exact key match or any key starting with "{prefix}."
            if not any(key == prefix or key.startswith(prefix + ".") for prefix in self.apply_to):
                continue

            tensor = data[key]
            if not isinstance(tensor, torch.Tensor):
                continue

            current_dim = tensor.shape[-1]
            if current_dim == self.target_dim:
                continue
            if current_dim > self.target_dim:
                raise ValueError(
                    f"StateActionZeroPadTransform: key '{key}' has last-dim {current_dim} "
                    f"which exceeds target_dim={self.target_dim}."
                )

            pad_size = self.target_dim - current_dim
            # Build pad tuple for F.pad: last dimension padding (left=0, right=pad_size)
            pad_tuple = (0, pad_size)
            import torch.nn.functional as F

            data[key] = F.pad(tensor, pad_tuple, mode="constant", value=0.0)

        return data


class SingleArmToDualArmTransform(ModalityTransform):
    """Maps a single-arm EEF tensor to the shared 14-D dual-arm layout.

    The target 14-D layout matches the shared action space used across all datasets::

        [left_pos(3), left_euler(3), left_gripper(1),
         right_pos(3), right_euler(3), right_gripper(1)]

    Two single-arm input widths are supported:

    - ``last_dim == 6``  — ``[pos3, euler3]``.  Both gripper slots (6, 13) are
      filled with ``gripper_pad_value`` because no gripper signal is present
      (typical for OXE_AugE which derives action by EEF-pose differencing).
    - ``last_dim == 7``  — ``[pos3, euler3, grip1]``.  The real gripper value is
      written to slot 13 (right gripper); slot 6 (left gripper) is filled with
      ``gripper_pad_value`` (typical for OXE original Bridge / RT-1 / etc.).

    Tensors that are already 14-D are passed through unchanged.

    Example::

        t = SingleArmToDualArmTransform(apply_to=["action"])
        # input  action: [T, 7]  -> output action: [T, 14]
    """

    apply_to: list[str] = Field(
        default_factory=lambda: ["action", "state"],
        description="Top-level modality keys (or prefixes) to transform.",
    )
    gripper_pad_value: float = Field(
        default=-1.0,
        description="Value used to fill gripper slots for arms without a gripper signal.",
    )

    def apply(self, data: dict[str, Any]) -> dict[str, Any]:
        for key in list(data.keys()):
            if not any(key == p or key.startswith(p + ".") for p in self.apply_to):
                continue

            tensor = data[key]
            if not isinstance(tensor, torch.Tensor):
                continue

            last_dim = tensor.shape[-1]
            if last_dim == 14:
                # Already dual-arm; emit an all-ones mask for `action` so loss
                # downstream is uniform with single-arm-padded samples.
                if key == "action":
                    data["action_mask"] = torch.ones_like(tensor, dtype=tensor.dtype)
                continue

            if last_dim not in (6, 7):
                raise ValueError(
                    f"SingleArmToDualArmTransform: key '{key}' has last-dim "
                    f"{last_dim}; expected 6 (pos3+euler3), 7 (pos3+euler3+grip1), "
                    f"or 14 (already dual-arm)."
                )

            batch_shape = tensor.shape[:-1]
            out = torch.zeros(*batch_shape, 14, dtype=tensor.dtype, device=tensor.device)
            # Left-arm gripper sentinel (slot 6)
            out[..., 6] = self.gripper_pad_value
            # Right-arm: pos(3) + euler(3)  →  slots 7-12
            out[..., 7:13] = tensor[..., 0:6]
            if last_dim == 7:
                # Real gripper value at slot 13
                out[..., 13] = tensor[..., 6]
            else:
                # No gripper signal — fill with sentinel
                out[..., 13] = self.gripper_pad_value
            data[key] = out

            # ── Loss mask ─────────────────────────────────────────────────
            # Slots populated with real signal get mask=1; padded zeros and
            # gripper sentinels get mask=0 so the framework's loss skips them.
            #   6-D input → only right pos+euler (slots 7-12) are real.
            #   7-D input → right pos+euler + right gripper (7-13) are real.
            # The mask is only emitted for the action key — state has no loss.
            if key == "action":
                mask = torch.zeros(*batch_shape, 14, dtype=out.dtype, device=out.device)
                mask[..., 7:13] = 1.0
                if last_dim == 7:
                    mask[..., 13] = 1.0
                data["action_mask"] = mask

        return data
