"""LIBERO benchmark — data config, embodiment tags, and mixtures."""

from vlaflow.dataloader.gr00t_lerobot.datasets import ModalityConfig
from vlaflow.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from vlaflow.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from vlaflow.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from vlaflow.dataloader.gr00t_lerobot.transform.pad import SingleArmToDualArmTransform, StateActionZeroPadTransform
from vlaflow.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from vlaflow.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


# ---------------------------------------------------------------------------
# DataConfig
# ---------------------------------------------------------------------------
class Libero4in1DataConfig:
    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.x": "min_max",
                        "action.y": "min_max",
                        "action.z": "min_max",
                        "action.roll": "min_max",
                        "action.pitch": "min_max",
                        "action.yaw": "min_max",
                    },
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka": Libero4in1DataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka": EmbodimentTag.FRANKA,
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "libero_all": [
        ("libero_object", 1.0, "libero_franka"),
        ("libero_goal", 1.0, "libero_franka"),
        ("libero_spatial", 1.0, "libero_franka"),
        ("libero_10", 1.0, "libero_franka"),
    ],
    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "multi_robot": [
        ("LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
}


# ---------------------------------------------------------------------------
# 14-D dual-arm DataConfig (for OXE_Mix pretrained checkpoint)
#
# Produces:
#   action      : [T, 14] — right-arm slots 7-13 filled, left-arm (0-6) zeroed
#   action_mask : [T, 14] — 1.0 on right-arm dims only; left-arm excluded from loss
#   state       : [T, 14] — native 8-D padded with 6 zeros
#
# Use with ``data_mix: libero_all_14d`` and a YAML that sets
# ``action_dim: 14, state_dim: 14`` to load from an OXE_Mix pretrained checkpoint.
# ---------------------------------------------------------------------------


class Libero14DDataConfig:
    """LIBERO Franka with 7-D → 14-D dual-arm padding.

    Extends the standard Libero4in1DataConfig with:
      - Video pipeline (resize to 224×224 + color augmentation)
      - ConcatTransform to merge per-key tensors into modality-level tensors
      - SingleArmToDualArmTransform: action [T,7] → [T,14] + action_mask
      - StateActionZeroPadTransform: state [T,8] → [T,14]
    """

    video_keys = [
        "video.primary_image",
        "video.wrist_image",
    ]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.roll",
        "state.pitch",
        "state.yaw",
        "state.pad",
        "state.gripper",
    ]
    action_keys = [
        "action.x",
        "action.y",
        "action.z",
        "action.roll",
        "action.pitch",
        "action.yaw",
        "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]
    observation_indices = [0]
    action_indices = list(range(16))
    state_indices = list(range(-16, 0))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                # ── Video ─────────────────────────────────────────────────────
                VideoToTensor(apply_to=self.video_keys),
                VideoCrop(apply_to=self.video_keys, scale=0.95),
                VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
                VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                VideoToNumpy(apply_to=self.video_keys),
                # ── State — convert to tensor (needed for ConcatTransform) ────
                StateActionToTensor(apply_to=self.state_keys),
                # ── Action ────────────────────────────────────────────────────
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.x": "min_max",
                        "action.y": "min_max",
                        "action.z": "min_max",
                        "action.roll": "min_max",
                        "action.pitch": "min_max",
                        "action.yaw": "min_max",
                    },
                ),
                # ── Concat then pad to 14-D ───────────────────────────────────
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
                # action [T, 7] → [T, 14]; emits action_mask (0 on left-arm slots 0-6)
                SingleArmToDualArmTransform(apply_to=["action"]),
                # state  [T, 8] → [T, 14] (zero-pad last 6 dims)
                StateActionZeroPadTransform(apply_to=["state"], target_dim=14),
            ]
        )


ROBOT_TYPE_CONFIG_MAP.update(
    {
        "libero_franka_14d": Libero14DDataConfig(),
    }
)

ROBOT_TYPE_TO_EMBODIMENT_TAG.update(
    {
        "libero_franka_14d": EmbodimentTag.FRANKA,
    }
)

DATASET_NAMED_MIXTURES.update(
    {
        # 14-D mixtures for OXE_Mix pretrained checkpoint fine-tuning
        "libero_all_14d": [
            ("libero_object", 1.0, "libero_franka_14d"),
            ("libero_goal", 1.0, "libero_franka_14d"),
            ("libero_spatial", 1.0, "libero_franka_14d"),
            ("libero_10", 1.0, "libero_franka_14d"),
        ],
        "libero_goal_14d": [
            ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_14d"),
        ],
    }
)


# ===========================================================================
# World Model (MindWPI) 14-D DataConfigs
# ===========================================================================
# WM variant inherits from Libero14DDataConfig and adds:
#   - compute_action_loss: bool (True — LIBERO actions are reliable)
#   - latent_view_key: str (primary camera for latent extraction)
#
# Use with lerobot_wm_datasets and MindWPI framework for downstream fine-tuning
# from OXE_Mix WM pretrained checkpoints.


class Libero14DWMDataConfig(Libero14DDataConfig):
    """WM variant of Libero14D for MindWPI fine-tuning."""

    compute_action_loss = True
    latent_view_key = "video.primary_image"


ROBOT_TYPE_CONFIG_MAP.update(
    {
        "libero_franka_14d_wm": Libero14DWMDataConfig(),
    }
)

ROBOT_TYPE_TO_EMBODIMENT_TAG.update(
    {
        "libero_franka_14d_wm": EmbodimentTag.FRANKA,
    }
)

DATASET_NAMED_MIXTURES.update(
    {
        # 14-D WM mixtures for MindWPI fine-tuning from OXE_Mix WM pretrained checkpoint
        "libero_all_14d_wm": [
            ("libero_object", 1.0, "libero_franka_14d_wm"),
            ("libero_goal", 1.0, "libero_franka_14d_wm"),
            ("libero_spatial", 1.0, "libero_franka_14d_wm"),
            ("libero_10", 1.0, "libero_franka_14d_wm"),
        ],
        "libero_goal_14d_wm": [
            ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_14d_wm"),
        ],
    }
)
