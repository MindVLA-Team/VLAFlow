"""OXE (Open-X Embodiment) benchmark — data config, embodiment tags, and mixtures."""

from vlaflow.dataloader.gr00t_lerobot.datasets import ModalityConfig
from vlaflow.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from vlaflow.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from vlaflow.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from vlaflow.dataloader.gr00t_lerobot.transform.pad import SingleArmToDualArmTransform, StateActionZeroPadTransform
from vlaflow.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from vlaflow.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoCrop,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)


# ---------------------------------------------------------------------------
# DataConfig — OXE Droid
# ---------------------------------------------------------------------------
class OxeDroidDataConfig:
    video_keys = [
        "video.exterior_image_1",
        "video.exterior_image_2",
        "video.wrist_image",
    ]
    state_keys = [
        "state.eef_position",
        "state.eef_rotation",
        "state.gripper_position",
    ]
    action_keys = [
        "action.eef_position_delta",
        "action.eef_rotation_delta",
        "action.gripper_position",
    ]
    language_keys = ["annotation.language.language_instruction"]
    observation_indices = [0]
    action_indices = list(range(16))

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                VideoToTensor(apply_to=self.video_keys),
                VideoCrop(apply_to=self.video_keys, scale=0.95),
                VideoResize(apply_to=self.video_keys, height=224, width=224, interpolation="linear"),
                VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                VideoToNumpy(apply_to=self.video_keys),
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={"state.eef_position": "min_max", "state.gripper_position": "min_max"},
                    target_rotations={"state.eef_rotation": "rotation_6d"},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={"action.gripper_position": "binary"},
                    target_rotations={"action.eef_rotation_delta": "axis_angle"},
                ),
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
            ]
        )


# ---------------------------------------------------------------------------
# DataConfig — OXE Bridge
# ---------------------------------------------------------------------------
class OxeBridgeDataConfig:
    video_keys = ["video.image_0"]
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

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={k: ("binary" if k == "state.gripper" else "q99") for k in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={k: ("binary" if k == "action.gripper" else "q99") for k in self.action_keys},
                ),
            ]
        )


# ---------------------------------------------------------------------------
# DataConfig — OXE RT-1
# ---------------------------------------------------------------------------
class OxeRT1DataConfig:
    video_keys = ["video.image"]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.rx",
        "state.ry",
        "state.rz",
        "state.rw",
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

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={k: ("binary" if k == "state.gripper" else "q99") for k in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={k: ("binary" if k == "action.gripper" else "q99") for k in self.action_keys},
                ),
            ]
        )


ROBOT_TYPE_CONFIG_MAP = {
    "oxe_droid": OxeDroidDataConfig(),
    "oxe_bridge": OxeBridgeDataConfig(),
    "oxe_rt1": OxeRT1DataConfig(),
}


# ---------------------------------------------------------------------------
# Embodiment Tags
# ---------------------------------------------------------------------------
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "oxe_droid": EmbodimentTag.OXE_DROID,
    "oxe_bridge": EmbodimentTag.OXE_BRIDGE,
    "oxe_rt1": EmbodimentTag.OXE_RT1,
}


# ---------------------------------------------------------------------------
# Mixtures
# ---------------------------------------------------------------------------
DATASET_NAMED_MIXTURES = {
    "bridge": [
        ("ipec_bridge_all", 1.0, "oxe_bridge"),
    ],
    "bridge_rt_1": [
        ("ipec_bridge_all", 1.0, "oxe_bridge"),
        ("fractal20220817_data", 1.0, "oxe_rt1"),
    ],
}


# ---------------------------------------------------------------------------
# 14-D dual-arm DataConfigs (for OXE_Mix pretrained checkpoint)
#
# These configs produce:
#   action : [T, 14]  — right-arm slots 7-13 filled, left-arm (0-6) zeroed
#   action_mask : [T, 14] — 1.0 for real right-arm dims, 0.0 for padded left arm
#   state  : [T, 14]  — native 8-D padded with 6 zeros
#
# Use with YAMLs that set action_dim=14, state_dim=14 to match the OXE_Mix
# pretrained checkpoint architecture.
# ---------------------------------------------------------------------------


class OxeBridge14DDataConfig:
    """OxeBridge with 7-D → 14-D dual-arm padding.

    Use with ``data_mix: bridge_14d`` or ``bridge_rt_1_14d`` and a YAML that
    sets ``action_dim: 14, state_dim: 14`` to load from an OXE_Mix pretrained
    checkpoint.
    """

    video_keys = ["video.image_0"]
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

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                # VideoToTensor(apply_to=self.video_keys),
                # VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                # VideoToNumpy(apply_to=self.video_keys),
                # ── State / Action: identical to OxeBridgeDataConfig ──────────
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={k: ("binary" if k == "state.gripper" else "q99") for k in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={k: ("binary" if k == "action.gripper" else "q99") for k in self.action_keys},
                ),
                # ── 14-D extension: concat then pad ───────────────────────────
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
                # action [T, 7] → [T, 14]; emits action_mask with 0 on left-arm slots
                SingleArmToDualArmTransform(apply_to=["action"]),
                # state  [T, 8] → [T, 14] (zero-pad last 6 dims)
                StateActionZeroPadTransform(apply_to=["state"], target_dim=14),
            ]
        )


class OxeRT114DDataConfig:
    """OxeRT1 with 7-D → 14-D dual-arm padding.

    Use with ``data_mix: bridge_rt_1_14d`` and a YAML that sets
    ``action_dim: 14, state_dim: 14``.
    """

    video_keys = ["video.image"]
    state_keys = [
        "state.x",
        "state.y",
        "state.z",
        "state.rx",
        "state.ry",
        "state.rz",
        "state.rw",
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

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                # VideoToTensor(apply_to=self.video_keys),
                # VideoColorJitter(apply_to=self.video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
                # VideoToNumpy(apply_to=self.video_keys),
                # ── State / Action: identical to OxeRT1DataConfig ─────────────
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={k: ("binary" if k == "state.gripper" else "q99") for k in self.state_keys},
                ),
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={k: ("binary" if k == "action.gripper" else "q99") for k in self.action_keys},
                ),
                # ── 14-D extension: concat then pad ───────────────────────────
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
                SingleArmToDualArmTransform(apply_to=["action"]),
                StateActionZeroPadTransform(apply_to=["state"], target_dim=14),
            ]
        )


ROBOT_TYPE_CONFIG_MAP.update(
    {
        "oxe_bridge_14d": OxeBridge14DDataConfig(),
        "oxe_rt1_14d": OxeRT114DDataConfig(),
    }
)

ROBOT_TYPE_TO_EMBODIMENT_TAG.update(
    {
        "oxe_bridge_14d": EmbodimentTag.OXE_BRIDGE,
        "oxe_rt1_14d": EmbodimentTag.OXE_RT1,
    }
)

DATASET_NAMED_MIXTURES.update(
    {
        # 14-D mixtures for OXE_Mix pretrained checkpoint fine-tuning
        "bridge_14d": [
            ("ipec_bridge_orig", 1.0, "oxe_bridge_14d"),
        ],
        "rt1_14d": [
            ("ipec_fractal20220817_data_lerobot", 1.0, "oxe_rt1_14d"),
        ],
        "bridge_rt_1_14d": [
            ("ipec_bridge_orig", 1.0, "oxe_bridge_14d"),
            ("ipec_fractal20220817_data_lerobot", 1.0, "oxe_rt1_14d"),
        ],
    }
)


# ===========================================================================
# World Model (MindWPI) 14-D DataConfigs
# ===========================================================================
# WM variants inherit from 14-D DataConfigs and add:
#   - compute_action_loss: bool (all True — SimplerEnv actions are reliable)
#   - latent_view_key: str | None (which video view for latent extraction)
#
# Use with lerobot_wm_datasets and MindWPI framework for downstream fine-tuning
# from OXE_Mix WM pretrained checkpoints.


class OxeBridge14DWMDataConfig(OxeBridge14DDataConfig):
    """WM variant of OxeBridge14D for MindWPI fine-tuning."""

    compute_action_loss = True
    latent_view_key = "video.image_0"


class OxeRT114DWMDataConfig(OxeRT114DDataConfig):
    """WM variant of OxeRT114D for MindWPI fine-tuning."""

    compute_action_loss = True
    latent_view_key = "video.image"


ROBOT_TYPE_CONFIG_MAP.update(
    {
        "oxe_bridge_14d_wm": OxeBridge14DWMDataConfig(),
        "oxe_rt1_14d_wm": OxeRT114DWMDataConfig(),
    }
)

ROBOT_TYPE_TO_EMBODIMENT_TAG.update(
    {
        "oxe_bridge_14d_wm": EmbodimentTag.OXE_BRIDGE,
        "oxe_rt1_14d_wm": EmbodimentTag.OXE_RT1,
    }
)

DATASET_NAMED_MIXTURES.update(
    {
        # 14-D WM mixtures for MindWPI fine-tuning from OXE_Mix WM pretrained checkpoint
        "bridge_14d_wm": [
            ("ipec_bridge_orig", 1.0, "oxe_bridge_14d_wm"),
        ],
        "rt1_14d_wm": [
            ("ipec_fractal20220817_data_lerobot", 1.0, "oxe_rt1_14d_wm"),
        ],
        "bridge_rt_1_14d_wm": [
            ("ipec_bridge_orig", 1.0, "oxe_bridge_14d_wm"),
            ("ipec_fractal20220817_data_lerobot", 1.0, "oxe_rt1_14d_wm"),
        ],
    }
)
