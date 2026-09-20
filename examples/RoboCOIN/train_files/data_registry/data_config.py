"""
RoboCOIN_SUB dataset configuration for vlaflow.

Three robot types are covered:

  AgilexCobotMagicDataConfig  — 80 datasets, dual-arm Agilex Cobot Magic
      • state/action: EEF pos+euler+gripper sliced from 26-D observation.state / action
      • 14-D output: left[pos3+euler3+gripper1] + right[pos3+euler3+gripper1]
      • videos: cam_head_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb

  GalaxeaR1LiteDataConfig     — 40 datasets, dual-arm Galaxea R1 Lite
      • state: eef_sim_pose_state (12-D) + grippers from observation.state
      • action: eef_sim_pose_action (12-D) + grippers from action column
      • 14-D output: same layout as above
      • videos: cam_head_left_rgb, cam_head_right_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb

  AgilexSplitAlohaDataConfig  — 1 dataset, same schema as Agilex_Cobot_Magic

Data path:
    data/RoboCOIN_SUB
"""

from vlaflow.dataloader.gr00t_lerobot.datasets import ModalityConfig
from vlaflow.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from vlaflow.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from vlaflow.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from vlaflow.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)
from vlaflow.dataloader.gr00t_lerobot.transform.video import (
    VideoColorJitter,
    VideoResize,
    VideoToNumpy,
    VideoToTensor,
)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

ROBOCOIN_OBS_INDICES = [0]
ROBOCOIN_ACT_INDICES = list(range(16))
ROBOCOIN_LANGUAGE_KEYS = ["annotation.language.task"]

import os

ROBOCOIN_DATA_ROOT = os.path.join(os.environ.get("DATA_ROOT", "data"), "RoboCOIN_SUB")


def _make_modality_config(video_keys, state_keys, action_keys):
    return {
        "video": ModalityConfig(delta_indices=ROBOCOIN_OBS_INDICES, modality_keys=video_keys),
        "state": ModalityConfig(delta_indices=ROBOCOIN_OBS_INDICES, modality_keys=state_keys),
        "action": ModalityConfig(delta_indices=ROBOCOIN_ACT_INDICES, modality_keys=action_keys),
        "language": ModalityConfig(delta_indices=ROBOCOIN_OBS_INDICES, modality_keys=ROBOCOIN_LANGUAGE_KEYS),
    }


def _video_transforms(video_keys):
    """Per-key tensorise+resize, then shared color-jitter+to-numpy.

    The base ``VideoTransform.apply`` ``np.concatenate``-stacks all keys in
    ``apply_to`` along the frame axis before applying its op, which requires
    every camera to share H/W. RoboCOIN Galaxea has cameras at different native
    resolutions (head 720p, wrist 360p), so each key must be resized to the
    common 224x224 *before* any multi-key step. Once all keys are 224x224,
    multi-key transforms (color jitter, to-numpy) work fine.
    """
    transforms: list = []
    for k in video_keys:
        transforms.append(VideoToTensor(apply_to=[k]))
        transforms.append(VideoResize(apply_to=[k], height=224, width=224, interpolation="linear"))
    transforms.append(VideoColorJitter(apply_to=video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08))
    transforms.append(VideoToNumpy(apply_to=video_keys))
    return transforms


# ---------------------------------------------------------------------------
# DataConfig — Agilex_Cobot_Magic  (also used for Agilex_Split_Aloha)
# ---------------------------------------------------------------------------


class AgilexCobotMagicDataConfig:
    """
    Dual-arm Agilex Cobot Magic robot.

    14-D action: left[pos3+euler3+gripper1] + right[pos3+euler3+gripper1]
    """

    video_keys = [
        "video.cam_head_rgb",
        "video.cam_left_wrist_rgb",
        "video.cam_right_wrist_rgb",
    ]
    state_keys = [
        "state.left.ee_pos",
        "state.left.ee_rot",
        "state.left.gripper",
        "state.right.ee_pos",
        "state.right.ee_rot",
        "state.right.gripper",
    ]
    action_keys = [
        "action.left.ee_pos",
        "action.left.ee_rot",
        "action.left.gripper",
        "action.right.ee_pos",
        "action.right.ee_rot",
        "action.right.gripper",
    ]

    data_cfg = {
        "lerobot_version": "v2.0",
        "action_mode": "delta",
        "action_mode_apply_keys": [
            "action.left.ee_pos",
            "action.left.ee_rot",
            "action.left.gripper",
            "action.right.ee_pos",
            "action.right.ee_rot",
            "action.right.gripper",
        ],
        "action_mode_state_map": {
            "action.left.ee_pos": "state.left.ee_pos",
            "action.left.ee_rot": "state.left.ee_rot",
            "action.left.gripper": "state.left.gripper",
            "action.right.ee_pos": "state.right.ee_pos",
            "action.right.ee_rot": "state.right.ee_rot",
            "action.right.gripper": "state.right.gripper",
        },
        "video_backend": "torchvision_av",
    }

    embodiment_tag = EmbodimentTag.AGILEX_COBOT_MAGIC

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys, self.action_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                *_video_transforms(self.video_keys),
                # state transforms
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        "state.left.ee_pos": "min_max",
                        "state.left.ee_rot": "min_max",
                        "state.left.gripper": "binary",
                        "state.right.ee_pos": "min_max",
                        "state.right.ee_rot": "min_max",
                        "state.right.gripper": "binary",
                    },
                ),
                # action transforms
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.left.ee_pos": "min_max",
                        "action.left.ee_rot": "min_max",
                        "action.left.gripper": "binary",
                        "action.right.ee_pos": "min_max",
                        "action.right.ee_rot": "min_max",
                        "action.right.gripper": "binary",
                    },
                ),
                # concat: produces 14-D action
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
            ]
        )


# ---------------------------------------------------------------------------
# DataConfig — Galaxea_R1_Lite
# ---------------------------------------------------------------------------


class GalaxeaR1LiteDataConfig:
    """
    Dual-arm Galaxea R1 Lite robot.

    State: eef_sim_pose_state (EEF pos+euler for both arms) + grippers from observation.state.
    Action: eef_sim_pose_action (delta EEF) + grippers from action column.
    14-D action: left[pos3+euler3+gripper1] + right[pos3+euler3+gripper1]
    """

    video_keys = [
        "video.cam_head_left_rgb",
        "video.cam_head_right_rgb",
        "video.cam_left_wrist_rgb",
        "video.cam_right_wrist_rgb",
    ]
    state_keys = [
        "state.left.ee_pos",
        "state.left.ee_rot",
        "state.left.gripper",
        "state.right.ee_pos",
        "state.right.ee_rot",
        "state.right.gripper",
    ]
    action_keys = [
        "action.left.ee_pos",
        "action.left.ee_rot",
        "action.left.gripper",
        "action.right.ee_pos",
        "action.right.ee_rot",
        "action.right.gripper",
    ]

    data_cfg = {
        "lerobot_version": "v2.0",
        "action_mode": "delta",
        "action_mode_apply_keys": [
            "action.left.ee_pos",
            "action.left.ee_rot",
            "action.left.gripper",
            "action.right.ee_pos",
            "action.right.ee_rot",
            "action.right.gripper",
        ],
        "action_mode_state_map": {
            "action.left.ee_pos": "state.left.ee_pos",
            "action.left.ee_rot": "state.left.ee_rot",
            "action.left.gripper": "state.left.gripper",
            "action.right.ee_pos": "state.right.ee_pos",
            "action.right.ee_rot": "state.right.ee_rot",
            "action.right.gripper": "state.right.gripper",
        },
        "video_backend": "torchvision_av",
    }

    embodiment_tag = EmbodimentTag.GALAXEA_R1_LITE

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys, self.action_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=[
                *_video_transforms(self.video_keys),
                # state transforms
                StateActionToTensor(apply_to=self.state_keys),
                StateActionTransform(
                    apply_to=self.state_keys,
                    normalization_modes={
                        "state.left.ee_pos": "min_max",
                        "state.left.ee_rot": "min_max",
                        "state.left.gripper": "binary",
                        "state.right.ee_pos": "min_max",
                        "state.right.ee_rot": "min_max",
                        "state.right.gripper": "binary",
                    },
                ),
                # action transforms
                StateActionToTensor(apply_to=self.action_keys),
                StateActionTransform(
                    apply_to=self.action_keys,
                    normalization_modes={
                        "action.left.ee_pos": "min_max",
                        "action.left.ee_rot": "min_max",
                        "action.left.gripper": "binary",
                        "action.right.ee_pos": "min_max",
                        "action.right.ee_rot": "min_max",
                        "action.right.gripper": "binary",
                    },
                ),
                # concat: produces 14-D action
                ConcatTransform(
                    video_concat_order=self.video_keys,
                    state_concat_order=self.state_keys,
                    action_concat_order=self.action_keys,
                ),
            ]
        )


# ---------------------------------------------------------------------------
# DataConfig — Agilex_Split_Aloha (same schema as Agilex_Cobot_Magic)
# ---------------------------------------------------------------------------


class AgilexSplitAlohaDataConfig(AgilexCobotMagicDataConfig):
    """Single Agilex Split Aloha dataset — same schema as Agilex_Cobot_Magic."""

    embodiment_tag = EmbodimentTag.AGILEX_SPLIT_ALOHA


# ---------------------------------------------------------------------------
# Robot-type registry
# ---------------------------------------------------------------------------

ROBOT_TYPE_CONFIG_MAP = {
    "agilex_cobot_magic": AgilexCobotMagicDataConfig(),
    "galaxea_r1_lite": GalaxeaR1LiteDataConfig(),
    "agilex_split_aloha": AgilexSplitAlohaDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "agilex_cobot_magic": EmbodimentTag.AGILEX_COBOT_MAGIC,
    "galaxea_r1_lite": EmbodimentTag.GALAXEA_R1_LITE,
    "agilex_split_aloha": EmbodimentTag.AGILEX_SPLIT_ALOHA,
}


# ---------------------------------------------------------------------------
# Dataset → robot-type assignment
# ---------------------------------------------------------------------------


def _auto_detect_robot_type(dataset_name: str) -> str:
    if dataset_name.startswith("Agilex_Cobot_Magic"):
        return "agilex_cobot_magic"
    if dataset_name.startswith("Galaxea_R1_Lite"):
        return "galaxea_r1_lite"
    if dataset_name.startswith("Agilex_Split_Aloha"):
        return "agilex_split_aloha"
    return "agilex_cobot_magic"  # fallback


# Enumerate all dataset names from the data root
import os as _os  # noqa: E402 (keep at bottom to avoid polluting namespace)

_AGILEX_COBOT_MAGIC_DATASETS = [
    d
    for d in (_os.listdir(ROBOCOIN_DATA_ROOT) if _os.path.isdir(ROBOCOIN_DATA_ROOT) else [])
    if (d.startswith("Agilex_Cobot_Magic") and "Agilex_Cobot_Magic_storage_towel" not in d)
]
_GALAXEA_R1_LITE_DATASETS = [
    d
    for d in (_os.listdir(ROBOCOIN_DATA_ROOT) if _os.path.isdir(ROBOCOIN_DATA_ROOT) else [])
    if d.startswith("Galaxea_R1_Lite")
]
_AGILEX_SPLIT_ALOHA_DATASETS = [
    d
    for d in (_os.listdir(ROBOCOIN_DATA_ROOT) if _os.path.isdir(ROBOCOIN_DATA_ROOT) else [])
    if d.startswith("Agilex_Split_Aloha")
]

# ---------------------------------------------------------------------------
# Named mixtures
# ---------------------------------------------------------------------------

DATASET_NAMED_MIXTURES: dict = {
    # All 121 RoboCOIN_SUB datasets
    "robocoin_all": (
        [(d, 1.0, "agilex_cobot_magic") for d in sorted(_AGILEX_COBOT_MAGIC_DATASETS)]
        + [(d, 1.0, "galaxea_r1_lite") for d in sorted(_GALAXEA_R1_LITE_DATASETS)]
        + [(d, 1.0, "agilex_split_aloha") for d in sorted(_AGILEX_SPLIT_ALOHA_DATASETS)]
    ),
    # Agilex_Cobot_Magic only (80 datasets)
    "robocoin_agilex": [(d, 1.0, "agilex_cobot_magic") for d in sorted(_AGILEX_COBOT_MAGIC_DATASETS)],
    # Galaxea_R1_Lite only (40 datasets)
    "robocoin_galaxea": [(d, 1.0, "galaxea_r1_lite") for d in sorted(_GALAXEA_R1_LITE_DATASETS)],
    # Agilex_Split_Aloha only (1 dataset)
    "robocoin_split_aloha": [(d, 1.0, "agilex_split_aloha") for d in sorted(_AGILEX_SPLIT_ALOHA_DATASETS)],
}


# ===========================================================================
# World Model (MindWPI) Configs
# ===========================================================================
# WM variants inherit from base DataConfig classes, adding WM-specific attributes.
# RoboCOIN datasets have reliable actions, so compute_action_loss = True for all.


class AgilexCobotMagicWMDataConfig(AgilexCobotMagicDataConfig):
    """WM variant for Agilex Cobot Magic."""

    compute_action_loss = True
    latent_view_key = "video.cam_head_rgb"


class GalaxeaR1LiteWMDataConfig(GalaxeaR1LiteDataConfig):
    """WM variant for Galaxea R1 Lite."""

    compute_action_loss = True
    latent_view_key = "video.cam_head_left_rgb"


class AgilexSplitAlohaWMDataConfig(AgilexSplitAlohaDataConfig):
    """WM variant for Agilex Split Aloha."""

    compute_action_loss = True
    latent_view_key = "video.cam_head_rgb"


# WM robot-type registry
ROBOT_TYPE_CONFIG_MAP["agilex_cobot_magic_wm"] = AgilexCobotMagicWMDataConfig()
ROBOT_TYPE_CONFIG_MAP["galaxea_r1_lite_wm"] = GalaxeaR1LiteWMDataConfig()
ROBOT_TYPE_CONFIG_MAP["agilex_split_aloha_wm"] = AgilexSplitAlohaWMDataConfig()

ROBOT_TYPE_TO_EMBODIMENT_TAG["agilex_cobot_magic_wm"] = EmbodimentTag.AGILEX_COBOT_MAGIC
ROBOT_TYPE_TO_EMBODIMENT_TAG["galaxea_r1_lite_wm"] = EmbodimentTag.GALAXEA_R1_LITE
ROBOT_TYPE_TO_EMBODIMENT_TAG["agilex_split_aloha_wm"] = EmbodimentTag.AGILEX_SPLIT_ALOHA

# WM named mixtures
DATASET_NAMED_MIXTURES["robocoin_all_wm"] = (
    [(d, 1.0, "agilex_cobot_magic_wm") for d in sorted(_AGILEX_COBOT_MAGIC_DATASETS)]
    + [(d, 1.0, "galaxea_r1_lite_wm") for d in sorted(_GALAXEA_R1_LITE_DATASETS)]
    + [(d, 1.0, "agilex_split_aloha_wm") for d in sorted(_AGILEX_SPLIT_ALOHA_DATASETS)]
)

DATASET_NAMED_MIXTURES["robocoin_agilex_wm"] = [
    (d, 1.0, "agilex_cobot_magic_wm") for d in sorted(_AGILEX_COBOT_MAGIC_DATASETS)
]

DATASET_NAMED_MIXTURES["robocoin_galaxea_wm"] = [
    (d, 1.0, "galaxea_r1_lite_wm") for d in sorted(_GALAXEA_R1_LITE_DATASETS)
]
