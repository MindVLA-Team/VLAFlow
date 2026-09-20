"""
OXE_AugE dataset configuration for vlaflow.

Each OXE_AugE dataset contains synthetic EEF-pose observations for 8 robot embodiments:
  widowX, sawyer, ur5e, google_robot, jaco, kinova3, kuka_iiwa, xarm7

ee_pose layout (7-dim per robot in the parquet):
  [0:3]  EEF position (x, y, z)
  [3:7]  EEF orientation — quaternion in wxyz order

There is no standalone action column; actions are derived from delta of consecutive
ee_pose frames (action_mode="delta").  At load time, the quaternion is automatically
converted to 3-D Euler XYZ (see RotationType.QUATERNION handling in datasets.py).

Final action representation: 6-D [pos3, euler3] mapped to the shared 14-D dual-arm
action space by SingleArmToDualArmTransform:
  [left_zeros(3), left_zeros(3), left_grip_pad(-1),
   right_pos(3),  right_euler(3), right_grip_pad(-1)]

One DataConfig class per robot type.  The comprehensive modality.json written by
scripts/gen_oxeauge_modality.py covers all 8 robots; each DataConfig selects its
robot's keys.

Data path:
    data/OXE_AugE
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
import os
import os as _os

from vlaflow.dataloader.gr00t_lerobot.datasets import ModalityConfig
from vlaflow.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from vlaflow.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from vlaflow.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from vlaflow.dataloader.gr00t_lerobot.transform.pad import SingleArmToDualArmTransform
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

OXE_AUGE_DATA_ROOT = os.path.join(os.environ.get("DATA_ROOT", "data"), "OXE_AugE_v2")

OXE_AUGE_OBS_INDICES = [0]
OXE_AUGE_ACT_INDICES = list(range(16))

# All 8 robot names used in column keys
OXE_AUGE_ROBOTS = [
    "widowX",
    "sawyer",
    "ur5e",
    "google_robot",
    "jaco",
    "kinova3",
    "kuka_iiwa",
    "xarm7",
]

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_modality_config(robot: str):
    """Build modality config selecting only the given robot's keys."""
    video_keys = [f"video.{robot}.image"]
    state_keys = [f"state.{robot}.ee_pos", f"state.{robot}.ee_rot"]
    action_keys = [f"action.{robot}.ee_pos", f"action.{robot}.ee_rot"]
    language_keys = ["annotation.language.natural_language_instruction"]
    return {
        "video": ModalityConfig(delta_indices=OXE_AUGE_OBS_INDICES, modality_keys=video_keys),
        "state": ModalityConfig(delta_indices=OXE_AUGE_OBS_INDICES, modality_keys=state_keys),
        "action": ModalityConfig(delta_indices=OXE_AUGE_ACT_INDICES, modality_keys=action_keys),
        "language": ModalityConfig(delta_indices=OXE_AUGE_OBS_INDICES, modality_keys=language_keys),
    }


def _make_data_cfg(robot: str) -> dict:
    """Build data_cfg dict for the given robot.

    lerobot_version is intentionally omitted — LeRobotSingleDataset auto-detects
    it from meta/info.json (OXE_AugE datasets can be v2.1 or v3.0).

    video_backend is "torchvision_av": some OXE_AugE datasets (e.g.
    language_table_*) use AV1-encoded mp4s that decord cannot decode.
    """
    return {
        "video_backend": "torchvision_av",
        "action_mode": "delta",
        "action_mode_apply_keys": [
            f"action.{robot}.ee_pos",
            f"action.{robot}.ee_rot",
        ],
        "action_mode_state_map": {
            f"action.{robot}.ee_pos": f"state.{robot}.ee_pos",
            f"action.{robot}.ee_rot": f"state.{robot}.ee_rot",
        },
    }


def _make_transform(robot: str):
    """Build composed transform for the given robot.

    Pipeline:
      1. Video: resize + colour-jitter
      2. State: tensor + normalise (pos=min_max, euler_rot=min_max with [-π,π] stats)
      3. Action: tensor + normalise (delta pos/euler, same normalization)
      4. ConcatTransform → 6-D action (pos3+euler3)
      5. SingleArmToDualArmTransform → 14-D shared dual-arm action space
    """
    video_keys = [f"video.{robot}.image"]
    state_keys = [f"state.{robot}.ee_pos", f"state.{robot}.ee_rot"]
    action_keys = [f"action.{robot}.ee_pos", f"action.{robot}.ee_rot"]

    return ComposedModalityTransform(
        transforms=[
            # video
            VideoToTensor(apply_to=video_keys),
            VideoResize(apply_to=video_keys, height=224, width=224, interpolation="linear"),
            VideoColorJitter(apply_to=video_keys, brightness=0.3, contrast=0.4, saturation=0.5, hue=0.08),
            VideoToNumpy(apply_to=video_keys),
            # state
            StateActionToTensor(apply_to=state_keys),
            StateActionTransform(
                apply_to=state_keys,
                normalization_modes={
                    f"state.{robot}.ee_pos": "min_max",
                    f"state.{robot}.ee_rot": "min_max",  # synthetic [-π,π] stats after quat→euler
                },
            ),
            # action
            StateActionToTensor(apply_to=action_keys),
            StateActionTransform(
                apply_to=action_keys,
                normalization_modes={
                    f"action.{robot}.ee_pos": "min_max",
                    f"action.{robot}.ee_rot": "min_max",  # synthetic [-π,π] stats after quat→euler
                },
            ),
            # concat → 6-D (pos3+euler3)
            ConcatTransform(
                video_concat_order=video_keys,
                state_concat_order=state_keys,
                action_concat_order=action_keys,
            ),
            # map single-arm 6-D to shared 14-D dual-arm action space
            SingleArmToDualArmTransform(apply_to=["action", "state"]),
        ]
    )


# ---------------------------------------------------------------------------
# DataConfig classes — one per robot
# ---------------------------------------------------------------------------


class OxeAugeWidowXDataConfig:
    robot = "widowX"
    embodiment_tag = EmbodimentTag.OXE_AUGE_WIDOWX
    data_cfg = _make_data_cfg(robot)

    def modality_config(self):
        return _make_modality_config(self.robot)

    def transform(self):
        return _make_transform(self.robot)


class OxeAugeSawyerDataConfig:
    robot = "sawyer"
    embodiment_tag = EmbodimentTag.OXE_AUGE_SAWYER
    data_cfg = _make_data_cfg(robot)

    def modality_config(self):
        return _make_modality_config(self.robot)

    def transform(self):
        return _make_transform(self.robot)


class OxeAugeUR5EDataConfig:
    robot = "ur5e"
    embodiment_tag = EmbodimentTag.OXE_AUGE_UR5E
    data_cfg = _make_data_cfg(robot)

    def modality_config(self):
        return _make_modality_config(self.robot)

    def transform(self):
        return _make_transform(self.robot)


class OxeAugeGoogleRobotDataConfig:
    robot = "google_robot"
    embodiment_tag = EmbodimentTag.OXE_AUGE_GOOGLE
    data_cfg = _make_data_cfg(robot)

    def modality_config(self):
        return _make_modality_config(self.robot)

    def transform(self):
        return _make_transform(self.robot)


class OxeAugeJacoDataConfig:
    robot = "jaco"
    embodiment_tag = EmbodimentTag.OXE_AUGE_JACO
    data_cfg = _make_data_cfg(robot)

    def modality_config(self):
        return _make_modality_config(self.robot)

    def transform(self):
        return _make_transform(self.robot)


class OxeAugeKinova3DataConfig:
    robot = "kinova3"
    embodiment_tag = EmbodimentTag.OXE_AUGE_KINOVA3
    data_cfg = _make_data_cfg(robot)

    def modality_config(self):
        return _make_modality_config(self.robot)

    def transform(self):
        return _make_transform(self.robot)


class OxeAugeKukaIIWADataConfig:
    robot = "kuka_iiwa"
    embodiment_tag = EmbodimentTag.OXE_AUGE_KUKA
    data_cfg = _make_data_cfg(robot)

    def modality_config(self):
        return _make_modality_config(self.robot)

    def transform(self):
        return _make_transform(self.robot)


class OxeAugeXArm7DataConfig:
    robot = "xarm7"
    embodiment_tag = EmbodimentTag.OXE_AUGE_XARM7
    data_cfg = _make_data_cfg(robot)

    def modality_config(self):
        return _make_modality_config(self.robot)

    def transform(self):
        return _make_transform(self.robot)


# ---------------------------------------------------------------------------
# Robot-type registry
# ---------------------------------------------------------------------------

ROBOT_TYPE_CONFIG_MAP = {
    "oxe_auge_widowx": OxeAugeWidowXDataConfig(),
    "oxe_auge_sawyer": OxeAugeSawyerDataConfig(),
    "oxe_auge_ur5e": OxeAugeUR5EDataConfig(),
    "oxe_auge_google_robot": OxeAugeGoogleRobotDataConfig(),
    "oxe_auge_jaco": OxeAugeJacoDataConfig(),
    "oxe_auge_kinova3": OxeAugeKinova3DataConfig(),
    "oxe_auge_kuka_iiwa": OxeAugeKukaIIWADataConfig(),
    "oxe_auge_xarm7": OxeAugeXArm7DataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "oxe_auge_widowx": EmbodimentTag.OXE_AUGE_WIDOWX,
    "oxe_auge_sawyer": EmbodimentTag.OXE_AUGE_SAWYER,
    "oxe_auge_ur5e": EmbodimentTag.OXE_AUGE_UR5E,
    "oxe_auge_google_robot": EmbodimentTag.OXE_AUGE_GOOGLE,
    "oxe_auge_jaco": EmbodimentTag.OXE_AUGE_JACO,
    "oxe_auge_kinova3": EmbodimentTag.OXE_AUGE_KINOVA3,
    "oxe_auge_kuka_iiwa": EmbodimentTag.OXE_AUGE_KUKA,
    "oxe_auge_xarm7": EmbodimentTag.OXE_AUGE_XARM7,
}

# Robot name (column-key stem) → robot-type string
_ROBOT_STEM_TO_TYPE = {
    "widowX": "oxe_auge_widowx",
    "sawyer": "oxe_auge_sawyer",
    "ur5e": "oxe_auge_ur5e",
    "google_robot": "oxe_auge_google_robot",
    "jaco": "oxe_auge_jaco",
    "kinova3": "oxe_auge_kinova3",
    "kuka_iiwa": "oxe_auge_kuka_iiwa",
    "xarm7": "oxe_auge_xarm7",
}

# ---------------------------------------------------------------------------
# Dataset discovery — only _augmented directories
# ---------------------------------------------------------------------------
#
# Excluded prefixes:
#   - "language_table_": Google Language Table is a 2-D planar push task.
#     Its end-effector z is essentially constant (range ~4e-5 m), which makes
#     min_max normalization degenerate, and quat slots are trivially fixed.
#     Mixing it into a 6-DOF EE-pose action space adds noise without learning
#     signal, so we skip it at discovery time.
_EXCLUDED_PREFIXES = ("language_table_",)

_AUGMENTED_DATASETS = sorted(
    [
        d
        for d in (_os.listdir(OXE_AUGE_DATA_ROOT) if _os.path.isdir(OXE_AUGE_DATA_ROOT) else [])
        if d.endswith("_augmented")
        and not any(d.startswith(p) for p in _EXCLUDED_PREFIXES)
        and _os.path.isdir(_os.path.join(OXE_AUGE_DATA_ROOT, d))
    ]
)

# ---------------------------------------------------------------------------
# Named mixtures
# ---------------------------------------------------------------------------

# Each augmented dataset is used once per robot (8 entries per dataset).
# Per-robot sub-mixtures use every dataset with that robot's DataConfig.
DATASET_NAMED_MIXTURES: dict = {}

for _robot_stem, _robot_type in _ROBOT_STEM_TO_TYPE.items():
    DATASET_NAMED_MIXTURES[f"oxeauge_{_robot_stem.lower()}"] = [(d, 1.0, _robot_type) for d in _AUGMENTED_DATASETS]

# Full mixture: all datasets × all 8 robots
DATASET_NAMED_MIXTURES["oxeauge_all"] = [
    (d, 1.0, _robot_type) for d in _AUGMENTED_DATASETS for _robot_type in _ROBOT_STEM_TO_TYPE.values()
]


# ===========================================================================
# World Model (MindWPI) Configs
# ===========================================================================
# WM variants inherit from base DataConfig classes, adding:
#   - compute_action_loss: bool (whether actions are reliable)
#   - latent_view_key: str | None (which video key for latent extraction)


class OxeAugeWidowXWMDataConfig(OxeAugeWidowXDataConfig):
    """WM variant: same as base + WM metadata."""

    compute_action_loss = True
    latent_view_key = "video.widowX.image"


class OxeAugeSawyerWMDataConfig(OxeAugeSawyerDataConfig):
    compute_action_loss = True
    latent_view_key = "video.sawyer.image"


class OxeAugeUR5EWMDataConfig(OxeAugeUR5EDataConfig):
    compute_action_loss = True
    latent_view_key = "video.ur5e.image"


class OxeAugeGoogleRobotWMDataConfig(OxeAugeGoogleRobotDataConfig):
    compute_action_loss = True
    latent_view_key = "video.google_robot.image"


class OxeAugeJacoWMDataConfig(OxeAugeJacoDataConfig):
    compute_action_loss = True
    latent_view_key = "video.jaco.image"


class OxeAugeKinova3WMDataConfig(OxeAugeKinova3DataConfig):
    compute_action_loss = True
    latent_view_key = "video.kinova3.image"


class OxeAugeKukaIIWAWMDataConfig(OxeAugeKukaIIWADataConfig):
    compute_action_loss = True
    latent_view_key = "video.kuka_iiwa.image"


class OxeAugeXArm7WMDataConfig(OxeAugeXArm7DataConfig):
    compute_action_loss = True
    latent_view_key = "video.xarm7.image"


# WM robot-type registry
ROBOT_TYPE_CONFIG_MAP["oxe_auge_widowx_wm"] = OxeAugeWidowXWMDataConfig()
ROBOT_TYPE_CONFIG_MAP["oxe_auge_sawyer_wm"] = OxeAugeSawyerWMDataConfig()
ROBOT_TYPE_CONFIG_MAP["oxe_auge_ur5e_wm"] = OxeAugeUR5EWMDataConfig()
ROBOT_TYPE_CONFIG_MAP["oxe_auge_google_robot_wm"] = OxeAugeGoogleRobotWMDataConfig()
ROBOT_TYPE_CONFIG_MAP["oxe_auge_jaco_wm"] = OxeAugeJacoWMDataConfig()
ROBOT_TYPE_CONFIG_MAP["oxe_auge_kinova3_wm"] = OxeAugeKinova3WMDataConfig()
ROBOT_TYPE_CONFIG_MAP["oxe_auge_kuka_iiwa_wm"] = OxeAugeKukaIIWAWMDataConfig()
ROBOT_TYPE_CONFIG_MAP["oxe_auge_xarm7_wm"] = OxeAugeXArm7WMDataConfig()

ROBOT_TYPE_TO_EMBODIMENT_TAG["oxe_auge_widowx_wm"] = EmbodimentTag.OXE_AUGE_WIDOWX
ROBOT_TYPE_TO_EMBODIMENT_TAG["oxe_auge_sawyer_wm"] = EmbodimentTag.OXE_AUGE_SAWYER
ROBOT_TYPE_TO_EMBODIMENT_TAG["oxe_auge_ur5e_wm"] = EmbodimentTag.OXE_AUGE_UR5E
ROBOT_TYPE_TO_EMBODIMENT_TAG["oxe_auge_google_robot_wm"] = EmbodimentTag.OXE_AUGE_GOOGLE
ROBOT_TYPE_TO_EMBODIMENT_TAG["oxe_auge_jaco_wm"] = EmbodimentTag.OXE_AUGE_JACO
ROBOT_TYPE_TO_EMBODIMENT_TAG["oxe_auge_kinova3_wm"] = EmbodimentTag.OXE_AUGE_KINOVA3
ROBOT_TYPE_TO_EMBODIMENT_TAG["oxe_auge_kuka_iiwa_wm"] = EmbodimentTag.OXE_AUGE_KUKA
ROBOT_TYPE_TO_EMBODIMENT_TAG["oxe_auge_xarm7_wm"] = EmbodimentTag.OXE_AUGE_XARM7

# WM robot-type stem mapping
_ROBOT_STEM_TO_WM_TYPE = {robot_stem: f"{robot_type}_wm" for robot_stem, robot_type in _ROBOT_STEM_TO_TYPE.items()}

# Per-robot WM sub-mixtures
for _robot_stem, _wm_robot_type in _ROBOT_STEM_TO_WM_TYPE.items():
    DATASET_NAMED_MIXTURES[f"oxeauge_{_robot_stem.lower()}_wm"] = [(d, 1.0, _wm_robot_type) for d in _AUGMENTED_DATASETS]

# Full WM mixture: all datasets × all 8 robots (WM variants)
DATASET_NAMED_MIXTURES["oxeauge_all_wm"] = [
    (d, 1.0, _wm_robot_type) for d in _AUGMENTED_DATASETS for _wm_robot_type in _ROBOT_STEM_TO_WM_TYPE.values()
]
