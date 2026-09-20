"""
OXE (Open-X Embodiment) dataset configuration for vlaflow.

Auto-discovered by vlaflow.dataloader.gr00t_lerobot.registry from
examples/OXE/train_files/data_registry/.

State layout varies across the 26 OXE datasets; six distinct formats exist:

  euler      (13): x y z roll pitch yaw pad gripper
  quat       ( 5): x y z rx ry rz rw gripper
  7j+pad     ( 3): motor_0..6 + pad
  7j+gripper ( 3): motor_0..6 + gripper
  6j+pad+g   ( 1): motor_0..5 + pad + gripper
  8j         ( 1): motor_0..7

All datasets share the same 7-D EEF action format:
  x y z roll pitch yaw gripper
"""

from vlaflow.dataloader.gr00t_lerobot.datasets import ModalityConfig
from vlaflow.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from vlaflow.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from vlaflow.dataloader.gr00t_lerobot.transform.concat import ConcatTransform
from vlaflow.dataloader.gr00t_lerobot.transform.pad import SingleArmToDualArmTransform
from vlaflow.dataloader.gr00t_lerobot.transform.state_action import (
    StateActionToTensor,
    StateActionTransform,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared action / language constants
# ─────────────────────────────────────────────────────────────────────────────

OXE_ACTION_KEYS = ["action.x", "action.y", "action.z", "action.roll", "action.pitch", "action.yaw", "action.gripper"]
OXE_LANGUAGE_KEYS = ["annotation.human.action.task_description"]
OXE_OBS_INDICES = [0]
OXE_ACT_INDICES = list(range(16))

# ── State key lists by type ──────────────────────────────────────────────────

STATE_KEYS_EULER = [
    "state.x",
    "state.y",
    "state.z",
    "state.roll",
    "state.pitch",
    "state.yaw",
    "state.pad",
    "state.gripper",
]
STATE_KEYS_QUAT = [
    "state.x",
    "state.y",
    "state.z",
    "state.rx",
    "state.ry",
    "state.rz",
    "state.rw",
    "state.gripper",
]
STATE_KEYS_7J_PAD = [
    "state.motor_0",
    "state.motor_1",
    "state.motor_2",
    "state.motor_3",
    "state.motor_4",
    "state.motor_5",
    "state.motor_6",
    "state.pad",
]
STATE_KEYS_7J_GRIPPER = [
    "state.motor_0",
    "state.motor_1",
    "state.motor_2",
    "state.motor_3",
    "state.motor_4",
    "state.motor_5",
    "state.motor_6",
    "state.gripper",
]
STATE_KEYS_6J_PAD_GRIPPER = [
    "state.motor_0",
    "state.motor_1",
    "state.motor_2",
    "state.motor_3",
    "state.motor_4",
    "state.motor_5",
    "state.pad",
    "state.gripper",
]
STATE_KEYS_8J = [
    "state.motor_0",
    "state.motor_1",
    "state.motor_2",
    "state.motor_3",
    "state.motor_4",
    "state.motor_5",
    "state.motor_6",
    "state.motor_7",
]

# ─────────────────────────────────────────────────────────────────────────────
# Transform helpers
# ─────────────────────────────────────────────────────────────────────────────


def _action_transforms():
    """Standard EEF action normalisation (shared by all OXE datasets)."""
    return [
        StateActionToTensor(apply_to=OXE_ACTION_KEYS),
        StateActionTransform(
            apply_to=OXE_ACTION_KEYS,
            normalization_modes={
                "action.x": "q99",
                "action.y": "q99",
                "action.z": "q99",
                "action.roll": "q99",
                "action.pitch": "q99",
                "action.yaw": "q99",
                "action.gripper": "binary",
            },
        ),
    ]


def _concat_dual_arm_tail(video_keys: list[str], state_keys: list[str]) -> list:
    """Common tail applied by every DataConfig.

    1. ``ConcatTransform`` merges per-key tensors into single ``video`` /
       ``state`` / ``action`` modality tensors (matching the post-Concat path
       expected by ``_pack_sample``).
    2. ``SingleArmToDualArmTransform`` maps the 7-D OXE action
       ``[x, y, z, roll, pitch, yaw, gripper]`` to the shared 14-D dual-arm
       layout (right-arm slots 7-12 + real gripper at slot 13). Required so OXE
       original can be mixed with OXE_AugE / RoboCOIN in ``oxe_mix_full``.

    Layout after this tail (action)::

        [0:6]   left arm   = 0
        [6]     left grip  = -1.0 (sentinel)
        [7:10]  right pos  = action.x/y/z   (q99-normalised)
        [10:13] right eul  = action.roll/pitch/yaw  (q99-normalised)
        [13]    right grip = action.gripper (binary 0 or 1)

    State is intentionally kept at its native width (varies across the 26 OXE
    datasets); the trainer's ``collate_fn`` returns a list of samples so
    heterogeneous state widths across embodiments are not a problem.
    """
    return [
        ConcatTransform(
            video_concat_order=video_keys,
            state_concat_order=state_keys,
            action_concat_order=OXE_ACTION_KEYS,
        ),
        SingleArmToDualArmTransform(apply_to=["action"]),
    ]


def _euler_state_transforms():
    return [
        StateActionToTensor(apply_to=STATE_KEYS_EULER),
        StateActionTransform(
            apply_to=STATE_KEYS_EULER,
            normalization_modes={
                "state.x": "q99",
                "state.y": "q99",
                "state.z": "q99",
                "state.roll": "q99",
                "state.pitch": "q99",
                "state.yaw": "q99",
                "state.pad": "q99",
                "state.gripper": "binary",
            },
        ),
    ]


def _quat_state_transforms():
    return [
        StateActionToTensor(apply_to=STATE_KEYS_QUAT),
        StateActionTransform(
            apply_to=STATE_KEYS_QUAT,
            normalization_modes={
                "state.x": "q99",
                "state.y": "q99",
                "state.z": "q99",
                "state.rx": "q99",
                "state.ry": "q99",
                "state.rz": "q99",
                "state.rw": "q99",
                "state.gripper": "binary",
            },
        ),
    ]


def _joint_transforms(state_keys, binary_keys=("state.gripper",), q99_keys=None):
    if q99_keys is None:
        q99_keys = [k for k in state_keys if k not in binary_keys and k != "state.pad"]
    modes = {k: "q99" for k in q99_keys}
    for k in binary_keys:
        if k in state_keys:
            modes[k] = "binary"
    for k in state_keys:
        if k not in modes:
            modes[k] = "q99"
    return [
        StateActionToTensor(apply_to=state_keys),
        StateActionTransform(apply_to=state_keys, normalization_modes=modes),
    ]


def _make_modality_config(video_keys, state_keys):
    return {
        "video": ModalityConfig(delta_indices=OXE_OBS_INDICES, modality_keys=video_keys),
        "state": ModalityConfig(delta_indices=OXE_OBS_INDICES, modality_keys=state_keys),
        "action": ModalityConfig(delta_indices=OXE_ACT_INDICES, modality_keys=OXE_ACTION_KEYS),
        "language": ModalityConfig(delta_indices=OXE_OBS_INDICES, modality_keys=OXE_LANGUAGE_KEYS),
    }


# ─────────────────────────────────────────────────────────────────────────────
# DataConfig classes
# ─────────────────────────────────────────────────────────────────────────────

# ── EEF Euler state  ─────────────────────────────────────────────────────────


class OxeEulerSingleCamDataConfig:
    """EEF euler state + single primary-view camera.
    Datasets: bc_z, cmu_stretch, dlr_edan, language_table (rgb→image).
    """

    video_keys = ["video.image"]
    state_keys = STATE_KEYS_EULER
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_euler_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


class OxeEulerWristOnlyDataConfig:
    """EEF euler state + wrist-only camera.  Dataset: dobbe."""

    video_keys = ["video.wrist_image"]
    state_keys = STATE_KEYS_EULER
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_euler_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


class OxeEulerDualCamDataConfig:
    """EEF euler state + image + wrist_image.
    Datasets: stanford_hydra, taco_play (rgb_static→image, rgb_gripper→wrist).
    """

    video_keys = ["video.image", "video.wrist_image"]
    state_keys = STATE_KEYS_EULER
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_euler_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


class OxeEulerJacoDataConfig:
    """EEF euler state + image + image_wrist.  Dataset: jaco_play."""

    video_keys = ["video.image", "video.image_wrist"]
    state_keys = STATE_KEYS_EULER
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_euler_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


class OxeEulerNyuFrankaDataConfig:
    """EEF euler state + image + image_additional_view.  Dataset: nyu_franka_play."""

    video_keys = ["video.image", "video.image_additional_view"]
    state_keys = STATE_KEYS_EULER
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_euler_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


class OxeBridgeEulerDataConfig:
    """EEF euler state + primary Bridge camera (``video.image_0``).

    Dataset: ``ipec_bridge_all`` (and other Bridge-style datasets whose
    LeRobot v2 ``features`` use ``observation.images.image_0`` as the main
    third-person view instead of the generic ``observation.images.image``).

    Replaces the legacy ``oxe_bridge`` robot_type (defined in
    ``vlaflow/dataloader/gr00t_lerobot/data_config.py``) which had its
    ConcatTransform and SingleArmToDualArmTransform commented out — hence
    produced a raw 7-D action incompatible with the 14-D dual-arm mixture.
    """

    video_keys = ["video.image_0"]
    state_keys = STATE_KEYS_EULER
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_euler_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


class OxeDroidLocalDataConfig:
    """EEF euler state + 3 droid cameras (_left suffix).  Dataset: droid."""

    video_keys = [
        "video.exterior_image_1_left",
        "video.exterior_image_2_left",
        "video.wrist_image_left",
    ]
    state_keys = STATE_KEYS_EULER
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_euler_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


class OxeFMBDataConfig:
    """EEF euler state + 4 cameras (2 side + 2 wrist).  Dataset: fmb."""

    video_keys = [
        "video.image_side_1",
        "video.image_side_2",
        "video.image_wrist_1",
        "video.image_wrist_2",
    ]
    state_keys = STATE_KEYS_EULER
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_euler_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


# ── EEF Quaternion state  ────────────────────────────────────────────────────


class OxeQuatSingleCamDataConfig:
    """EEF quaternion state + single primary-view camera.
    Datasets: fractal20220817_data, kuka.
    """

    video_keys = ["video.image"]
    state_keys = STATE_KEYS_QUAT
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_quat_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


class OxeQuatDualCamDataConfig:
    """EEF quaternion state + image + wrist_image.
    Datasets: austin_sailor, austin_sirius.
    """

    video_keys = ["video.image", "video.wrist_image"]
    state_keys = STATE_KEYS_QUAT
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_quat_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


class OxeQuatUR5DataConfig:
    """EEF quaternion state + image + hand_image.  Dataset: berkeley_autolab_ur5."""

    video_keys = ["video.image", "video.hand_image"]
    state_keys = STATE_KEYS_QUAT
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_quat_state_transforms()
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


# ── Joint-space: motor_0..6 + pad  ──────────────────────────────────────────


class OxeJoints7PadSingleCamDataConfig:
    """Joint-space state (7 joints + pad) + single camera.
    Datasets: toto, ucsd_kitchen.
    """

    video_keys = ["video.image"]
    state_keys = STATE_KEYS_7J_PAD
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_joint_transforms(STATE_KEYS_7J_PAD, binary_keys=())
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


class OxeJoints7PadCableRoutingDataConfig:
    """Joint-space state (7 joints + pad) + 4 angled cameras.
    Dataset: berkeley_cable_routing.
    """

    video_keys = [
        "video.image",
        "video.wrist45_image",
        "video.top_image",
        "video.wrist225_image",
    ]
    state_keys = STATE_KEYS_7J_PAD
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_joint_transforms(STATE_KEYS_7J_PAD, binary_keys=())
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


# ── Joint-space: motor_0..6 + gripper  ──────────────────────────────────────


class OxeJoints7GripperDualCamDataConfig:
    """Joint-space state (7 joints + gripper) + image + wrist_image.
    Datasets: austin_buds, utaustin_mutex,
              viola (agentview_rgb→image, eye_in_hand_rgb→wrist_image).
    """

    video_keys = ["video.image", "video.wrist_image"]
    state_keys = STATE_KEYS_7J_GRIPPER
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_joint_transforms(STATE_KEYS_7J_GRIPPER)
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


# ── Joint-space: motor_0..5 + pad + gripper  ────────────────────────────────


class OxeJoints6PadGripperDualCamDataConfig:
    """Joint-space state (6 joints + pad + gripper) + image + wrist_image.
    Dataset: berkeley_fanuc_manipulation.
    """

    video_keys = ["video.image", "video.wrist_image"]
    state_keys = STATE_KEYS_6J_PAD_GRIPPER
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_joint_transforms(STATE_KEYS_6J_PAD_GRIPPER)
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


# ── Joint-space: motor_0..7 (8 joints)  ─────────────────────────────────────


class OxeJoints8SingleCamDataConfig:
    """Joint-space state (8 unnamed joints) + single camera.
    Dataset: roboturk (front_rgb→image).
    """

    video_keys = ["video.image"]
    state_keys = STATE_KEYS_8J
    action_keys = OXE_ACTION_KEYS
    language_keys = OXE_LANGUAGE_KEYS
    observation_indices = OXE_OBS_INDICES
    action_indices = OXE_ACT_INDICES

    def modality_config(self):
        return _make_modality_config(self.video_keys, self.state_keys)

    def transform(self):
        return ComposedModalityTransform(
            transforms=_joint_transforms(STATE_KEYS_8J, binary_keys=())
            + _action_transforms()
            + _concat_dual_arm_tail(self.video_keys, self.state_keys)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Dataset → robot-type mapping
# ─────────────────────────────────────────────────────────────────────────────

DATASET_ROBOT_TYPE: dict[str, str] = {
    # ── EEF Euler ──────────────────────────────────────────────────────────
    "bc_z": "oxe_euler_single",
    "cmu_stretch": "oxe_euler_single",
    "dlr_edan_shared_control_converted_externally_to_rlds": "oxe_euler_single",
    "language_table": "oxe_euler_single",
    "dobbe": "oxe_euler_wrist",
    "stanford_hydra_dataset_converted_externally_to_rlds": "oxe_euler_dual",
    "taco_play": "oxe_euler_dual",
    "jaco_play": "oxe_euler_jaco",
    "nyu_franka_play_dataset_converted_externally_to_rlds": "oxe_euler_nyu_franka",
    # "bridge_dataset":                                        "oxe_bridge",    # existing type
    "ipec_bridge_all": "oxe_bridge_euler",  # new 14-D-compatible type
    "droid": "oxe_droid_local",
    "fmb": "oxe_fmb",
    # ── EEF Quaternion ─────────────────────────────────────────────────────
    "fractal20220817_data": "oxe_quat_single",
    "kuka": "oxe_quat_single",
    "austin_sailor_dataset_converted_externally_to_rlds": "oxe_quat_dual",
    "austin_sirius_dataset_converted_externally_to_rlds": "oxe_quat_dual",
    "berkeley_autolab_ur5": "oxe_quat_ur5",
    # ── Joint-space: 7 joints + pad ────────────────────────────────────────
    "toto": "oxe_joints7pad_single",
    "ucsd_kitchen_dataset_converted_externally_to_rlds": "oxe_joints7pad_single",
    "berkeley_cable_routing": "oxe_joints7pad_cable",
    # ── Joint-space: 7 joints + gripper ────────────────────────────────────
    "austin_buds_dataset_converted_externally_to_rlds": "oxe_joints7g_dual",
    "utaustin_mutex": "oxe_joints7g_dual",
    "viola": "oxe_joints7g_dual",
    # ── Joint-space: 6 joints + pad + gripper ──────────────────────────────
    "berkeley_fanuc_manipulation": "oxe_joints6pg_dual",
    # ── Joint-space: 8 joints ──────────────────────────────────────────────
    "roboturk": "oxe_joints8_single",
}

# ─────────────────────────────────────────────────────────────────────────────
# Registry exports  (auto-merged by registry.py on import)
# ─────────────────────────────────────────────────────────────────────────────

ROBOT_TYPE_CONFIG_MAP: dict = {
    # EEF Euler
    "oxe_euler_single": OxeEulerSingleCamDataConfig(),
    "oxe_euler_wrist": OxeEulerWristOnlyDataConfig(),
    "oxe_euler_dual": OxeEulerDualCamDataConfig(),
    "oxe_euler_jaco": OxeEulerJacoDataConfig(),
    "oxe_euler_nyu_franka": OxeEulerNyuFrankaDataConfig(),
    "oxe_droid_local": OxeDroidLocalDataConfig(),
    "oxe_fmb": OxeFMBDataConfig(),
    "oxe_bridge_euler": OxeBridgeEulerDataConfig(),
    # EEF Quaternion
    "oxe_quat_single": OxeQuatSingleCamDataConfig(),
    "oxe_quat_dual": OxeQuatDualCamDataConfig(),
    "oxe_quat_ur5": OxeQuatUR5DataConfig(),
    # Joint-space
    "oxe_joints7pad_single": OxeJoints7PadSingleCamDataConfig(),
    "oxe_joints7pad_cable": OxeJoints7PadCableRoutingDataConfig(),
    "oxe_joints7g_dual": OxeJoints7GripperDualCamDataConfig(),
    "oxe_joints6pg_dual": OxeJoints6PadGripperDualCamDataConfig(),
    "oxe_joints8_single": OxeJoints8SingleCamDataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG: dict = {
    # EEF Euler
    "oxe_euler_single": EmbodimentTag.OXE_BRIDGE,
    "oxe_euler_wrist": EmbodimentTag.OXE_BRIDGE,
    "oxe_euler_dual": EmbodimentTag.OXE_BRIDGE,
    "oxe_euler_jaco": EmbodimentTag.OXE_BRIDGE,
    "oxe_euler_nyu_franka": EmbodimentTag.OXE_BRIDGE,
    "oxe_droid_local": EmbodimentTag.OXE_DROID,
    "oxe_fmb": EmbodimentTag.OXE_BRIDGE,
    "oxe_bridge_euler": EmbodimentTag.OXE_BRIDGE,
    # EEF Quaternion
    "oxe_quat_single": EmbodimentTag.OXE_RT1,
    "oxe_quat_dual": EmbodimentTag.OXE_RT1,
    "oxe_quat_ur5": EmbodimentTag.OXE_RT1,
    # Joint-space → use NEW_EMBODIMENT since they don't match OXE standard EEF tags
    "oxe_joints7pad_single": EmbodimentTag.NEW_EMBODIMENT,
    "oxe_joints7pad_cable": EmbodimentTag.NEW_EMBODIMENT,
    "oxe_joints7g_dual": EmbodimentTag.NEW_EMBODIMENT,
    "oxe_joints6pg_dual": EmbodimentTag.NEW_EMBODIMENT,
    "oxe_joints8_single": EmbodimentTag.NEW_EMBODIMENT,
}

DATASET_NAMED_MIXTURES: dict = {
    # All 26 OXE datasets
    "oxe_all": [(name, 1.0, rtype) for name, rtype in DATASET_ROBOT_TYPE.items()],
    # All except the language table
    "oxe_no_lang": [(name, 1.0, rtype) for name, rtype in DATASET_ROBOT_TYPE.items() if name != "language_table"],
    # All except the large droid dataset
    "oxe_no_droid": [(name, 1.0, rtype) for name, rtype in DATASET_ROBOT_TYPE.items() if name != "droid"],
    # EEF-only datasets (euler + quaternion, no joint-space)
    "oxe_eef_only": [
        (name, 1.0, rtype)
        for name, rtype in DATASET_ROBOT_TYPE.items()
        if rtype.startswith("oxe_euler")
        or rtype.startswith("oxe_quat")
        or rtype in ("oxe_bridge", "oxe_droid_local", "oxe_fmb")
    ],
    # Sub-groups by state type
    "oxe_euler_datasets": [
        (name, 1.0, rtype)
        for name, rtype in DATASET_ROBOT_TYPE.items()
        if "euler" in rtype or rtype in ("oxe_bridge", "oxe_droid_local", "oxe_fmb")
    ],
    "oxe_quat_datasets": [(name, 1.0, rtype) for name, rtype in DATASET_ROBOT_TYPE.items() if "quat" in rtype],
    "oxe_joint_datasets": [(name, 1.0, rtype) for name, rtype in DATASET_ROBOT_TYPE.items() if "joints" in rtype],
    # Bridge datasets (existing oxe_bridge type)
    "oxe_bridge_datasets": [
        ("bridge_dataset", 1.0, "oxe_bridge"),
        ("ipec_bridge_all", 1.0, "oxe_bridge"),
    ],
    # RT-1 / Fractal only
    "oxe_rt1_only": [
        ("fractal20220817_data", 1.0, "oxe_quat_single"),
    ],
    # Droid only
    "oxe_droid_only": [
        ("droid", 1.0, "oxe_droid_local"),
    ],
    # Bridge + RT-1 (the two datasets used for SimplerEnv / OXE-Mix pretraining)
    "bridge_rt_1": [
        ("ipec_bridge_all", 1.0, "oxe_bridge"),
        ("fractal20220817_data", 1.0, "oxe_quat_single"),
    ],
}


# ===========================================================================
# World Model (MindWPI) Configs
# ===========================================================================
# WM variants reuse the same DataConfig classes (inherit all transforms/modality)
# but add compute_action_loss and latent_view_key attributes for the WM pipeline.
# OXE datasets have reliable actions, so compute_action_loss = True for all.


class _WMMixin:
    """Mixin that adds WM-specific attributes to any OXE DataConfig."""

    compute_action_loss = True
    latent_view_key = None  # auto-detect first non-wrist primary view


class OxeBridgeEulerWMDataConfig(_WMMixin, OxeBridgeEulerDataConfig):
    pass


class OxeEulerWristOnlyWMDataConfig(_WMMixin, OxeEulerWristOnlyDataConfig):
    pass


class OxeEulerDualCamWMDataConfig(_WMMixin, OxeEulerDualCamDataConfig):
    pass


class OxeEulerJacoWMDataConfig(_WMMixin, OxeEulerJacoDataConfig):
    pass


class OxeEulerNyuFrankaWMDataConfig(_WMMixin, OxeEulerNyuFrankaDataConfig):
    pass


class OxeDroidLocalWMDataConfig(_WMMixin, OxeDroidLocalDataConfig):
    pass


class OxeFMBWMDataConfig(_WMMixin, OxeFMBDataConfig):
    pass


class OxeEulerSingleCamWMDataConfig(_WMMixin, OxeEulerSingleCamDataConfig):
    pass


class OxeQuatSingleCamWMDataConfig(_WMMixin, OxeQuatSingleCamDataConfig):
    pass


class OxeQuatDualCamWMDataConfig(_WMMixin, OxeQuatDualCamDataConfig):
    pass


class OxeQuatUR5WMDataConfig(_WMMixin, OxeQuatUR5DataConfig):
    pass


class OxeJoints7PadSingleCamWMDataConfig(_WMMixin, OxeJoints7PadSingleCamDataConfig):
    pass


class OxeJoints7PadCableRoutingWMDataConfig(_WMMixin, OxeJoints7PadCableRoutingDataConfig):
    pass


class OxeJoints7GripperDualCamWMDataConfig(_WMMixin, OxeJoints7GripperDualCamDataConfig):
    pass


class OxeJoints6PadGripperDualCamWMDataConfig(_WMMixin, OxeJoints6PadGripperDualCamDataConfig):
    pass


class OxeJoints8SingleCamWMDataConfig(_WMMixin, OxeJoints8SingleCamDataConfig):
    pass


# WM robot-type registry — maps "<original_type>_wm" to the WM variant class
_WM_CONFIG_MAP = {
    "oxe_euler_single_wm": OxeEulerSingleCamWMDataConfig(),
    "oxe_euler_wrist_wm": OxeEulerWristOnlyWMDataConfig(),
    "oxe_euler_dual_wm": OxeEulerDualCamWMDataConfig(),
    "oxe_euler_jaco_wm": OxeEulerJacoWMDataConfig(),
    "oxe_euler_nyu_franka_wm": OxeEulerNyuFrankaWMDataConfig(),
    "oxe_droid_local_wm": OxeDroidLocalWMDataConfig(),
    "oxe_fmb_wm": OxeFMBWMDataConfig(),
    "oxe_bridge_euler_wm": OxeBridgeEulerWMDataConfig(),
    "oxe_quat_single_wm": OxeQuatSingleCamWMDataConfig(),
    "oxe_quat_dual_wm": OxeQuatDualCamWMDataConfig(),
    "oxe_quat_ur5_wm": OxeQuatUR5WMDataConfig(),
    "oxe_joints7pad_single_wm": OxeJoints7PadSingleCamWMDataConfig(),
    "oxe_joints7pad_cable_wm": OxeJoints7PadCableRoutingWMDataConfig(),
    "oxe_joints7g_dual_wm": OxeJoints7GripperDualCamWMDataConfig(),
    "oxe_joints6pg_dual_wm": OxeJoints6PadGripperDualCamWMDataConfig(),
    "oxe_joints8_single_wm": OxeJoints8SingleCamWMDataConfig(),
}

ROBOT_TYPE_CONFIG_MAP.update(_WM_CONFIG_MAP)

# WM embodiment tags (same as base types)
_WM_EMBODIMENT_TAG_MAP = {f"{k}_wm": v for k, v in ROBOT_TYPE_TO_EMBODIMENT_TAG.items()}
ROBOT_TYPE_TO_EMBODIMENT_TAG.update(_WM_EMBODIMENT_TAG_MAP)

# WM named mixtures — mirror all base mixtures with "_wm" suffix robot types
DATASET_NAMED_MIXTURES["oxe_all_wm"] = [(name, 1.0, f"{rtype}_wm") for name, rtype in DATASET_ROBOT_TYPE.items()]

DATASET_NAMED_MIXTURES["bridge_rt_1_wm"] = [
    ("ipec_bridge_all", 1.0, "oxe_bridge_euler_wm"),
    ("fractal20220817_data", 1.0, "oxe_quat_single_wm"),
]

DATASET_NAMED_MIXTURES["oxe_droid_only_wm"] = [
    ("droid", 1.0, "oxe_droid_local_wm"),
]

# All except the language table (WM counterpart of "oxe_no_lang").
DATASET_NAMED_MIXTURES["oxe_no_lang_wm"] = [
    (name, 1.0, f"{rtype}_wm") for name, rtype in DATASET_ROBOT_TYPE.items() if name != "language_table"
]

# All except the large droid dataset (WM counterpart of "oxe_no_droid").
DATASET_NAMED_MIXTURES["oxe_no_droid_wm"] = [
    (name, 1.0, f"{rtype}_wm") for name, rtype in DATASET_ROBOT_TYPE.items() if name != "droid"
]

# RT-1 / Fractal only (WM counterpart of "oxe_rt1_only").
DATASET_NAMED_MIXTURES["oxe_rt1_only_wm"] = [
    ("fractal20220817_data", 1.0, "oxe_quat_single_wm"),
]
