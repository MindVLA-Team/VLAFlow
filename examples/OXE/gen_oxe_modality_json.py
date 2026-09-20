#!/usr/bin/env python3
"""
Generate modality.json for all Open-X Embodiment (OXE) datasets
converted to LeRobot format.

Each modality.json tells the vlaflow dataloader how to map the raw
LeRobot parquet columns (observation.state, action, observation.images.*)
to the named modality keys used by the DataConfig classes.

State layout varies across datasets.  This script reads each dataset's
meta/info.json to determine the actual state layout and writes the
correct modality.json accordingly.

Usage:
    python examples/OXE/gen_oxe_modality_json.py
    python examples/OXE/gen_oxe_modality_json.py --oxe_root /path/to/oxe --dry_run
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# State layout definitions
#
# Each dataset's observation.state is an 8-D float32 vector.  The MEANING of
# those 8 values differs across datasets:
#
#   TYPE_EULER      (13 datasets):  x, y, z, roll, pitch, yaw, pad, gripper
#   TYPE_QUAT       ( 5 datasets):  x, y, z, rx,   ry,   rz,  rw,  gripper
#   TYPE_7J_PAD     ( 3 datasets):  motor_0..6, pad
#   TYPE_7J_GRIPPER ( 3 datasets):  motor_0..6, gripper
#   TYPE_6J_PAD_G   ( 1 dataset ):  motor_0..5, pad, gripper
#   TYPE_8J         ( 1 dataset ):  motor_0..7
# ---------------------------------------------------------------------------

# All actions are 7-D with the same EEF-space format across every dataset:
#   x, y, z, roll, pitch, yaw, gripper
STANDARD_ACTION = {
    "x": {"start": 0, "end": 1},
    "y": {"start": 1, "end": 2},
    "z": {"start": 2, "end": 3},
    "roll": {"start": 3, "end": 4},
    "pitch": {"start": 4, "end": 5},
    "yaw": {"start": 5, "end": 6},
    "gripper": {"start": 6, "end": 7},
}

# Language annotation: task_index column → annotation key
STANDARD_ANNOTATION = {
    "human.action.task_description": {"original_key": "task_index"},
}

# EEF euler  (x y z roll pitch yaw pad gripper)
STATE_EULER = {
    "x": {"start": 0, "end": 1},
    "y": {"start": 1, "end": 2},
    "z": {"start": 2, "end": 3},
    "roll": {"start": 3, "end": 4},
    "pitch": {"start": 4, "end": 5},
    "yaw": {"start": 5, "end": 6},
    "pad": {"start": 6, "end": 7},
    "gripper": {"start": 7, "end": 8},
}

# EEF quaternion  (x y z rx ry rz rw gripper)
STATE_QUAT = {
    "x": {"start": 0, "end": 1},
    "y": {"start": 1, "end": 2},
    "z": {"start": 2, "end": 3},
    "rx": {"start": 3, "end": 4},
    "ry": {"start": 4, "end": 5},
    "rz": {"start": 5, "end": 6},
    "rw": {"start": 6, "end": 7},
    "gripper": {"start": 7, "end": 8},
}

# Joint-space: 7 joints + pad (no gripper dimension)
STATE_7J_PAD = {
    "motor_0": {"start": 0, "end": 1},
    "motor_1": {"start": 1, "end": 2},
    "motor_2": {"start": 2, "end": 3},
    "motor_3": {"start": 3, "end": 4},
    "motor_4": {"start": 4, "end": 5},
    "motor_5": {"start": 5, "end": 6},
    "motor_6": {"start": 6, "end": 7},
    "pad": {"start": 7, "end": 8},
}

# Joint-space: 7 joints + gripper
STATE_7J_GRIPPER = {
    "motor_0": {"start": 0, "end": 1},
    "motor_1": {"start": 1, "end": 2},
    "motor_2": {"start": 2, "end": 3},
    "motor_3": {"start": 3, "end": 4},
    "motor_4": {"start": 4, "end": 5},
    "motor_5": {"start": 5, "end": 6},
    "motor_6": {"start": 6, "end": 7},
    "gripper": {"start": 7, "end": 8},
}

# Joint-space: 6 joints + pad + gripper
STATE_6J_PAD_GRIPPER = {
    "motor_0": {"start": 0, "end": 1},
    "motor_1": {"start": 1, "end": 2},
    "motor_2": {"start": 2, "end": 3},
    "motor_3": {"start": 3, "end": 4},
    "motor_4": {"start": 4, "end": 5},
    "motor_5": {"start": 5, "end": 6},
    "pad": {"start": 6, "end": 7},
    "gripper": {"start": 7, "end": 8},
}

# Joint-space: 8 unnamed joints (roboturk)
STATE_8J = {
    "motor_0": {"start": 0, "end": 1},
    "motor_1": {"start": 1, "end": 2},
    "motor_2": {"start": 2, "end": 3},
    "motor_3": {"start": 3, "end": 4},
    "motor_4": {"start": 4, "end": 5},
    "motor_5": {"start": 5, "end": 6},
    "motor_6": {"start": 6, "end": 7},
    "motor_7": {"start": 7, "end": 8},
}

# ---------------------------------------------------------------------------
# Per-dataset configuration
#
# Each entry maps a dataset folder name to:
#   state  : one of the STATE_* dicts above
#   video  : {alias → original_key}  – camera mapping
#   robot_type : the robot type string for ROBOT_TYPE_CONFIG_MAP
# ---------------------------------------------------------------------------

DATASET_CONFIG: dict[str, dict] = {
    # ── EEF Euler state  ────────────────────────────────────────────────────
    "bc_z": {
        "state": STATE_EULER,
        "video": {"image": "observation.images.image"},
        "robot_type": "oxe_euler_single",
    },
    "cmu_stretch": {
        "state": STATE_EULER,
        "video": {"image": "observation.images.image"},
        "robot_type": "oxe_euler_single",
    },
    "dlr_edan_shared_control_converted_externally_to_rlds": {
        "state": STATE_EULER,
        "video": {"image": "observation.images.image"},
        "robot_type": "oxe_euler_single",
    },
    # rgb → aliased to "image"
    "language_table": {
        "state": STATE_EULER,
        "video": {"image": "observation.images.rgb"},
        "robot_type": "oxe_euler_single",
    },
    # wrist-only
    "dobbe": {
        "state": STATE_EULER,
        "video": {"wrist_image": "observation.images.wrist_image"},
        "robot_type": "oxe_euler_wrist",
    },
    # dual cam
    "stanford_hydra_dataset_converted_externally_to_rlds": {
        "state": STATE_EULER,
        "video": {
            "image": "observation.images.image",
            "wrist_image": "observation.images.wrist_image",
        },
        "robot_type": "oxe_euler_dual",
    },
    # rgb_static → image,  rgb_gripper → wrist_image
    "taco_play": {
        "state": STATE_EULER,
        "video": {
            "image": "observation.images.rgb_static",
            "wrist_image": "observation.images.rgb_gripper",
        },
        "robot_type": "oxe_euler_dual",
    },
    # image + image_wrist (jaco)
    "jaco_play": {
        "state": STATE_EULER,
        "video": {
            "image": "observation.images.image",
            "image_wrist": "observation.images.image_wrist",
        },
        "robot_type": "oxe_euler_jaco",
    },
    # image + image_additional_view
    "nyu_franka_play_dataset_converted_externally_to_rlds": {
        "state": STATE_EULER,
        "video": {
            "image": "observation.images.image",
            "image_additional_view": "observation.images.image_additional_view",
        },
        "robot_type": "oxe_euler_nyu_franka",
    },
    # bridge-style: image_0..3  (existing oxe_bridge DataConfig uses image_0 only)
    "bridge_dataset": {
        "state": STATE_EULER,
        "video": {
            "image_0": "observation.images.image_0",
            "image_1": "observation.images.image_1",
            "image_2": "observation.images.image_2",
            "image_3": "observation.images.image_3",
        },
        "robot_type": "oxe_bridge",
    },
    "ipec_bridge_all": {
        "state": STATE_EULER,
        "video": {
            "image_0": "observation.images.image_0",
            "image_1": "observation.images.image_1",
            "image_2": "observation.images.image_2",
            "image_3": "observation.images.image_3",
        },
        "robot_type": "oxe_bridge",
    },
    # droid: 3 cameras (_left suffix after lerobot conversion)
    "droid": {
        "state": STATE_EULER,
        "video": {
            "exterior_image_1_left": "observation.images.exterior_image_1_left",
            "exterior_image_2_left": "observation.images.exterior_image_2_left",
            "wrist_image_left": "observation.images.wrist_image_left",
        },
        "robot_type": "oxe_droid_local",
    },
    # fmb: 4 cameras
    "fmb": {
        "state": STATE_EULER,
        "video": {
            "image_side_1": "observation.images.image_side_1",
            "image_side_2": "observation.images.image_side_2",
            "image_wrist_1": "observation.images.image_wrist_1",
            "image_wrist_2": "observation.images.image_wrist_2",
        },
        "robot_type": "oxe_fmb",
    },
    # ── EEF Quaternion state  ────────────────────────────────────────────────
    "fractal20220817_data": {
        "state": STATE_QUAT,
        "video": {"image": "observation.images.image"},
        "robot_type": "oxe_quat_single",
    },
    "kuka": {
        "state": STATE_QUAT,
        "video": {"image": "observation.images.image"},
        "robot_type": "oxe_quat_single",
    },
    "austin_sailor_dataset_converted_externally_to_rlds": {
        "state": STATE_QUAT,
        "video": {
            "image": "observation.images.image",
            "wrist_image": "observation.images.wrist_image",
        },
        "robot_type": "oxe_quat_dual",
    },
    "austin_sirius_dataset_converted_externally_to_rlds": {
        "state": STATE_QUAT,
        "video": {
            "image": "observation.images.image",
            "wrist_image": "observation.images.wrist_image",
        },
        "robot_type": "oxe_quat_dual",
    },
    # ur5: image + hand_image
    "berkeley_autolab_ur5": {
        "state": STATE_QUAT,
        "video": {
            "image": "observation.images.image",
            "hand_image": "observation.images.hand_image",
        },
        "robot_type": "oxe_quat_ur5",
    },
    # ── Joint-space state: motor_0..6 + pad  ────────────────────────────────
    "toto": {
        "state": STATE_7J_PAD,
        "video": {"image": "observation.images.image"},
        "robot_type": "oxe_joints7pad_single",
    },
    "ucsd_kitchen_dataset_converted_externally_to_rlds": {
        "state": STATE_7J_PAD,
        "video": {"image": "observation.images.image"},
        "robot_type": "oxe_joints7pad_single",
    },
    # cable routing: 4 cameras
    "berkeley_cable_routing": {
        "state": STATE_7J_PAD,
        "video": {
            "image": "observation.images.image",
            "wrist45_image": "observation.images.wrist45_image",
            "top_image": "observation.images.top_image",
            "wrist225_image": "observation.images.wrist225_image",
        },
        "robot_type": "oxe_joints7pad_cable",
    },
    # ── Joint-space state: motor_0..6 + gripper  ────────────────────────────
    # motor_0..6 per info.json names, but gripper is actual last element
    "austin_buds_dataset_converted_externally_to_rlds": {
        "state": STATE_7J_GRIPPER,
        "video": {
            "image": "observation.images.image",
            "wrist_image": "observation.images.wrist_image",
        },
        "robot_type": "oxe_joints7g_dual",
    },
    "utaustin_mutex": {
        "state": STATE_7J_GRIPPER,
        "video": {
            "image": "observation.images.image",
            "wrist_image": "observation.images.wrist_image",
        },
        "robot_type": "oxe_joints7g_dual",
    },
    # agentview_rgb → image,  eye_in_hand_rgb → wrist_image
    "viola": {
        "state": STATE_7J_GRIPPER,
        "video": {
            "image": "observation.images.agentview_rgb",
            "wrist_image": "observation.images.eye_in_hand_rgb",
        },
        "robot_type": "oxe_joints7g_dual",
    },
    # ── Joint-space state: motor_0..5 + pad + gripper  ──────────────────────
    "berkeley_fanuc_manipulation": {
        "state": STATE_6J_PAD_GRIPPER,
        "video": {
            "image": "observation.images.image",
            "wrist_image": "observation.images.wrist_image",
        },
        "robot_type": "oxe_joints6pg_dual",
    },
    # ── Joint-space state: motor_0..7 (8 joints, no gripper dim)  ───────────
    # front_rgb → aliased to "image"
    "roboturk": {
        "state": STATE_8J,
        "video": {"image": "observation.images.front_rgb"},
        "robot_type": "oxe_joints8_single",
    },
}


def build_modality_json(state: dict, video: dict[str, str]) -> dict:
    """Assemble a complete modality.json dict."""
    return {
        "state": state,
        "action": STANDARD_ACTION,
        "video": {alias: {"original_key": original} for alias, original in video.items()},
        "annotation": STANDARD_ANNOTATION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate modality.json for OXE datasets")
    parser.add_argument(
        "--oxe_root",
        type=Path,
        default=Path("data/openx-embodiment-lerobot"),
        help="Root directory containing all OXE dataset sub-folders",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    oxe_root: Path = args.oxe_root.resolve()
    if not oxe_root.is_dir():
        raise FileNotFoundError(f"OXE root not found: {oxe_root}")

    written = missing = 0
    for dataset_name, cfg in sorted(DATASET_CONFIG.items()):
        dataset_dir = oxe_root / dataset_name
        if not dataset_dir.is_dir():
            print(f"[SKIP]  not found: {dataset_dir}")
            missing += 1
            continue

        meta_dir = dataset_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        modality_path = meta_dir / "modality.json"

        modality = build_modality_json(cfg["state"], cfg["video"])

        if args.dry_run:
            print(f"[DRY]   {modality_path}  state_type={_state_type_name(cfg['state'])}")
            print(json.dumps(modality, indent=4))
            print()
        else:
            with open(modality_path, "w") as f:
                json.dump(modality, f, indent=4)
            state_keys = list(cfg["state"].keys())
            print(f"[OK]    {dataset_name:<60}  state={state_keys}  robot_type={cfg['robot_type']}")
            written += 1

    action = "Would write" if args.dry_run else "Written"
    print(f"\nDone. {action}={written}, Missing={missing}")


def _state_type_name(state: dict) -> str:
    keys = list(state.keys())
    if "roll" in keys:
        return "euler"
    if "rx" in keys:
        return "quat"
    if "motor_7" in keys:
        return "8joints"
    if "pad" in keys and "gripper" in keys:
        return "6j+pad+gripper"
    if "gripper" in keys:
        return "7j+gripper"
    return "7j+pad"


if __name__ == "__main__":
    main()
