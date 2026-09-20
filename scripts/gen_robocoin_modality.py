#!/usr/bin/env python3
"""
Generate meta/modality.json for all RoboCOIN_SUB datasets.

Three robot types are handled:
  * Agilex_Cobot_Magic  — 26-D state/action, EEF pose sliced from observation.state / action
  * Galaxea_R1_Lite     — 12-D eef_sim_pose_state/action + grippers from observation.state
  * Agilex_Split_Aloha  — same template as Agilex_Cobot_Magic

Usage:
    python scripts/gen_robocoin_modality.py
    python scripts/gen_robocoin_modality.py --root /path/to/RoboCOIN_SUB --dry_run
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Modality templates
# ---------------------------------------------------------------------------

# Agilex_Cobot_Magic observation.state layout (26-dim):
#   [0-5]  left_arm_joints (6)
#   [6]    left_gripper_open
#   [7:10] left_eef_pos (x, y, z)
#   [10:13] left_eef_rot_euler (x, y, z)
#   [13-18] right_arm_joints (6)
#   [19]   right_gripper_open
#   [20:23] right_eef_pos (x, y, z)
#   [23:26] right_eef_rot_euler (x, y, z)
AGILEX_COBOT_MAGIC_STATE = {
    "left.ee_pos": {"original_key": "observation.state", "start": 7, "end": 10, "dtype": "float32", "absolute": True},
    "left.ee_rot": {"original_key": "observation.state", "start": 10, "end": 13, "dtype": "float32", "absolute": True},
    "left.gripper": {"original_key": "observation.state", "start": 6, "end": 7, "dtype": "float32", "absolute": True},
    "right.ee_pos": {"original_key": "observation.state", "start": 20, "end": 23, "dtype": "float32", "absolute": True},
    "right.ee_rot": {"original_key": "observation.state", "start": 23, "end": 26, "dtype": "float32", "absolute": True},
    "right.gripper": {"original_key": "observation.state", "start": 19, "end": 20, "dtype": "float32", "absolute": True},
}

AGILEX_COBOT_MAGIC_ACTION = {
    "left.ee_pos": {"original_key": "action", "start": 7, "end": 10, "dtype": "float32", "absolute": False},
    "left.ee_rot": {"original_key": "action", "start": 10, "end": 13, "dtype": "float32", "absolute": False},
    "left.gripper": {"original_key": "action", "start": 6, "end": 7, "dtype": "float32", "absolute": False},
    "right.ee_pos": {"original_key": "action", "start": 20, "end": 23, "dtype": "float32", "absolute": False},
    "right.ee_rot": {"original_key": "action", "start": 23, "end": 26, "dtype": "float32", "absolute": False},
    "right.gripper": {"original_key": "action", "start": 19, "end": 20, "dtype": "float32", "absolute": False},
}

AGILEX_COBOT_MAGIC_VIDEO = {
    "cam_head_rgb": {"original_key": "observation.images.cam_head_rgb"},
    "cam_left_wrist_rgb": {"original_key": "observation.images.cam_left_wrist_rgb"},
    "cam_right_wrist_rgb": {"original_key": "observation.images.cam_right_wrist_rgb"},
}

AGILEX_COBOT_MAGIC_ANNOTATION = {
    "language.task": {"original_key": "task_index"},
}


# Galaxea_R1_Lite:
#   eef_sim_pose_state / eef_sim_pose_action (12-dim):
#     [0:3]  left_eef_pos (x, y, z)
#     [3:6]  left_eef_rot_euler (x, y, z)
#     [6:9]  right_eef_pos (x, y, z)
#     [9:12] right_eef_rot_euler (x, y, z)
#   observation.state / action (14-dim):
#     [12]   left_gripper_open
#     [13]   right_gripper_open
GALAXEA_STATE = {
    "left.ee_pos": {"original_key": "eef_sim_pose_state", "start": 0, "end": 3, "dtype": "float32", "absolute": True},
    "left.ee_rot": {"original_key": "eef_sim_pose_state", "start": 3, "end": 6, "dtype": "float32", "absolute": True},
    "left.gripper": {"original_key": "observation.state", "start": 12, "end": 13, "dtype": "float32", "absolute": True},
    "right.ee_pos": {"original_key": "eef_sim_pose_state", "start": 6, "end": 9, "dtype": "float32", "absolute": True},
    "right.ee_rot": {"original_key": "eef_sim_pose_state", "start": 9, "end": 12, "dtype": "float32", "absolute": True},
    "right.gripper": {"original_key": "observation.state", "start": 13, "end": 14, "dtype": "float32", "absolute": True},
}

GALAXEA_ACTION = {
    "left.ee_pos": {"original_key": "eef_sim_pose_action", "start": 0, "end": 3, "dtype": "float32", "absolute": False},
    "left.ee_rot": {"original_key": "eef_sim_pose_action", "start": 3, "end": 6, "dtype": "float32", "absolute": False},
    "left.gripper": {"original_key": "action", "start": 12, "end": 13, "dtype": "float32", "absolute": False},
    "right.ee_pos": {"original_key": "eef_sim_pose_action", "start": 6, "end": 9, "dtype": "float32", "absolute": False},
    "right.ee_rot": {
        "original_key": "eef_sim_pose_action",
        "start": 9,
        "end": 12,
        "dtype": "float32",
        "absolute": False,
    },
    "right.gripper": {"original_key": "action", "start": 13, "end": 14, "dtype": "float32", "absolute": False},
}

GALAXEA_VIDEO = {
    "cam_head_left_rgb": {"original_key": "observation.images.cam_head_left_rgb"},
    "cam_head_right_rgb": {"original_key": "observation.images.cam_head_right_rgb"},
    "cam_left_wrist_rgb": {"original_key": "observation.images.cam_left_wrist_rgb"},
    "cam_right_wrist_rgb": {"original_key": "observation.images.cam_right_wrist_rgb"},
}

GALAXEA_ANNOTATION = {
    "language.task": {"original_key": "task_index"},
}


def _build_modality_json(state: dict, action: dict, video: dict, annotation: dict) -> dict:
    return {
        "state": state,
        "action": action,
        "video": video,
        "annotation": annotation,
    }


def _detect_robot_type(dataset_name: str) -> str:
    """Infer robot type from the dataset directory name prefix."""
    if dataset_name.startswith("Agilex_Cobot_Magic"):
        return "agilex_cobot_magic"
    if dataset_name.startswith("Galaxea_R1_Lite"):
        return "galaxea_r1_lite"
    if dataset_name.startswith("Agilex_Split_Aloha"):
        return "agilex_split_aloha"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate modality.json for RoboCOIN_SUB datasets")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/RoboCOIN_SUB"),
        help="Root directory containing RoboCOIN_SUB dataset sub-folders",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"RoboCOIN_SUB root not found: {root}")

    written = skipped = unknown = 0
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        robot_type = _detect_robot_type(dataset_dir.name)
        if robot_type == "unknown":
            print(f"[SKIP]  unknown prefix: {dataset_dir.name}")
            unknown += 1
            continue

        if robot_type in ("agilex_cobot_magic", "agilex_split_aloha"):
            modality = _build_modality_json(
                AGILEX_COBOT_MAGIC_STATE,
                AGILEX_COBOT_MAGIC_ACTION,
                AGILEX_COBOT_MAGIC_VIDEO,
                AGILEX_COBOT_MAGIC_ANNOTATION,
            )
        else:  # galaxea_r1_lite
            modality = _build_modality_json(
                GALAXEA_STATE,
                GALAXEA_ACTION,
                GALAXEA_VIDEO,
                GALAXEA_ANNOTATION,
            )

        meta_dir = dataset_dir / "meta"
        modality_path = meta_dir / "modality.json"

        if args.dry_run:
            print(f"[DRY]  {dataset_dir.name}  robot_type={robot_type}")
            print(json.dumps(modality, indent=4))
            print()
        else:
            meta_dir.mkdir(parents=True, exist_ok=True)
            with open(modality_path, "w") as f:
                json.dump(modality, f, indent=4)
            print(f"[OK]   {dataset_dir.name:<70}  robot_type={robot_type}")
            written += 1

    action_word = "Would write" if args.dry_run else "Written"
    print(f"\nDone. {action_word}={written}, Unknown={unknown}, Skipped={skipped}")


if __name__ == "__main__":
    main()
