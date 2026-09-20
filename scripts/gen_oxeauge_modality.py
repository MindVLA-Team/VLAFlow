#!/usr/bin/env python3
"""
Generate meta/modality.json for all OXE_AugE datasets (augmented subset).

Each dataset contains synthetic ee_pose data for 8 robots:
  widowX, sawyer, ur5e, google_robot, jaco, kinova3, kuka_iiwa, xarm7

The ee_pose column per robot is 7-dim: pos[3] + quat[4] (no gripper).
Since there is no standalone `action` column, actions are derived from
the delta of consecutive ee_pose observations (`absolute: false`).

A single comprehensive modality.json is written per dataset that covers
ALL 8 robots.  Each DataConfig in data_config.py selects its robot's
subset of keys via the modality_config() method.

Usage:
    python scripts/gen_oxeauge_modality.py
    python scripts/gen_oxeauge_modality.py --root /path/to/OXE_AugE --dry_run
    python scripts/gen_oxeauge_modality.py --only_augmented  # skip non-_augmented dirs
"""

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Schema constants
# ---------------------------------------------------------------------------

# OXE_AugE v3.0 robots — all share the same ee_pose / image column layout
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

# ee_pose is 7-dim: pos[0:3] + quat[3:7]  (no gripper column)
EE_POS_START, EE_POS_END = 0, 3
EE_ROT_START, EE_ROT_END = 3, 7  # quaternion (w component last in pytorch3d convention)

# Language annotation: column 'natural_language_instruction' (string)
ANNOTATION = {
    "language.natural_language_instruction": {"original_key": "natural_language_instruction"},
}


def _build_modality_json() -> dict:
    """Build a comprehensive modality.json covering all 8 OXE_AugE robots."""
    state = {}
    action = {}
    video = {}

    for robot in OXE_AUGE_ROBOTS:
        # state — absolute ee_pose
        state[f"{robot}.ee_pos"] = {
            "original_key": f"observation.{robot}.ee_pose",
            "start": EE_POS_START,
            "end": EE_POS_END,
            "dtype": "float32",
            "absolute": True,
        }
        state[f"{robot}.ee_rot"] = {
            "original_key": f"observation.{robot}.ee_pose",
            "start": EE_ROT_START,
            "end": EE_ROT_END,
            "dtype": "float32",
            "absolute": True,
            "rotation_type": "quaternion",  # wxyz order; converted to 3D Euler XYZ at load time
        }

        # action — delta of ee_pose (absolute=false triggers delta computation)
        action[f"{robot}.ee_pos"] = {
            "original_key": f"observation.{robot}.ee_pose",
            "start": EE_POS_START,
            "end": EE_POS_END,
            "dtype": "float32",
            "absolute": False,
        }
        action[f"{robot}.ee_rot"] = {
            "original_key": f"observation.{robot}.ee_pose",
            "start": EE_ROT_START,
            "end": EE_ROT_END,
            "dtype": "float32",
            "absolute": False,
            "rotation_type": "quaternion",  # wxyz order; converted to 3D Euler XYZ at load time
        }

        # video
        video[f"{robot}.image"] = {"original_key": f"observation.images.{robot}"}

    return {
        "state": state,
        "action": action,
        "video": video,
        "annotation": ANNOTATION,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate modality.json for OXE_AugE datasets")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/OXE_AugE"),
        help="Root directory containing OXE_AugE dataset sub-folders",
    )
    parser.add_argument(
        "--only_augmented",
        action="store_true",
        default=True,
        help="Only process directories ending in '_augmented' (skip inpainting/other dirs)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print without writing")
    args = parser.parse_args()

    root: Path = args.root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"OXE_AugE root not found: {root}")

    modality = _build_modality_json()

    written = skipped = 0
    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir():
            continue

        if args.only_augmented and not dataset_dir.name.endswith("_augmented"):
            skipped += 1
            continue

        # Verify it has a meta/info.json (basic structure check)
        info_path = dataset_dir / "meta" / "info.json"
        if not info_path.exists():
            print(f"[SKIP]  no meta/info.json: {dataset_dir.name}")
            skipped += 1
            continue

        meta_dir = dataset_dir / "meta"
        modality_path = meta_dir / "modality.json"

        if args.dry_run:
            print(f"[DRY]  {dataset_dir.name}")
        else:
            meta_dir.mkdir(parents=True, exist_ok=True)
            with open(modality_path, "w") as f:
                json.dump(modality, f, indent=4)
            print(f"[OK]   {dataset_dir.name}")
            written += 1

    if args.dry_run:
        print(f"\n[DRY] Would write modality.json to {written + (len(list(root.iterdir())) - skipped)} dirs")
    else:
        print(f"\nDone. Written={written}, Skipped={skipped}")


if __name__ == "__main__":
    main()
