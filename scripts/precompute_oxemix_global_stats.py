#!/usr/bin/env python3
"""
Offline script: compute global normalization statistics for OXE-Mix.

Following LAP's approach:
  global_q01[dim] = min(q01 across all datasets)   ← most conservative lower bound
  global_q99[dim] = max(q99 across all datasets)   ← most conservative upper bound

This produces a unified normalization scale so that verbose_eef_with_rotation format
gives consistent human-readable values across all 800+ OXE-Mix datasets.

Usage:
    cd /path/to/vlaflow
    python scripts/precompute_oxemix_global_stats.py \
        --data_root_dir data \
        --data_mix oxe_mix_full \
        --output_path data/oxemix_global_norm_stats.json

    # Or faster — use existing dataset_statistics.json if available:
    python scripts/precompute_oxemix_global_stats.py \
        --from_dataset_statistics /path/to/dataset_statistics.json \
        --output_path /path/to/oxemix_global_norm_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# ── Action key definitions ──────────────────────────────────────────────────
# OXE single-arm: action.x/y/z (translation m), action.roll/pitch/yaw (rotation rad),
#                  action.gripper (binary)
# OXE-Mix 14D dual-arm layout: [left_arm_7d, right_arm_7d]
#   left:  indices 0-6  → x, y, z, roll, pitch, yaw, gripper
#   right: indices 7-13 → same (zero-padded for single-arm datasets)
OXE_CONTINUOUS_ACTION_KEYS = ["x", "y", "z", "roll", "pitch", "yaw"]
OXE_BINARY_ACTION_KEYS = ["gripper"]
ALL_ACTION_KEYS = OXE_CONTINUOUS_ACTION_KEYS + OXE_BINARY_ACTION_KEYS

# Fallback 14D flat index to key mapping (used when stats are stored as flat arrays)
# Order matches ConcatTransform output in OXE data_config.py
FLAT_14D_KEY_ORDER = [
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "gripper",  # left arm (0-6)
    "x_r",
    "y_r",
    "z_r",
    "roll_r",
    "pitch_r",
    "yaw_r",
    "gripper_r",  # right arm (7-13)
]


# ── Stats file parsers ──────────────────────────────────────────────────────


def _try_parse_per_key(action_data: dict) -> dict | None:
    """Try parsing per-subkey format: {"x": {q01, q99, ...}, "y": {...}, ...}."""
    if "x" in action_data and isinstance(action_data["x"], dict):
        return action_data
    return None


def _try_parse_flat_array(action_data: dict, key_order: list) -> dict | None:
    """
    Try parsing flat-array format: {"q01": [14_floats], "q99": [...], ...}
    and split into per-subkey dicts.
    """
    if "q01" not in action_data:
        return None
    q01 = action_data.get("q01", [])
    q99 = action_data.get("q99", [])
    if not q01:
        return None

    result = {}
    for i, key in enumerate(key_order):
        if i >= len(q01):
            break
        result[key] = {
            "q01": q01[i] if isinstance(q01[i], float) else float(q01[i]),
            "q99": q99[i] if isinstance(q99[i], float) else float(q99[i]),
        }
        for stat in ("mean", "std", "min", "max"):
            vals = action_data.get(stat)
            if vals and i < len(vals):
                result[key][stat] = float(vals[i])
    return result


def _try_parse_gr00t_v2(raw: dict) -> dict | None:
    """Parse GR00T v2 format: {"statistics": {"action.{robot}.ee_pos": {"q01": [3], "q99": [3]}, ...}}.

    OXE_AugE datasets store per-robot 3D translation bounds under
    ``statistics.action.{robot}.ee_pos``.  This function:
      1. Collects per-robot per-dim q01/q99 for ee_pos (x=dim0, y=dim1, z=dim2).
      2. Aggregates to per-dim bounds: q01[dim] = min across robots, q99[dim] = max.
      3. Rotation (roll/pitch/yaw) uses hardcoded synthetic [-π, π] bounds that
         match what OXE_AugE's StateActionTransform uses after quat→euler conversion.

    Returns:
        dict mapping subkey → {q01, q99, min, max}, or None if no ee_pos keys found.
    """
    import math

    if "statistics" not in raw:
        return None
    stats = raw["statistics"]

    # Collect per-dim ee_pos stats across all robots in this dataset
    per_dim_q01: list[list] = [[], [], []]
    per_dim_q99: list[list] = [[], [], []]
    per_dim_min: list[list] = [[], [], []]
    per_dim_max: list[list] = [[], [], []]

    for key, val in stats.items():
        if not (key.startswith("action.") and key.endswith(".ee_pos")):
            continue
        if not isinstance(val, dict):
            continue
        q01 = val.get("q01", [])
        q99 = val.get("q99", [])
        if not q01 or len(q01) < 3:
            continue
        for i in range(3):
            per_dim_q01[i].append(float(q01[i]))
            per_dim_q99[i].append(float(q99[i]))
        mn = val.get("min", [])
        mx = val.get("max", [])
        for i in range(3):
            if mn and i < len(mn):
                per_dim_min[i].append(float(mn[i]))
            if mx and i < len(mx):
                per_dim_max[i].append(float(mx[i]))

    if not per_dim_q01[0]:
        return None  # no ee_pos keys found

    dim_names = ["x", "y", "z"]
    result = {}
    for i, name in enumerate(dim_names):
        entry: dict = {
            "q01": float(np.min(per_dim_q01[i])),
            "q99": float(np.max(per_dim_q99[i])),
        }
        if per_dim_min[i]:
            entry["min"] = float(np.min(per_dim_min[i]))
        if per_dim_max[i]:
            entry["max"] = float(np.max(per_dim_max[i]))
        result[name] = entry

    # Rotation: synthetic [-π, π] bounds matching OXE_AugE pipeline
    PI = math.pi
    for name in ["roll", "pitch", "yaw"]:
        result[name] = {"q01": -PI, "q99": PI, "min": -PI, "max": PI}

    return result


def load_dataset_action_stats(dataset_path: Path) -> dict | None:
    """
    Load action statistics from a LeRobot dataset directory.

    Tries to read meta/stats_gr00t.json and handles multiple possible formats:
      Format V2: {"statistics": {"action.{robot}.ee_pos": {"q01": [3], ...}, ...}}  (GR00T v2, OXE_AugE)
      Format A:  {"action": {"x": {q01, q99, ...}, ...}, ...}
      Format B:  {"action": {"q01": [14_floats], ...}, ...}  (flat array)
      Format C:  {"tag_key": {"action": ...}}  (outer embodiment tag wrapper)

    Returns:
        dict mapping subkey → {q01, q99, mean, std, min, max}, or None on failure.
    """
    stats_path = dataset_path / "meta" / "stats_gr00t.json"
    if not stats_path.exists():
        return None

    try:
        with open(stats_path) as f:
            raw = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    # Try GR00T v2 format first (OXE_AugE datasets with top-level "statistics" key)
    result = _try_parse_gr00t_v2(raw)
    if result is not None:
        return result

    # Unwrap outer tag wrapper if present
    action_data = None
    if "action" in raw:
        action_data = raw["action"]
    else:
        # Look for a nested dict that contains "action"
        for v in raw.values():
            if isinstance(v, dict) and "action" in v:
                action_data = v["action"]
                break

    if action_data is None:
        return None

    # Try per-key format first (more common in GR00T v2)
    result = _try_parse_per_key(action_data)
    if result is not None:
        return result

    # Fall back to flat-array format (7D or 14D)
    n_dims = len(action_data.get("q01", []))
    if n_dims == 7:
        key_order = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]
    elif n_dims == 14:
        key_order = FLAT_14D_KEY_ORDER
    else:
        key_order = FLAT_14D_KEY_ORDER[:n_dims]
    return _try_parse_flat_array(action_data, key_order)


# ── Global stats aggregation ────────────────────────────────────────────────


def compute_global_stats_from_datasets(
    data_root_dir: str,
    data_mix: str,
    min_loaded_fraction: float = 0.5,
) -> dict:
    """
    Walk through all datasets in data_mix, load per-dataset action stats,
    and aggregate to global bounds.

    Raises SystemExit if the fraction of successfully loaded datasets is below
    min_loaded_fraction (default 0.5), so that partial results from a missing
    data mount do not silently produce wrong stats.
    """
    sys.path.insert(0, str(Path(__file__).parent.parent))
    import os

    os.environ["DATA_ROOT"] = str(Path(data_root_dir).resolve())
    from vlaflow.dataloader.gr00t_lerobot.registry import DATASET_NAMED_MIXTURES, validate_mixture_sources

    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    validate_mixture_sources(data_mix, mixture_spec)
    data_root = Path(data_root_dir)

    # Per-key aggregation buffers
    per_key: dict[str, dict] = {
        k: {"q01": [], "q99": [], "min": [], "max": [], "mean_n": [], "n": []} for k in OXE_CONTINUOUS_ACTION_KEYS
    }

    n_loaded = n_skipped = 0
    missing_paths: list[str] = []
    seen = set()

    for d_name, _weight, _robot_type in mixture_spec:
        if d_name in seen:
            continue
        seen.add(d_name)

        dataset_path = data_root / d_name
        stats = load_dataset_action_stats(dataset_path)
        if stats is None:
            missing_paths.append(str(dataset_path))
            n_skipped += 1
            continue

        # Sanity-check units before including this dataset's stats.
        # Delta-EEF actions must be in metres (translation) and radians (rotation).
        # A dataset whose q99 translation exceeds 2 m, or whose rotation exceeds
        # 2π rad, almost certainly stores values in millimetres or degrees — skip it.
        x_stats = stats.get("x", {})
        r_stats = stats.get("roll", {})
        x_range = abs(x_stats.get("q99", 0)) + abs(x_stats.get("q01", 0))
        r_range = abs(r_stats.get("q99", 0)) + abs(r_stats.get("q01", 0))
        if x_range > 2.0 or r_range > 7.0:  # 7 rad > 2π — impossible for angle
            print(
                f"  SKIPPING {d_name}: stats appear to be in wrong units "
                f"(x_range={x_range:.3f}, r_range={r_range:.3f})"
            )
            n_skipped += 1
            missing_paths.append(f"[wrong-units] {dataset_path}")
            continue

        for key in OXE_CONTINUOUS_ACTION_KEYS:
            if key not in stats:
                continue
            s = stats[key]
            q01 = s.get("q01")
            q99 = s.get("q99")
            if q01 is None or q99 is None:
                continue
            per_key[key]["q01"].append(float(q01))
            per_key[key]["q99"].append(float(q99))
            if "min" in s:
                per_key[key]["min"].append(float(s["min"]))
            if "max" in s:
                per_key[key]["max"].append(float(s["max"]))
            # Weighted mean (if available)
            if "mean" in s and "num_transitions" in stats:
                n = int(stats.get("num_transitions", 1))
                per_key[key]["mean_n"].append(float(s["mean"]) * n)
                per_key[key]["n"].append(n)

        n_loaded += 1
        if n_loaded % 50 == 0:
            print(f"  Processed {n_loaded} datasets...")

    total = n_loaded + n_skipped
    loaded_fraction = n_loaded / total if total > 0 else 0.0
    print(f"\nLoaded: {n_loaded}  Skipped: {n_skipped}  ({loaded_fraction:.1%} of {total} unique datasets)")

    # wrong-units entries are in missing_paths but not truly "missing" — separate them
    wrong_units = [p for p in missing_paths if p.startswith("[wrong-units]")]
    truly_missing = [p for p in missing_paths if not p.startswith("[wrong-units]")]
    if wrong_units:
        print(
            f"  Excluded {len(wrong_units)} dataset(s) with wrong units: "
            + ", ".join(p.replace("[wrong-units] ", "") for p in wrong_units[:5])
        )

    truly_loaded_fraction = n_loaded / (n_loaded + len(truly_missing)) if (n_loaded + len(truly_missing)) > 0 else 0.0
    if min_loaded_fraction > 0.0 and truly_loaded_fraction < min_loaded_fraction:
        print(
            f"\nERROR: only {truly_loaded_fraction:.1%} of datasets were found on disk "
            f"(threshold: {min_loaded_fraction:.1%})."
        )
        print(f"  {len(truly_missing)} dataset paths are missing. First 20:")
        for p in truly_missing[:20]:
            print(f"    {p}")
        if len(truly_missing) > 20:
            print(f"    ... and {len(truly_missing) - 20} more")
        print("\nPlease mount or download the missing data, then re-run this script.")
        sys.exit(1)

    return _aggregate_per_key(per_key)


def compute_global_stats_from_statistics_json(statistics_json_path: str) -> dict:
    """
    Compute global stats from an existing dataset_statistics.json (saved by LeRobotMixtureDataset).

    This is much faster than re-loading all datasets from disk.
    Format of dataset_statistics.json:
      {
        "embodiment_tag_dataset": {
          "action": {"q01": [14_floats], "q99": [...], ...},
          "num_transitions": N
        },
        ...
      }
    """
    with open(statistics_json_path) as f:
        all_stats = json.load(f)

    per_key: dict[str, dict] = {
        k: {"q01": [], "q99": [], "min": [], "max": [], "mean_n": [], "n": []} for k in OXE_CONTINUOUS_ACTION_KEYS
    }

    n_entries = 0
    for tag, entry in all_stats.items():
        if "action" not in entry:
            continue
        action_data = entry["action"]

        # Parse action stats
        stats = _try_parse_per_key(action_data)
        if stats is None:
            n_dims = len(action_data.get("q01", []))
            key_order = (
                ["x", "y", "z", "roll", "pitch", "yaw", "gripper"] if n_dims == 7 else FLAT_14D_KEY_ORDER[:n_dims]
            )
            stats = _try_parse_flat_array(action_data, key_order)
        if stats is None:
            continue

        # Sanity-check units before including this entry (mirrors the disk-scan path).
        # Delta-EEF translation must be in metres and rotation in radians. A q99
        # translation above 2 m or rotation above 7 rad (>2pi) means the stats are
        # in millimetres or degrees — skip so they don't poison the global bounds.
        x_stats = stats.get("x", {})
        r_stats = stats.get("roll", {})
        x_range = abs(x_stats.get("q99", 0)) + abs(x_stats.get("q01", 0))
        r_range = abs(r_stats.get("q99", 0)) + abs(r_stats.get("q01", 0))
        if x_range > 2.0 or r_range > 7.0:
            print(
                f"  SKIPPING {tag}: stats appear to be in wrong units " f"(x_range={x_range:.3f}, r_range={r_range:.3f})"
            )
            continue

        num_t = int(entry.get("num_transitions", 1))
        for key in OXE_CONTINUOUS_ACTION_KEYS:
            if key not in stats:
                continue
            s = stats[key]
            per_key[key]["q01"].append(float(s.get("q01", s.get("q01", 0))))
            per_key[key]["q99"].append(float(s.get("q99", 1)))
            if "min" in s:
                per_key[key]["min"].append(float(s["min"]))
            if "max" in s:
                per_key[key]["max"].append(float(s["max"]))
            if "mean" in s:
                per_key[key]["mean_n"].append(float(s["mean"]) * num_t)
                per_key[key]["n"].append(num_t)
        n_entries += 1

    print(f"Parsed {n_entries} entries from {statistics_json_path}")
    return _aggregate_per_key(per_key)


def _aggregate_per_key(per_key: dict) -> dict:
    """Aggregate per-key buffers into global stats dict."""
    global_stats = {}
    for key in OXE_CONTINUOUS_ACTION_KEYS:
        bufs = per_key[key]
        if not bufs["q01"]:
            print(f"  WARNING: no stats found for action.{key}")
            continue
        result = {
            "q01": float(np.min(bufs["q01"])),
            "q99": float(np.max(bufs["q99"])),
        }
        if bufs["min"]:
            result["min"] = float(np.min(bufs["min"]))
        if bufs["max"]:
            result["max"] = float(np.max(bufs["max"]))
        if bufs["n"]:
            result["mean"] = float(np.sum(bufs["mean_n"]) / np.sum(bufs["n"]))
        global_stats[key] = result

    return global_stats


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Compute global normalization stats for OXE-Mix")
    parser.add_argument(
        "--data_root_dir",
        type=str,
        help="Root directory containing all datasets (used with --data_mix)",
    )
    parser.add_argument(
        "--data_mix",
        type=str,
        default="oxe_mix_full",
        help="Named data mixture from data_registry (default: oxe_mix_full)",
    )
    parser.add_argument(
        "--from_dataset_statistics",
        type=str,
        default=None,
        help="Path to an existing dataset_statistics.json (faster than re-loading datasets)",
    )
    parser.add_argument(
        "--min_loaded_fraction",
        type=float,
        default=0.5,
        help=(
            "Abort if the fraction of successfully loaded datasets is below this threshold "
            "(default: 0.5). Set to 0.0 to disable the check."
        ),
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path for global_norm_stats.json",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("OXE-Mix Global Normalization Stats Computation")
    print("=" * 60)

    if args.from_dataset_statistics:
        print("\nMode: from existing dataset_statistics.json")
        print(f"  Input:  {args.from_dataset_statistics}")
        global_stats = compute_global_stats_from_statistics_json(args.from_dataset_statistics)
    elif args.data_root_dir:
        print("\nMode: scan datasets from disk")
        print(f"  data_root_dir: {args.data_root_dir}")
        print(f"  data_mix:      {args.data_mix}")
        global_stats = compute_global_stats_from_datasets(
            args.data_root_dir, args.data_mix, min_loaded_fraction=args.min_loaded_fraction
        )
    else:
        print("ERROR: provide --data_root_dir or --from_dataset_statistics")
        sys.exit(1)

    # Print summary
    print("\n[Global Stats Summary]")
    print(f"{'Key':<12}  {'q01':>10}  {'q99':>10}  {'range':>10}")
    print("-" * 48)
    for key in OXE_CONTINUOUS_ACTION_KEYS:
        if key not in global_stats:
            continue
        s = global_stats[key]
        rng = s["q99"] - s["q01"]
        print(f"action.{key:<6}  {s['q01']:>10.4f}  {s['q99']:>10.4f}  {rng:>10.4f}")

    # Save
    output = {
        "description": (
            "Global normalization statistics for OXE-Mix action keys. "
            "Computed following LAP: global_q01=min(all q01), global_q99=max(all q99). "
            "Use with lang_action_format=verbose for cross-dataset consistent action text."
        ),
        "data_mix": args.data_mix if args.data_root_dir else "from_statistics_json",
        "normalization_mode": "q99",
        "action": global_stats,
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
