#!/usr/bin/env python3
"""Convert OXE_AugE datasets from LeRobot v3.0 to v2.1 format with parallel processing.

For every dataset directory under ``OXE_AugE/``:
  - v3.0 → full conversion: parquet split + video extraction (ffmpeg -c:v libx264)
  - v2.1 / other → symlinked into the output directory via a generated shell script

Output root defaults to a sibling ``OXE_AugE_v2/`` of the input so that the
existing OXE_AugE data-config paths only need a one-line root change.

Parallelism
-----------
- Outer: ``--workers`` processes handle different datasets concurrently (ProcessPoolExecutor).
- Inner: ``--video_threads`` threads per dataset for concurrent ffmpeg episode extractions.

Usage::

    # Default paths, 8 parallel datasets, 4 ffmpeg threads each
    python scripts/convert_oxeauge_v30_to_v21.py

    # Custom parallelism
    python scripts/convert_oxeauge_v30_to_v21.py --workers 16 --video_threads 8

    # Convert a single dataset (selective conversion / resume)
    python scripts/convert_oxeauge_v30_to_v21.py --dataset austin_buds_dataset_augmented

    # Force re-conversion even if completion marker exists
    python scripts/convert_oxeauge_v30_to_v21.py --force
"""

from __future__ import annotations

import argparse
import json
import math
import shlex
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_CHUNK_SIZE = 1000

# v3.0 consolidated layout
_V3_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_V3_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_V3_EPISODES_GLOB = "meta/episodes/chunk-*/file-*.parquet"

# v2.1 per-episode layout
_V21_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
_V21_VIDEO_PATH = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
_V21_EPISODES_PATH = "meta/episodes.jsonl"
_V21_EPISODES_STATS_PATH = "meta/episodes_stats.jsonl"
_V21_TASKS_PATH = "meta/tasks.jsonl"

_LEGACY_STATS_KEYS = {"mean", "std", "min", "max", "count"}
_COMPLETION_MARKER = "[].conversion_completed.json"

# Default paths — adjust if data lives elsewhere
INPUT_ROOT = Path("data/OXE_AugE")
OUTPUT_ROOT = Path("data/OXE_AugE_v2")


# ---------------------------------------------------------------------------
# Small serialisation helpers
# ---------------------------------------------------------------------------


def _to_py(v: Any) -> Any:
    """Recursively convert numpy / pyarrow scalars to plain Python types."""
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_to_py(x) for x in v]
    if isinstance(v, dict):
        return {k: _to_py(w) for k, w in v.items()}
    return v


def _serialize(v: Any) -> Any:
    """Alias of _to_py — kept for clarity at call sites."""
    return _to_py(v)


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Unflatten slash-separated keys: {'a/b/c': v} → {'a': {'b': {'c': v}}}."""
    out: dict[str, Any] = {}
    for key, val in flat.items():
        parts = key.split("/")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


# ---------------------------------------------------------------------------
# Info.json helpers
# ---------------------------------------------------------------------------


def _load_info(root: Path) -> dict[str, Any]:
    return json.loads((root / "meta" / "info.json").read_text())


def _write_info(info: dict[str, Any], root: Path) -> None:
    out = root / "meta" / "info.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(info, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Step 0 – Load episode records from v3.0 consolidated metadata
# ---------------------------------------------------------------------------


def _load_episode_records(root: Path) -> list[dict[str, Any]]:
    pq_paths = sorted(root.glob(_V3_EPISODES_GLOB))
    if not pq_paths:
        raise FileNotFoundError(f"No episode parquet files found under {root / 'meta/episodes'}")
    records: list[dict[str, Any]] = []
    for p in pq_paths:
        records.extend(pq.read_table(p).to_pylist())
    records.sort(key=lambda r: int(r["episode_index"]))
    return records


def _get_video_keys(root: Path) -> list[str]:
    info = _load_info(root)
    return [k for k, ft in info.get("features", {}).items() if ft.get("dtype") == "video"]


# ---------------------------------------------------------------------------
# Step 1 – Convert info.json  (v3.0 → v2.1 schema)
# ---------------------------------------------------------------------------


def _convert_info(
    src: Path,
    dst: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
) -> None:
    info = _load_info(src)
    chunks_size = info.get("chunks_size", _DEFAULT_CHUNK_SIZE)
    total_episodes = len(episode_records)

    info["codebase_version"] = "v2.1"
    info["data_path"] = _V21_DATA_PATH
    info["video_path"] = _V21_VIDEO_PATH if video_keys else None

    # Remove v3-only sizing hints
    info.pop("data_files_size_in_mb", None)
    info.pop("video_files_size_in_mb", None)

    # Non-video features must not carry a top-level fps field in v2.1
    for _key, ft in info.get("features", {}).items():
        if ft.get("dtype") != "video":
            ft.pop("fps", None)

    info["total_episodes"] = total_episodes
    info["total_chunks"] = math.ceil(total_episodes / chunks_size) if total_episodes else 0
    info["total_videos"] = total_episodes * len(video_keys)

    _write_info(info, dst)


# ---------------------------------------------------------------------------
# Step 2 – Convert tasks (parquet → JSONL)
# ---------------------------------------------------------------------------


def _convert_tasks(src: Path, dst: Path) -> None:
    tasks_pq = src / "meta" / "tasks.parquet"
    if not tasks_pq.exists():
        return

    rows = pq.read_table(tasks_pq).to_pylist()
    rows.sort(key=lambda r: int(r.get("task_index", 0)))

    out = dst / _V21_TASKS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            # The task string may live under different column names
            task_val = row.get("task") or row.get("task_description") or row.get("language_instruction", "")
            f.write(
                json.dumps(
                    {"task_index": int(row["task_index"]), "task": _to_py(task_val)},
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Step 3 – Split consolidated parquet → per-episode parquets
# ---------------------------------------------------------------------------


def _convert_data(
    src: Path,
    dst: Path,
    episode_records: list[dict[str, Any]],
) -> None:
    # Group records by the source parquet file they reside in
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for rec in episode_records:
        grouped[(int(rec["data/chunk_index"]), int(rec["data/file_index"]))].append(rec)

    for (chunk_idx, file_idx), recs in grouped.items():
        src_pq = src / _V3_DATA_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        if not src_pq.exists():
            raise FileNotFoundError(f"Source parquet not found: {src_pq}")

        table = pq.read_table(src_pq)
        recs_sorted = sorted(recs, key=lambda r: int(r["dataset_from_index"]))
        # dataset_from_index is absolute within the whole dataset; subtract the
        # file-local offset so we can use table.slice()
        file_offset = int(recs_sorted[0]["dataset_from_index"])

        for rec in recs_sorted:
            ep_idx = int(rec["episode_index"])
            start = int(rec["dataset_from_index"]) - file_offset
            stop = int(rec["dataset_to_index"]) - file_offset
            length = stop - start
            if length <= 0:
                raise ValueError(f"Invalid episode slice ep={ep_idx}: start={start} stop={stop}")

            ep_table = table.slice(start, length)
            dest_chunk = ep_idx // _DEFAULT_CHUNK_SIZE
            dest_pq = dst / _V21_DATA_PATH.format(episode_chunk=dest_chunk, episode_index=ep_idx)
            dest_pq.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(ep_table, dest_pq)


# ---------------------------------------------------------------------------
# Step 4 – Extract per-episode mp4 clips from concatenated video files
# ---------------------------------------------------------------------------


def _get_fps_ffprobe(video_path: Path) -> float:
    """Read video FPS via ffprobe; fall back to 30.0 on any error."""
    try:
        res = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=r_frame_rate",
                "-of",
                "csv=p=0",
                str(video_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        frac = res.stdout.strip()
        if "/" in frac:
            a, b = frac.split("/")
            fps = float(a) / float(b)
        else:
            fps = float(frac)
        return fps if fps > 0 else 30.0
    except Exception:
        return 30.0


def _extract_segment(
    src: Path,
    dst: Path,
    start: float,
    end: float,
    fps: float,
) -> None:
    """Extract [start, end) seconds from src into dst using ffmpeg (H.264 output)."""
    start_frame = int(round(start * fps))
    end_frame = int(round(end * fps))
    num_frames = max(1, end_frame - start_frame)

    dst.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{start:.6f}",  # fast keyframe seek
        "-i",
        str(src),
        "-frames:v",
        str(num_frames),  # exact frame count
        "-vsync",
        "0",  # preserve original frame timing
        "-c:v",
        "libx264",
        "-preset",
        "fast",
        "-crf",
        "23",
        "-y",
        str(dst),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg failed ({src.name} -> {dst.name}): {res.stderr[:300]}")


def _convert_videos(
    src: Path,
    dst: Path,
    episode_records: list[dict[str, Any]],
    video_keys: list[str],
    video_threads: int,
) -> None:
    if not video_keys:
        return

    # Build a flat list of all (src_vid, dst_vid, start, end, fps) tasks across
    # every video key and every source video file so we can batch-submit them.
    all_tasks: list[tuple[Path, Path, float, float, float]] = []

    for vk in video_keys:
        chunk_col = f"videos/{vk}/chunk_index"
        file_col = f"videos/{vk}/file_index"
        ts_from = f"videos/{vk}/from_timestamp"
        ts_to = f"videos/{vk}/to_timestamp"

        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for rec in episode_records:
            if rec.get(chunk_col) is None:
                continue
            grouped[(int(rec[chunk_col]), int(rec[file_col]))].append(rec)

        for (chunk_idx, file_idx), recs in grouped.items():
            src_vid = src / _V3_VIDEO_PATH.format(video_key=vk, chunk_index=chunk_idx, file_index=file_idx)
            if not src_vid.exists():
                raise FileNotFoundError(f"Source video not found: {src_vid}")

            fps = _get_fps_ffprobe(src_vid)
            recs_sorted = sorted(recs, key=lambda r: float(r[ts_from]))
            for rec in recs_sorted:
                ep_idx = int(rec["episode_index"])
                t0 = float(rec[ts_from])
                t1 = float(rec[ts_to])
                dest_chunk = ep_idx // _DEFAULT_CHUNK_SIZE
                dst_vid = dst / _V21_VIDEO_PATH.format(episode_chunk=dest_chunk, video_key=vk, episode_index=ep_idx)
                all_tasks.append((src_vid, dst_vid, t0, t1, fps))

    if video_threads <= 1:
        for src_v, dst_v, t0, t1, fps in all_tasks:
            _extract_segment(src_v, dst_v, t0, t1, fps)
    else:
        with ThreadPoolExecutor(max_workers=video_threads) as pool:
            futures = {pool.submit(_extract_segment, sv, dv, t0, t1, fps): (sv, dv) for sv, dv, t0, t1, fps in all_tasks}
            for fut in as_completed(futures):
                sv, dv = futures[fut]
                exc = fut.exception()
                if exc:
                    raise RuntimeError(f"Video extraction failed {sv.name} → {dv.name}: {exc}")


# ---------------------------------------------------------------------------
# Step 5 – Reconstruct legacy episode metadata JSONL files
# ---------------------------------------------------------------------------


def _convert_episodes_meta(
    dst: Path,
    episode_records: list[dict[str, Any]],
) -> None:
    episodes_path = dst / _V21_EPISODES_PATH
    stats_path = dst / _V21_EPISODES_STATS_PATH
    episodes_path.parent.mkdir(parents=True, exist_ok=True)

    with (
        episodes_path.open("w", encoding="utf-8") as ef,
        stats_path.open("w", encoding="utf-8") as sf,
    ):
        for rec in sorted(episode_records, key=lambda r: int(r["episode_index"])):
            # episodes.jsonl — drop v3-specific synthetic keys
            ep_dict = {
                k: _to_py(v)
                for k, v in rec.items()
                if not k.startswith(("data/", "videos/", "stats/", "meta/"))
                and k not in {"dataset_from_index", "dataset_to_index"}
            }
            ef.write(json.dumps(ep_dict, ensure_ascii=False) + "\n")

            # episodes_stats.jsonl — unflatten stats/* and keep only legacy keys
            flat_stats = {k: rec[k] for k in rec if k.startswith("stats/")}
            nested = _unflatten(flat_stats).get("stats", {})
            filtered: dict[str, Any] = {
                feat: {sk: sv for sk, sv in vals.items() if sk in _LEGACY_STATS_KEYS}
                for feat, vals in nested.items()
                if isinstance(vals, dict)
            }
            sf.write(
                json.dumps(
                    {
                        "episode_index": int(rec["episode_index"]),
                        "stats": _serialize(filtered),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Step 6 – Copy ancillary meta files
# ---------------------------------------------------------------------------


def _copy_ancillary(src: Path, dst: Path) -> None:
    """Copy modality.json and stats.json (if present).
    stats_gr00t.json is intentionally excluded — it will be regenerated by the
    training pipeline with the correct delta-action schema."""
    for fname in ("modality.json", "stats.json"):
        p = src / "meta" / fname
        if p.exists():
            out = dst / "meta" / fname
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, out)


# ---------------------------------------------------------------------------
# Per-dataset entry point (runs inside worker process)
# ---------------------------------------------------------------------------


def convert_one_dataset(
    src_dir: Path,
    dst_dir: Path,
    video_threads: int,
    force: bool,
) -> tuple[str, str, str]:
    """Convert one dataset from v3.0 → v2.1.

    Returns (dataset_name, status, detail) where status ∈ {"ok","skipped","error"}.
    """
    name = src_dir.name
    try:
        info_path = src_dir / "meta" / "info.json"
        if not info_path.exists():
            return (name, "skipped", "no meta/info.json")

        ver = json.loads(info_path.read_text()).get("codebase_version", "?")
        if ver != "v3.0":
            return (name, "skipped", f"version={ver}")

        marker = dst_dir / _COMPLETION_MARKER
        if not force and marker.exists():
            return (name, "skipped", "already converted (marker found)")

        # Remove any leftover partial output
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        episode_records = _load_episode_records(src_dir)
        video_keys = _get_video_keys(src_dir)

        _convert_info(src_dir, dst_dir, episode_records, video_keys)
        _convert_tasks(src_dir, dst_dir)
        _convert_data(src_dir, dst_dir, episode_records)
        _convert_videos(src_dir, dst_dir, episode_records, video_keys, video_threads)
        _convert_episodes_meta(dst_dir, episode_records)
        _copy_ancillary(src_dir, dst_dir)

        # Reconcile info.json totals from actual on-disk episode files
        actual_eps = len(list(dst_dir.glob("data/chunk-*/episode_*.parquet")))
        info = _load_info(dst_dir)
        chunks_size = info.get("chunks_size", _DEFAULT_CHUNK_SIZE)
        info["total_episodes"] = actual_eps
        info["total_chunks"] = math.ceil(actual_eps / chunks_size) if actual_eps else 0
        info["total_videos"] = actual_eps * len(video_keys)
        _write_info(info, dst_dir)

        marker.write_text(
            json.dumps(
                {
                    "conversion_completed": True,
                    "completed_at": datetime.now().isoformat(),
                    "input_path": str(src_dir),
                    "output_path": str(dst_dir),
                    "episodes": len(episode_records),
                    "video_keys": video_keys,
                },
                indent=2,
            )
        )
        return (name, "ok", f"{len(episode_records)} eps, {len(video_keys)} cams")

    except Exception as exc:
        tb = traceback.format_exc()
        return (name, "error", f"{exc}\n{tb}")


# ---------------------------------------------------------------------------
# Symlink script generator (for non-v3.0 datasets)
# ---------------------------------------------------------------------------


def _write_symlink_script(
    non_v30_dirs: list[Path],
    output_root: Path,
    script_path: Path,
) -> None:
    lines = [
        "#!/bin/bash",
        "# Auto-generated by convert_oxeauge_v30_to_v21.py",
        "# Symlinks non-v3.0 OXE_AugE datasets into OXE_AugE_v2",
        f"OUTPUT_ROOT={shlex.quote(str(output_root.resolve()))}",
        "",
        'mkdir -p "$OUTPUT_ROOT"',
        "",
    ]
    for src in sorted(non_v30_dirs):
        dst = output_root / src.name
        lines.append(shlex.join(["ln", "-sfn", "--", str(src.resolve()), str(dst.absolute())]))
    lines += ["", "echo 'Symlink creation complete.'", ""]
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("\n".join(lines))
    script_path.chmod(0o755)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input",
        type=Path,
        default=INPUT_ROOT,
        help=f"Source OXE_AugE directory (default: {INPUT_ROOT})",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Destination OXE_AugE_v2 directory (default: {OUTPUT_ROOT})",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of datasets to convert in parallel (default: 8)",
    )
    ap.add_argument(
        "--video_threads",
        type=int,
        default=4,
        help="ffmpeg threads per dataset for video extraction (default: 4)",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Convert only this dataset name (selective conversion / resume mode)",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-convert even if completion marker exists",
    )
    args = ap.parse_args()

    if not args.input.is_dir():
        sys.exit(f"Input directory not found: {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Enumerate datasets ────────────────────────────────────────────────
    if args.dataset:
        all_dirs = [args.input / args.dataset]
    else:
        all_dirs = sorted(d for d in args.input.iterdir() if d.is_dir())

    v30_dirs: list[Path] = []
    non_v30_dirs: list[Path] = []

    for d in all_dirs:
        info_p = d / "meta" / "info.json"
        if not info_p.exists():
            non_v30_dirs.append(d)
            continue
        ver = json.loads(info_p.read_text()).get("codebase_version", "?")
        if ver == "v3.0":
            v30_dirs.append(d)
        else:
            non_v30_dirs.append(d)

    print("OXE_AugE scan complete:")
    print(f"  v3.0 to convert : {len(v30_dirs)}")
    print(f"  non-v3.0 (symlink): {len(non_v30_dirs)}")
    print(f"  Output root      : {args.output}")
    print(f"  Workers          : {args.workers}")
    print(f"  Video threads/ds : {args.video_threads}")
    print()

    # ── Write symlink script (non-v3.0 datasets) ─────────────────────────
    if not args.dataset and non_v30_dirs:
        script_path = args.output / "run_link_existing.sh"
        _write_symlink_script(non_v30_dirs, args.output, script_path)
        print(f"Symlink script written → {script_path}")
        print("Run it once to link v2.1 / inpainting datasets:")
        print(f"  bash {script_path}")
        print()

    if not v30_dirs:
        print("No v3.0 datasets to convert. Done.")
        return 0

    # ── Parallel conversion ───────────────────────────────────────────────
    total = len(v30_dirs)
    ok_count = 0
    err_count = 0
    skip_count = 0
    errors: list[tuple[str, str]] = []
    done = 0

    print(f"Starting conversion of {total} v3.0 datasets…")
    print("-" * 88)

    with ProcessPoolExecutor(max_workers=min(args.workers, total)) as executor:
        future_map = {
            executor.submit(
                convert_one_dataset,
                src,
                args.output / src.name,
                args.video_threads,
                args.force,
            ): src.name
            for src in v30_dirs
        }

        for fut in as_completed(future_map):
            name, status, detail = fut.result()
            done += 1

            if status == "ok":
                ok_count += 1
            elif status == "skipped":
                skip_count += 1
            else:
                err_count += 1
                errors.append((name, detail))

            # Progress bar
            filled = done * 40 // total
            bar = "#" * filled + "." * (40 - filled)
            # Truncate detail for display
            disp = detail[:72] if len(detail) <= 72 else detail[:69] + "…"
            tag = {"ok": "✓", "skipped": "~", "error": "✗"}.get(status, "?")
            print(
                f"  [{done:4d}/{total}] [{bar}] {tag} {name:<55s} {disp}",
                flush=True,
            )

    # ── Summary ───────────────────────────────────────────────────────────
    print("=" * 88)
    print("Conversion finished:")
    print(f"  ✓ OK      : {ok_count}")
    print(f"  ~ Skipped : {skip_count}")
    print(f"  ✗ Errors  : {err_count}")

    if errors:
        print("\nFailed datasets:")
        for name, detail in errors:
            # Print first 5 lines of traceback only
            lines = detail.strip().splitlines()
            print(f"\n  [{name}]")
            for ln in lines[:8]:
                print(f"    {ln}")
            if len(lines) > 8:
                print(f"    … ({len(lines) - 8} more lines)")

    if non_v30_dirs and not args.dataset:
        script_path = args.output / "run_link_existing.sh"
        print(f"\nReminder: run  bash {script_path}  to symlink non-v3.0 datasets.")

    return 0 if err_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
