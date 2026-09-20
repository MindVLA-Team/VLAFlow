"""Summarize metrics-only evaluation JSON files without creating plots."""

import argparse
import json
from pathlib import Path


def summarize(root):
    rows = []
    for path in sorted(root.rglob("*.json")):
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or not {"episodes", "successes", "success_rate"} <= data.keys():
            continue
        rows.append(data)
    if not rows:
        raise ValueError(f"No completed evaluation metrics under {root}")
    episodes = sum(row["episodes"] for row in rows)
    successes = sum(row["successes"] for row in rows)
    if episodes <= 0:
        raise ValueError("Evaluation metrics contain no episodes")
    return {
        "cases": len(rows),
        "episodes": episodes,
        "successes": successes,
        "pooled_success_rate": successes / episodes,
        "mean_case_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.root), indent=2))
