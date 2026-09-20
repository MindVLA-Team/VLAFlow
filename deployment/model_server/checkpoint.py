"""Read checkpoint metadata without importing the training stack."""

import json
from pathlib import Path

import yaml


def read_mode_config(checkpoint):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
    run_dir = checkpoint.parent.parent
    config_path = run_dir / "config.full.yaml"
    if not config_path.is_file():
        config_path = run_dir / "config.yaml"
    with config_path.open() as stream:
        config = yaml.safe_load(stream)
    with (run_dir / "dataset_statistics.json").open() as stream:
        statistics = json.load(stream)
    return config, statistics
