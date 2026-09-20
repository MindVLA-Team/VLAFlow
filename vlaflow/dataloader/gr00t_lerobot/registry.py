"""
Centralized registry that auto-discovers benchmark-specific data configs from
``examples/*/train_files/data_registry/`` and merges them with the base
registries defined in this package.

Three registries are maintained:

* ``DATASET_NAMED_MIXTURES``       – mixture_name → [(dataset, weight, robot_type)]
* ``ROBOT_TYPE_CONFIG_MAP``       – robot_type → DataConfig instance
* ``ROBOT_TYPE_TO_EMBODIMENT_TAG`` – robot_type → EmbodimentTag


Usage::

    from vlaflow.dataloader.gr00t_lerobot.registry import (
        ROBOT_TYPE_CONFIG_MAP,
        ROBOT_TYPE_TO_EMBODIMENT_TAG,
        DATASET_NAMED_MIXTURES,
    )
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from pathlib import Path

# Base registries (kept as fallback / seed values)
_BASE_CONFIG_MAP = {}
from vlaflow.dataloader.gr00t_lerobot.embodiment_tags import (
    ROBOT_TYPE_TO_EMBODIMENT_TAG as _BASE_EMBODIMENT_MAP,
)
from vlaflow.dataloader.gr00t_lerobot.embodiment_tags import (
    EmbodimentTag,  # noqa: F401  – re-export for convenience
)

_BASE_MIXTURES = {}

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mutable copies – will be extended by discovered modules
# ---------------------------------------------------------------------------
ROBOT_TYPE_CONFIG_MAP: dict = dict(_BASE_CONFIG_MAP)
ROBOT_TYPE_TO_EMBODIMENT_TAG: dict = dict(_BASE_EMBODIMENT_MAP)
DATASET_NAMED_MIXTURES: dict = dict(_BASE_MIXTURES)

# ---------------------------------------------------------------------------
# Discovery logic
# ---------------------------------------------------------------------------
_REGISTRY_DIR_NAME = "data_registry"
_DISCOVERED = False


def _find_registry_dirs() -> list[Path]:
    """Return all ``examples/*/train_files/data_registry/`` directories."""
    # Walk up from this file to the repo root
    # registry.py is at vlaflow/dataloader/gr00t_lerobot/registry.py
    #   parents: [0]=gr00t_lerobot, [1]=dataloader, [2]=vlaflow(pkg), [3]=repo root
    repo_root = Path(__file__).resolve().parents[3]
    examples_dir = repo_root / "examples"
    if not examples_dir.is_dir():
        return []
    dirs: list[Path] = []
    for bench_dir in sorted(examples_dir.iterdir()):
        registry_dir = bench_dir / "train_files" / _REGISTRY_DIR_NAME
        if registry_dir.is_dir():
            dirs.append(registry_dir)
    return dirs


def _load_module_from_path(module_name: str, file_path: Path):
    """Import a Python file as a module with the given name."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        return None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


def discover_and_merge() -> None:
    """Scan ``examples/*/train_files/data_registry/`` and merge into global registries."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True

    for registry_dir in _find_registry_dirs():
        bench_name = registry_dir.parents[1].name  # examples/<BenchName>/train_files/data_registry
        prefix = f"_data_registry_{bench_name}"

        # --- data_config.py (may contain all three registries) ---
        cfg_file = registry_dir / "data_config.py"
        if cfg_file.is_file():
            mod = _load_module_from_path(f"{prefix}.data_config", cfg_file)
            if mod:
                if hasattr(mod, "ROBOT_TYPE_CONFIG_MAP"):
                    ROBOT_TYPE_CONFIG_MAP.update(mod.ROBOT_TYPE_CONFIG_MAP)
                    logger.info(
                        f"[registry] Loaded data_config from {bench_name}: {list(mod.ROBOT_TYPE_CONFIG_MAP.keys())}"
                    )
                if hasattr(mod, "ROBOT_TYPE_TO_EMBODIMENT_TAG"):
                    ROBOT_TYPE_TO_EMBODIMENT_TAG.update(mod.ROBOT_TYPE_TO_EMBODIMENT_TAG)
                    logger.info(
                        f"[registry] Loaded embodiment_tags from {bench_name} (data_config): {list(mod.ROBOT_TYPE_TO_EMBODIMENT_TAG.keys())}"
                    )
                if hasattr(mod, "DATASET_NAMED_MIXTURES"):
                    DATASET_NAMED_MIXTURES.update(mod.DATASET_NAMED_MIXTURES)
                    logger.info(
                        f"[registry] Loaded mixtures from {bench_name} (data_config): {list(mod.DATASET_NAMED_MIXTURES.keys())}"
                    )


# Run discovery on first import
discover_and_merge()

# Public recipes expose only the retained pretraining and downstream mixtures.
_PUBLIC_MIXTURES = {
    "oxe_mix_full",
    "oxe_mix_full_wm",
    "oxe_original",
    "oxe_original_wm",
    "libero_all_14d",
    "libero_all_14d_wm",
    "bridge_14d",
    "bridge_14d_wm",
    "rt1_14d",
    "rt1_14d_wm",
}
DATASET_NAMED_MIXTURES = {key: value for key, value in DATASET_NAMED_MIXTURES.items() if key in _PUBLIC_MIXTURES}


def validate_mixture_sources(data_mix: str, mixture_spec: list) -> None:
    """Reject a full OXEMix recipe if a source contributed no datasets."""
    if not mixture_spec:
        raise ValueError(f"Data mixture {data_mix!r} is empty")
    if data_mix in {"oxe_mix_full", "oxe_mix_full_wm"}:
        present = {name.split("/", 1)[0] for name, _weight, _robot in mixture_spec}
        required = {"openx-embodiment-lerobot", "OXE_AugE_v2", "RoboCOIN_SUB"}
        missing = required - present
        if missing:
            raise ValueError(f"Full OXEMix requires datasets from all sources; empty sources: {sorted(missing)}")
