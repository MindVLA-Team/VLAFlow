# Copyright (c) 2026 Li Auto Inc. (VLAFlow modifications)
# VLAFlow modifications are licensed under Apache-2.0; see LICENSE.
#
# Original upstream notice (terms retained in LICENSES/LicenseRef-StarVLA.txt):
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
Top-level dataset factory for World Model (MindWPI) training.

Mirrors lerobot_datasets.py but uses LeRobotWMSingleDataset which provides
present/future frames and per-sample compute_action_loss flags.

Usage in YAML config:
  datasets:
    vla_data:
      dataset_py: lerobot_wm_datasets
      future_frame_offset: 5
      data_mix: oxe_mix_wm
"""

from pathlib import Path

from omegaconf import OmegaConf

from vlaflow.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    EmbodimentTag,
    validate_mixture_sources,
)
from vlaflow.dataloader.gr00t_lerobot.wm_datasets import LeRobotWMMixtureDataset, LeRobotWMSingleDataset


def collate_fn(batch):
    return batch


def make_LeRobotWMSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
    future_frame_offset: int = 5,
    num_future_frames: int = 1,
) -> LeRobotWMSingleDataset:
    """
    Make a LeRobotWMSingleDataset object for world-model training.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param delete_pause_frame: Whether to delete pause frames.
    :param data_cfg: Global data configuration from YAML.
    :param future_frame_offset: Stride (in steps) between consecutive future frames.
    :param num_future_frames: Number of future frames to supply (offsets s, 2s, ..., N*s).
    :return: A LeRobotWMSingleDataset object.
    """
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name

    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(
            f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default"
        )
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]

    # Merge global data_cfg with per-DataConfig data_cfg
    per_robot_cfg = getattr(data_config, "data_cfg", None)
    if per_robot_cfg is not None or data_cfg is not None:
        if data_cfg is not None:
            try:
                merged = dict(OmegaConf.to_container(data_cfg, resolve=True))
            except Exception:
                merged = dict(data_cfg)
        else:
            merged = {}
        if per_robot_cfg is not None:
            merged.update(per_robot_cfg)
        effective_data_cfg = merged
    else:
        effective_data_cfg = None

    video_backend = effective_data_cfg.get("video_backend", "decord") if effective_data_cfg else "torchvision_av"

    # WM-specific: extract per-DataConfig flags
    compute_action_loss = getattr(data_config, "compute_action_loss", True)
    latent_view_key = getattr(data_config, "latent_view_key", None)

    return LeRobotWMSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        delete_pause_frame=delete_pause_frame,
        data_cfg=effective_data_cfg,
        # WM-specific params
        future_frame_offset=future_frame_offset,
        num_future_frames=num_future_frames,
        latent_view_key=latent_view_key,
        compute_action_loss=compute_action_loss,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotWMMixtureDataset:
    """
    Get a LeRobotWMMixtureDataset for world-model training.

    Same interface as lerobot_datasets.get_vla_dataset but uses WM dataset classes
    and passes future_frame_offset through.
    """
    if "balance_dataset_weights" in data_cfg:
        balance_dataset_weights = bool(data_cfg.get("balance_dataset_weights"))
    if "balance_trajectory_weights" in data_cfg:
        balance_trajectory_weights = bool(data_cfg.get("balance_trajectory_weights"))
    print(
        f"[get_wm_vla_dataset] balance_dataset_weights={balance_dataset_weights}, "
        f"balance_trajectory_weights={balance_trajectory_weights}"
    )

    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    future_frame_offset = int(data_cfg.get("future_frame_offset", 5))
    num_future_frames = int(data_cfg.get("num_future_frames", 1))

    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    validate_mixture_sources(data_mix, mixture_spec)
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))

    dataset_mixture = []
    skipped = 0
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        try:
            ds = make_LeRobotWMSingleDataset(
                Path(data_root_dir),
                d_name,
                robot_type,
                delete_pause_frame=delete_pause_frame,
                data_cfg=data_cfg,
                future_frame_offset=future_frame_offset,
                num_future_frames=num_future_frames,
            )
            dataset_mixture.append((ds, d_weight))
        except (ValueError, KeyError, AssertionError, FileNotFoundError) as e:
            if data_mix in {"oxe_mix_full", "oxe_mix_full_wm"}:
                raise ValueError(f"Full OXEMix could not load dataset {d_name!r}: {e}") from e
            import warnings

            warnings.warn(
                f"Skipping dataset '{d_name}' (robot_type='{robot_type}'): {e}",
                stacklevel=2,
            )
            skipped += 1
    if skipped:
        print(f"[get_wm_vla_dataset] Skipped {skipped} dataset(s) due to missing modalities.")

    return LeRobotWMMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
