# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Fangjing Wang/ SUST University] in [2025].
# Modification: [return raw data and suport multi-dataset mixture].
# Modified by [Jinhui YE/ HKUST University] in [2025].
# Modification: [suport topdowm processing, suport param from config].

from pathlib import Path

from omegaconf import OmegaConf

from vlaflow.dataloader.gr00t_lerobot.datasets import LeRobotMixtureDataset, LeRobotSingleDataset
from vlaflow.dataloader.gr00t_lerobot.registry import (
    DATASET_NAMED_MIXTURES,
    ROBOT_TYPE_CONFIG_MAP,
    ROBOT_TYPE_TO_EMBODIMENT_TAG,
    EmbodimentTag,
    validate_mixture_sources,
)


def collate_fn(batch):
    return batch


def make_LeRobotSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object.

    :param data_root_dir: The root directory of the dataset.
    :param data_name: The name of the dataset.
    :param robot_type: The robot type config to use.
    :param crop_obs_camera: Whether to crop the observation camera images.
    :return: A LeRobotSingleDataset object.
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

    # Merge global data_cfg with per-DataConfig data_cfg.
    # The per-DataConfig dict takes precedence, allowing each robot type to override
    # dataset-specific fields such as lerobot_version, action_mode, action_mode_apply_keys,
    # and action_mode_state_map without affecting shared fields (video_backend, etc.).
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
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,  # decord is more efficiency | torchvision_av for video.av1
        delete_pause_frame=delete_pause_frame,
        data_cfg=effective_data_cfg,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> LeRobotMixtureDataset:
    """
    Get a LeRobotMixtureDataset object.

    Sampling-balance flags can be set either via the function arguments or via
    YAML keys (``data_cfg.balance_dataset_weights`` and
    ``data_cfg.balance_trajectory_weights``). When both are provided, the YAML
    value takes precedence so users can flip between equal-per-dataset sampling
    (default, all flags off) and step-uniform proportional sampling
    (``balance_dataset_weights=True`` and ``balance_trajectory_weights=True``)
    without editing code.
    """
    if "balance_dataset_weights" in data_cfg:
        balance_dataset_weights = bool(data_cfg.get("balance_dataset_weights"))
    if "balance_trajectory_weights" in data_cfg:
        balance_trajectory_weights = bool(data_cfg.get("balance_trajectory_weights"))
    print(
        f"[get_vla_dataset] balance_dataset_weights={balance_dataset_weights}, "
        f"balance_trajectory_weights={balance_trajectory_weights}"
    )
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
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
            ds = make_LeRobotSingleDataset(
                Path(data_root_dir),
                d_name,
                robot_type,
                delete_pause_frame=delete_pause_frame,
                data_cfg=data_cfg,
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
        print(f"[get_vla_dataset] Skipped {skipped} dataset(s) due to missing modalities.")

    return LeRobotMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        **kwargs,
    )
