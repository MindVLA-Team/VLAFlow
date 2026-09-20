import ctypes
import json
import os
from pathlib import Path

import numpy as np
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader

logger = get_logger(__name__)


def _worker_init_fn(worker_id: int) -> None:
    """Called in each DataLoader worker right after fork.

    Runs ``malloc_trim(0)`` so glibc releases COW'd parent-process arenas back
    to the OS — without this, mixture training with 800 sub-datasets and long
    iterations steadily grows worker RSS even when Python has dropped all its
    references, eventually triggering the kernel OOM killer (SIGKILL).
    """
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.malloc_trim(0)
    except Exception:
        pass


def save_dataset_statistics(dataset_statistics, run_dir):
    """Saves a `dataset_statistics.json` file."""
    out_path = run_dir / "dataset_statistics.json"
    with open(out_path, "w") as f_json:
        for _, stats in dataset_statistics.items():
            for k in stats["action"].keys():
                if isinstance(stats["action"][k], np.ndarray):
                    stats["action"][k] = stats["action"][k].tolist()
            if "proprio" in stats:
                for k in stats["proprio"].keys():
                    if isinstance(stats["proprio"][k], np.ndarray):
                        stats["proprio"][k] = stats["proprio"][k].tolist()
            if "num_trajectories" in stats:
                if isinstance(stats["num_trajectories"], np.ndarray):
                    stats["num_trajectories"] = stats["num_trajectories"].item()
            if "num_transitions" in stats:
                if isinstance(stats["num_transitions"], np.ndarray):
                    stats["num_transitions"] = stats["num_transitions"].item()
        json.dump(dataset_statistics, f_json, indent=2)
    logger.info(f"Saved dataset statistics file at path {out_path}")


def build_dataloader(
    cfg, dataset_py="lerobot_datasets_oxe"
):  # TODO now here only is get dataset, we need mv dataloader to here

    if dataset_py == "lerobot_datasets":
        from vlaflow.dataloader.lerobot_datasets import collate_fn, get_vla_dataset

        vla_dataset_cfg = cfg.datasets.vla_data

        vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

        # ── DataLoader knobs (all YAML-overridable via datasets.vla_data.*) ──
        # Defaults are conservative to keep per-worker RSS bounded for the
        # 800-entry OXE_Mix mixture where worker memory leaks are the main
        # cause of SIGKILL OOM kills mid-training.
        num_workers = int(vla_dataset_cfg.get("num_workers", 4))
        prefetch_factor = int(vla_dataset_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
        pin_memory = bool(vla_dataset_cfg.get("pin_memory", False))
        persistent_workers = bool(vla_dataset_cfg.get("persistent_workers", False)) and num_workers > 0
        print(
            f"[build_dataloader] num_workers={num_workers}, "
            f"prefetch_factor={prefetch_factor}, pin_memory={pin_memory}, "
            f"persistent_workers={persistent_workers}"
        )

        dl_kwargs = dict(
            batch_size=cfg.datasets.vla_data.per_device_batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        )
        if num_workers > 0:
            dl_kwargs["prefetch_factor"] = prefetch_factor

        vla_train_dataloader = DataLoader(vla_dataset, **dl_kwargs)
        if dist.get_rank() == 0:

            output_dir = Path(cfg.output_dir)
            vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")
        return vla_train_dataloader
    else:
        raise ValueError(f"Unsupported dataset module: {dataset_py}")
