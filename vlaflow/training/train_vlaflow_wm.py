# Copyright (c) 2026 Li Auto Inc. (VLAFlow modifications)
# VLAFlow modifications are licensed under Apache-2.0; see LICENSE.
#
# Original upstream notice (terms retained in LICENSES/LicenseRef-StarVLA.txt):
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
vlaflow World-Model (MindWPI) Trainer

Based on train_vlaflow.py but with:
  - Multi-loss monitoring (action_fm_loss, latent_pred_loss, moe_aux_loss, total_loss)
  - Uses lerobot_wm_datasets for future-frame support
  - Independent from train_vlaflow.py to ensure zero risk to existing pipelines
"""

# Standard Library
import argparse
import json
import os
import time
from pathlib import Path
from typing import Tuple

# Third-Party Libraries
import torch
import torch.distributed as dist
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

# Local Modules
from vlaflow.dataloader.lerobot_wm_datasets import collate_fn, get_vla_dataset
from vlaflow.model.framework.base_framework import build_framework
from vlaflow.model.framework.share_tools import apply_config_compat
from vlaflow.training.trainer_utils.config_tracker import AccessTrackedConfig, wrap_config
from vlaflow.training.trainer_utils.trainer_tools import TrainerUtils, build_param_lr_groups, normalize_dotlist_args

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize logger
logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    """Create output directory and checkpoint directory."""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> DataLoader:
    """Prepare WM VLA training data using lerobot_wm_datasets."""
    logger.info(f"Creating WM VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_dataset_cfg = cfg.datasets.vla_data
    vla_dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)

    num_workers = int(vla_dataset_cfg.get("num_workers", 4))
    prefetch_factor = int(vla_dataset_cfg.get("prefetch_factor", 2)) if num_workers > 0 else None
    pin_memory = bool(vla_dataset_cfg.get("pin_memory", False))
    persistent_workers = bool(vla_dataset_cfg.get("persistent_workers", False)) and num_workers > 0
    print(
        f"[build_wm_dataloader] num_workers={num_workers}, "
        f"prefetch_factor={prefetch_factor}, pin_memory={pin_memory}, "
        f"persistent_workers={persistent_workers}"
    )

    dl_kwargs = dict(
        batch_size=cfg.datasets.vla_data.per_device_batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    if num_workers > 0:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    vla_train_dataloader = DataLoader(vla_dataset, **dl_kwargs)
    if dist.get_rank() == 0:
        vla_dataset.save_dataset_statistics(output_dir / "dataset_statistics.json")

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()
    return vla_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """Set optimizer and scheduler."""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for group in optimizer.param_groups:
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class WMVLATrainer(TrainerUtils):
    """World-Model VLA Trainer with separate action/latent loss monitoring."""

    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        self._save_initial_configs()
        self._init_checkpointing()
        self._adjust_lr_scheduler_for_resume()

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)
        self.print_trainable_parameters(self.model)

        self.model, self.optimizer, self.vla_train_dataloader = self.setup_distributed_training(
            self.accelerator,
            self.model,
            self.optimizer,
            self.vla_train_dataloader,
        )

    def _calculate_total_batch_size(self):
        """Calculate global batch size."""
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _save_initial_configs(self):
        """Save full config and training script at the very start of training."""
        if not self.accelerator.is_main_process:
            return

        output_dir = Path(self.config.output_dir)

        if isinstance(self.config, AccessTrackedConfig):
            full_cfg = self.config.unwrap()
        else:
            full_cfg = self.config
        full_yaml_path = output_dir / "config.full.yaml"
        OmegaConf.save(full_cfg, full_yaml_path, resolve=True)
        logger.info(f"Full config saved at {full_yaml_path}")

        if isinstance(self.config, AccessTrackedConfig):
            self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)

    def _init_checkpointing(self):
        """Initialize checkpoint directory and handle checkpoint loading."""
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)
        self.resume_from_checkpoint = pretrained_checkpoint

        if is_resume:
            resume_from_checkpoint, self.completed_steps = self._get_latest_checkpoint(self.checkpoint_dir)
            if resume_from_checkpoint:
                self.resume_from_checkpoint = resume_from_checkpoint
                self.model = self.load_pretrained_backbones(self.model, self.resume_from_checkpoint, reload_modules=None)
                logger.info(
                    f"Resuming training from checkpoint: {self.resume_from_checkpoint}, steps: {self.completed_steps}"
                )
                return

            logger.warning(f"No valid checkpoint found in {self.checkpoint_dir}. Starting training from scratch.")
            self.completed_steps = 0

        if pretrained_checkpoint:
            reload_modules = getattr(self.config.trainer, "reload_modules", None)
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
            self.completed_steps = 0
            self.resume_from_checkpoint = pretrained_checkpoint
            logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
        else:
            logger.info("No pretrained checkpoint provided. Starting training from scratch.")
            self.completed_steps = 0

    def _adjust_lr_scheduler_for_resume(self):
        """Adjust LR scheduler state after resuming from non-zero steps."""
        if self.completed_steps > 0:
            logger.info(f"Adjusting LR scheduler for resume from step {self.completed_steps}")
            for _ in range(self.completed_steps):
                self.lr_scheduler.step()
            logger.info(
                f"LR scheduler adjusted to step {self.completed_steps}, current LR: {self.lr_scheduler.get_last_lr()}"
            )

    def _save_checkpoint(self):
        """Save current training state."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")

            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, checkpoint_path + "_model.safetensors")
            elif save_format == "pt":
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`. Expected `pt` or `safetensors`.")

            summary_data = {"steps": self.completed_steps}
            with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                f.write(json.dumps(summary_data) + "\n")
            self.accelerator.print(f"Checkpoint saved at {checkpoint_path}")

            if isinstance(self.config, AccessTrackedConfig):
                output_dir = Path(self.config.output_dir)
                self.config.save_accessed_config(output_dir / "config.yaml", use_original_values=False)

        self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        """Record training metrics."""
        if (
            self.accelerator.sync_gradients
            and self.completed_steps % self.config.trainer.logging_frequency == 0
            and self.accelerator.is_main_process
        ):
            metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
            metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
            with open(Path(self.config.output_dir) / "metrics.jsonl", "a") as stream:
                stream.write(json.dumps({"step": self.completed_steps, **metrics}) + "\n")
            logger.info(f"Step {self.completed_steps}, Metrics: {metrics})")

    def _create_data_iterators(self):
        """Create data iterators."""
        self.vla_iter = iter(self.vla_train_dataloader)

    def _get_next_batch(self):
        """Get next batch (automatically handle data loop)."""
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        return batch_vla

    def train(self):
        """Execute training loop with multi-loss monitoring."""
        self._log_training_config()
        self._create_data_iterators()
        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla = self._get_next_batch()
            t_end_data = time.perf_counter()

            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla)
            t_end_model = time.perf_counter()

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1

            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if (
                self.accelerator.sync_gradients
                and self.completed_steps % self.config.trainer.save_interval == 0
                and self.completed_steps > 0
            ):
                self._save_checkpoint()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def _log_training_config(self):
        """Record training config."""
        if self.accelerator.is_main_process:
            logger.info("***** WM-VLA Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  Per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.accelerator.gradient_accumulation_steps}")
            logger.info(f"  Total batch size = {self.total_batch_size}")
            logger.info(f"  Future frame offset = {self.config.datasets.vla_data.get('future_frame_offset', 5)}")

    def _train_step(self, batch_vla):
        """Execute single training step with multi-loss monitoring."""
        with self.accelerator.accumulate(self.model):

            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                # action_loss is the total weighted loss (backward target)
                total_loss = output_dict["action_loss"]

            self.accelerator.backward(total_loss)

            if self.accelerator.sync_gradients and self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
            self.optimizer.zero_grad()

        # Multi-loss monitoring: log all loss components separately
        metrics = {
            "total_loss": total_loss.item(),
        }

        # action_fm_loss and latent_pred_loss are provided by MindWPI for monitoring
        if "action_fm_loss" in output_dict:
            metrics["action_fm_loss"] = output_dict["action_fm_loss"].item()
        if "latent_pred_loss" in output_dict:
            metrics["latent_pred_loss"] = output_dict["latent_pred_loss"].item()

        return metrics

    def _finalize_training(self):
        """Training end processing."""
        if self.accelerator.is_main_process:
            save_format = getattr(self.config.trainer, "save_format", "pt")
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            state_dict = self.accelerator.get_state_dict(self.model)
            if save_format == "safetensors":
                from safetensors.torch import save_file

                save_file(state_dict, os.path.join(final_checkpoint, "model.safetensors"))
            elif save_format == "pt":
                torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            else:
                raise ValueError(f"Unsupported save_format `{save_format}`.")
            self.accelerator.print(f"Final model saved at {final_checkpoint}")

        self.accelerator.wait_for_everyone()


def main():
    parser = argparse.ArgumentParser(description="vlaflow World-Model (MindWPI) Training")
    parser.add_argument("--config_yaml", type=str, required=True, help="Path to YAML config")
    args, cli_overrides = parser.parse_known_args()

    # Load YAML config
    cfg = OmegaConf.load(args.config_yaml)

    # Apply CLI overrides (dot-list format)
    if cli_overrides:
        cli_overrides = normalize_dotlist_args(cli_overrides)
        cli_cfg = OmegaConf.from_dotlist(cli_overrides)
        cfg = OmegaConf.merge(cfg, cli_cfg)

    # Apply compatibility transforms
    cfg = apply_config_compat(cfg)

    # Wrap config for access tracking
    cfg = wrap_config(cfg)

    # Setup directories
    output_dir = setup_directories(cfg)

    # Build model
    model = build_framework(cfg)

    # Prepare data
    vla_train_dataloader = prepare_data(cfg, accelerator, output_dir)

    # Setup optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, cfg)

    # Create trainer and run
    trainer = WMVLATrainer(cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator)
    trainer.prepare_training()
    trainer.train()


if __name__ == "__main__":
    main()
