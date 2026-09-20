# Copyright (c) 2026 Li Auto Inc. (VLAFlow modifications)
# VLAFlow modifications are licensed under Apache-2.0; see LICENSE.
#
# Original upstream notice (terms retained in LICENSES/LicenseRef-StarVLA.txt):
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
World Model (MindWPI) Dataset — extends LeRobotSingleDataset with future-frame support.

Each sample additionally provides:
  - present_image: List[PIL.Image] — current-frame image for latent extraction
  - future_image: List[PIL.Image] — future-frame image for latent supervision
  - compute_action_loss: bool — whether this sample contributes to action loss
"""

from pathlib import Path

import numpy as np
from PIL import Image

from vlaflow.dataloader.gr00t_lerobot.datasets import (
    LeRobotMixtureDataset,
    LeRobotSingleDataset,
    ModalityConfig,
)
from vlaflow.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag
from vlaflow.dataloader.gr00t_lerobot.transform import ComposedModalityTransform
from vlaflow.dataloader.gr00t_lerobot.video import get_frames_by_timestamps


class LeRobotWMSingleDataset(LeRobotSingleDataset):
    """Dataset that additionally provides present/future frames for latent world prediction.

    Extends LeRobotSingleDataset by:
      1. Fetching a raw PIL image at the current timestep (present_image)
      2. Fetching a raw PIL image at a configurable future offset (future_image)
      3. Adding a per-sample `compute_action_loss` flag

    The latent extractor (VJEPA-2/DINOv3) has its own image processor, so we provide
    raw PIL images (not the augmented/resized ones used for VLM input).
    """

    def __init__(
        self,
        dataset_path: Path | str,
        modality_configs: dict[str, ModalityConfig],
        embodiment_tag: str | EmbodimentTag,
        video_backend: str = "decord",
        video_backend_kwargs: dict | None = None,
        transforms: ComposedModalityTransform | None = None,
        delete_pause_frame: bool = False,
        data_cfg=None,
        # WM-specific params
        future_frame_offset: int = 5,
        num_future_frames: int = 1,
        latent_view_key: str | None = None,
        compute_action_loss: bool = True,
        **kwargs,
    ):
        super().__init__(
            dataset_path=dataset_path,
            modality_configs=modality_configs,
            embodiment_tag=embodiment_tag,
            video_backend=video_backend,
            video_backend_kwargs=video_backend_kwargs,
            transforms=transforms,
            delete_pause_frame=delete_pause_frame,
            data_cfg=data_cfg,
            **kwargs,
        )
        self.future_frame_offset = future_frame_offset
        self.num_future_frames = max(1, int(num_future_frames))
        self.latent_view_key = latent_view_key
        self.compute_action_loss = compute_action_loss

    def _default_latent_view_key(self) -> str:
        """Pick the first non-wrist primary video key as the default latent view."""
        video_keys = self.modality_keys.get("video", [])
        for key in video_keys:
            if "wrist" not in key.lower():
                return key
        # Fallback to the first video key if all are wrist views
        return video_keys[0] if video_keys else None

    def _get_latent_view_key(self) -> str:
        """Get the video key to use for latent extraction."""
        if self.latent_view_key is not None:
            return self.latent_view_key
        return self._default_latent_view_key()

    def _load_single_frame_as_pil(self, trajectory_id: int, video_key: str, step_index: int) -> Image.Image:
        """Load a single raw video frame as PIL.Image.

        Uses the same video loading infrastructure as get_video() but only
        requests a single frame index and returns a PIL Image.

        Args:
            trajectory_id: Trajectory ID
            video_key: Full video modality key (e.g. "video.cam_high")
            step_index: Absolute step index within the trajectory

        Returns:
            PIL.Image.Image: Raw frame (RGB)
        """
        # Ensure trajectory data is loaded (needed for timestamp lookup and image-in-parquet fallback)
        self.curr_traj_data = self.get_trajectory_data(trajectory_id)
        self.curr_traj_id = trajectory_id

        # Strip the "video." prefix for internal use
        key_stripped = video_key.replace("video.", "")

        # Check if images are stored in parquet columns (image-only datasets)
        import io

        original_key = self.lerobot_modality_meta.video[key_stripped].original_key
        if original_key is None:
            original_key = key_stripped

        if self.curr_traj_data is not None and original_key in self.curr_traj_data.columns:
            image_entries = self.curr_traj_data[original_key].tolist()
            safe_idx = int(min(max(step_index, 0), len(image_entries) - 1))
            entry = image_entries[safe_idx]

            if isinstance(entry, np.ndarray):
                return Image.fromarray(entry).convert("RGB")
            if isinstance(entry, Image.Image):
                return entry.convert("RGB")
            if isinstance(entry, dict):
                img_bytes = entry.get("bytes", None)
                img_path = entry.get("path", None)
                if img_bytes is not None:
                    return Image.open(io.BytesIO(img_bytes)).convert("RGB")
                if img_path is not None:
                    path_obj = Path(img_path)
                    if not path_obj.is_absolute():
                        path_obj = self.dataset_path / path_obj
                    return Image.open(path_obj).convert("RGB")
            raise TypeError(f"Unsupported image entry type: {type(entry)}")

        # Video file path
        video_path = self.get_video_path(trajectory_id, key_stripped)

        # Get timestamp for the requested step
        assert self.curr_traj_data is not None
        assert "timestamp" in self.curr_traj_data.columns
        timestamp: np.ndarray = self.curr_traj_data["timestamp"].to_numpy()
        safe_idx = int(min(max(step_index, 0), len(timestamp) - 1))
        video_timestamp = np.array([timestamp[safe_idx]])

        if self._lerobot_version == "v3.0":
            episode_meta = self.trajectory_ids_to_metadata.get(trajectory_id, {})
            from_timestamps = episode_meta.get("videos/from_timestamps", {})
            original_video_key = self.lerobot_modality_meta.video[key_stripped].original_key
            if original_video_key is None:
                original_video_key = key_stripped
            from_timestamp = float(from_timestamps.get(original_video_key, 0.0))
            video_timestamp = video_timestamp + from_timestamp

        # Fetch single frame: returns [1, H, W, C]
        frames = get_frames_by_timestamps(
            video_path.as_posix(),
            video_timestamp,
            video_backend=self.video_backend,
            video_backend_kwargs=self.video_backend_kwargs,
        )
        frame = frames[0]  # [H, W, C]
        return Image.fromarray(frame).convert("RGB")

    def _add_wm_fields(self, sample: dict, trajectory_id: int, base_index: int) -> dict:
        """Add WM-specific fields (present/future images, compute_action_loss) to a sample.

        This is extracted as a separate method so it can be called both from
        __getitem__ (single dataset) and from the mixture dataset's __getitem__.
        """
        latent_view_key = self._get_latent_view_key()

        if latent_view_key is not None:
            trajectory_index = self.get_trajectory_index(trajectory_id)
            traj_len = self.trajectory_lengths[trajectory_index]

            present_idx = min(base_index, traj_len - 1)
            present_frame = self._load_single_frame_as_pil(trajectory_id, latent_view_key, present_idx)

            # Multi-frame future targets at offsets s, 2s, ..., N*s (stride = future_frame_offset).
            # Frames past the episode end are clamped to the last frame and flagged invalid so the
            # world-model loss can mask them out. N == 1 reproduces the original single-frame sample.
            future_frames, future_valid = [], []
            for f in range(1, self.num_future_frames + 1):
                raw_idx = base_index + f * self.future_frame_offset
                is_valid = raw_idx <= (traj_len - 1)
                clamped_idx = min(raw_idx, traj_len - 1)
                future_frames.append(self._load_single_frame_as_pil(trajectory_id, latent_view_key, clamped_idx))
                future_valid.append(is_valid)

            sample["present_image"] = [present_frame]
            sample["future_image"] = future_frames
            sample["future_valid"] = future_valid
        else:
            sample["present_image"] = [sample["image"][0]]
            sample["future_image"] = [sample["image"][0]] * self.num_future_frames
            sample["future_valid"] = [True] * self.num_future_frames

        sample["compute_action_loss"] = self.compute_action_loss
        return sample

    def __getitem__(self, index: int) -> dict:
        """Get the data for a single step, including present/future images for WM training.

        Returns:
            dict: Standard training sample + present_image, future_image, compute_action_loss
        """
        trajectory_id, base_index = self.all_steps[index]

        # Standard data loading (calls get_step_data + transforms + _pack_sample)
        raw_data = self.get_step_data(trajectory_id, base_index)
        data = self.transforms(raw_data)
        sample = self._pack_sample(data)

        return self._add_wm_fields(sample, trajectory_id, base_index)


class LeRobotWMMixtureDataset(LeRobotMixtureDataset):
    """Mixture dataset for WM training — wraps multiple LeRobotWMSingleDataset instances.

    Overrides __getitem__ to add WM fields (present/future images, compute_action_loss)
    after the standard mixture sampling logic, since the base class directly calls
    dataset.get_step_data() + transforms + _pack_sample() bypassing __getitem__.
    """

    def __getitem__(self, index: int) -> dict:
        """Sample a step from the mixture and add WM fields."""
        import gc
        import os
        import random

        self._getitem_count += 1
        if self._getitem_count % 1000 == 0:
            gc.collect()

        max_retries = 10
        last_exception = None

        for attempt in range(max_retries):
            try:
                sample_tries = 0
                max_sample_tries = 200
                while True:
                    sample_tries += 1
                    if sample_tries > max_sample_tries:
                        raise RuntimeError(f"Unable to sample a valid item after {max_sample_tries} attempts.")

                    dataset, trajectory_id, step = self.sample_step(index)
                    total_videos = int(dataset.lerobot_info_meta.get("total_videos", 0))
                    if total_videos == 0:
                        break

                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)

                # Standard pipeline
                raw_data = dataset.get_step_data(trajectory_id, step)
                data = dataset.transforms(raw_data)
                sample = dataset._pack_sample(data)
                sample["robot_tag"] = dataset.tag

                # Add WM fields (present/future images, compute_action_loss)
                if hasattr(dataset, "_add_wm_fields"):
                    sample = dataset._add_wm_fields(sample, trajectory_id, step)

                return sample

            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"[WM Mixture] Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    index = random.randint(0, len(self) - 1)
                else:
                    print(f"[WM Mixture] All {max_retries} attempts failed for index {index}")
                    raise last_exception
