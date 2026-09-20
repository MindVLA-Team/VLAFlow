# Copyright (c) 2026 Li Auto Inc. (VLAFlow modifications)
# VLAFlow modifications are licensed under Apache-2.0; see LICENSE.
#
# Original upstream notice (terms retained in LICENSES/LicenseRef-StarVLA.txt):
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""MindLWPI with AvgPool-k4 future-latent alignment and action-language supervision.

Action and latent losses share the KV-conditioned action head. Optional language
cross-entropy adds a separate VLM pass during pretraining; downstream fine-tuning
disables it. Module names are retained for checkpoint compatibility.
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from deployment.model_server.tools.image_tools import to_pil_preserve
from vlaflow.training.trainer_utils import initialize_overwatch

logger = initialize_overwatch(__name__)

# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

from vlaflow.model.framework.base_framework import baseframework
from vlaflow.model.modules.action_model.LayerwiseFM_LatentCompressedActionHeader import get_latent_action_model
from vlaflow.model.modules.latent_extractor import get_latent_model
from vlaflow.model.modules.vlm import get_vlm_model
from vlaflow.model.tools import FRAMEWORK_REGISTRY
from vlaflow.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("MindLWPI_Compressed")
class Mind_LWPI_Compressed(baseframework):
    """Action modeling, AvgPool-k4 latent alignment, and optional language supervision."""

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        self.latent_extractor = get_latent_model(config=self.config)

        # Freeze the latent extractor — it's a pretrained feature provider
        for param in self.latent_extractor.parameters():
            param.requires_grad = False

        # Dynamically read LLM config
        _model_cfg = self.qwen_vl_interface.model.config
        llm_cfg = getattr(_model_cfg, "text_config", _model_cfg)
        num_vl_layers = llm_cfg.num_hidden_layers
        llm_hidden_size = llm_cfg.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        latent_dim = self.latent_extractor.get_extractor_dim()

        self.action_model = get_latent_action_model(config=self.config, latent_dim=latent_dim)

        _compress_ratio = int(getattr(self.config.framework.action_model, "latent_compress_ratio", 1))
        _compress_method = str(getattr(self.config.framework.action_model, "latent_compress_method", "avgpool"))
        print(
            f"MindLWPI_Compressed Action Model "
            f"(compress_ratio={_compress_ratio}, compress_method={_compress_method!r}) "
            f"total parameters: "
            f"{sum(p.numel() for p in self.action_model.parameters()) / 1e6:.2f}M, "
            f"trainable parameters: "
            f"{sum(p.numel() for p in self.action_model.parameters() if p.requires_grad) / 1e6:.2f}M"
        )

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Number of times to replicate each batch element for flow-matching Monte-Carlo coverage
        self.repeated_diffusion_steps = int(getattr(self.config.framework.action_model, "repeated_diffusion_steps", 1))

        # Meta query for soft-connection
        num_queries = int(self.config.framework.qwenvl.get("num_queries", 0))
        if num_queries > 0:
            self.meta_queries = nn.Parameter(torch.randn(num_queries, llm_hidden_size))
        else:
            self.meta_queries = None

        # ── LAP language-action settings ───────────────────────────────────
        # Read directly from config.framework (MindWPI-style direct read), with
        # safe defaults so the keys are optional (e.g. downstream YAMLs / smoke-test).
        fw = self.config.framework
        self.enable_lang_action = bool(fw.get("enable_lang_action", False))
        self.lang_action_format = str(fw.get("lang_action_format", "verbose"))
        self.vla0_num_bins = int(fw.get("vla0_num_bins", 1000))
        self.include_rotation = bool(fw.get("include_rotation", True))
        self.detach_action_kv = bool(fw.get("detach_action_kv", False))
        self.lang_action_loss_weight = float(fw.get("lang_action_loss_weight", 0.05))

        # Global norm stats for verbose/compact denormalization.
        # Loaded only when format is verbose/compact — vla0 works on normalized [-1,1].
        self._global_norm_q01: Optional[np.ndarray] = None  # shape [6]: x/y/z/roll/pitch/yaw
        self._global_norm_q99: Optional[np.ndarray] = None
        if self.enable_lang_action and self.lang_action_format in {"verbose", "compact"}:
            _stats_path = str(fw.get("global_norm_stats_path", "") or "")
            if _stats_path:
                self._global_norm_q01, self._global_norm_q99 = self._load_global_norm_stats(_stats_path)
            else:
                logger.warning(
                    "[MindLWPI_Compressed] lang_action_format='%s' requires global_norm_stats_path "
                    "to denormalize actions to physical units. Actions will NOT be correctly "
                    "converted to text without it. Set framework.global_norm_stats_path in config.",
                    self.lang_action_format,
                )

        logger.info(
            f"[MindLWPI_Compressed] enable_lang_action={self.enable_lang_action}, "
            f"lang_action_format={self.lang_action_format!r}, "
            f"detach_action_kv={self.detach_action_kv}, "
            f"lang_w={self.lang_action_loss_weight}, "
            f"include_rotation={self.include_rotation}, "
            f"compress_ratio={_compress_ratio}, compress_method={_compress_method!r}, "
            f"future_window={self.future_action_window_size}, "
            f"repeated_diffusion_steps={self.repeated_diffusion_steps}"
        )

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Forward pass: compressed future-frame latent prediction + flow-matching action +
        (optional) LAP language-action CE.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
                - present_image: List[PIL.Image] (single front-view, current frame)
                - future_image: List[PIL.Image] (single front-view, future frame(s))
        Returns:
            dict:
                action_loss (torch.Tensor):       backward target (weighted sum of all losses)
                action_fm_loss (torch.Tensor):    pure action flow-matching loss (monitor)
                latent_pred_loss (torch.Tensor):  compressed future-latent prediction loss (monitor)
                lang_action_loss (torch.Tensor):  language-action CE loss (monitor; 0 when disabled)
        """
        batch_images = [example["image"] for example in examples]  # [B, [PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # [B, len, action_dim]

        state = (
            [example["state"] for example in examples]
            if self.config.framework.get("use_state", False) and "state" in examples[0]
            else None
        )

        # For latent world prediction
        future_images = [example["future_image"] for example in examples]  # [B, [PIL]]
        present_images = [example["present_image"] for example in examples]  # [B, [PIL]]

        # Per-element loss mask
        action_masks = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None

        # Per-sample flag: whether to compute action loss for this sample.
        compute_action_loss_flags = None
        if "compute_action_loss" in examples[0]:
            compute_action_loss_flags = [example["compute_action_loss"] for example in examples]

        # ── Path L: LAP language-action CE loss → VLM (pretraining only) ────
        # Computed first as an independent VLM forward (with the action description
        # as the assistant target). Disabled downstream (enable_lang_action=False).
        lang_loss = torch.zeros((), device=self.qwen_vl_interface.model.device, dtype=torch.float32)
        if self.enable_lang_action:
            lang_actions = [self._action_to_language(a) for a in actions]
            lang_inputs = self._build_langact_inputs(batch_images, instructions, lang_actions)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lang_out = self.qwen_vl_interface(**lang_inputs)
            _ll = lang_out.loss
            if _ll is not None and not torch.isnan(_ll):
                lang_loss = _ll

        # Step 1: Build VLM inputs (action+latent path; no assistant solution)
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        # Step 2: Extract latents (frozen extractor, no grad)
        with torch.no_grad():
            future_latents = self.latent_extractor(future_images)  # [B, N, latent_dim]
            present_latents = self.latent_extractor(present_images)  # [B, N, latent_dim]

        # Step 3: VLM forward with KV cache enabled
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
                meta_queries=self.meta_queries,
            )

            # detach_action_kv=False (default): KV cache attached → action/latent loss
            #   gradients flow back to the VLM (paper's "w/o stop-gradient" setting).
            # detach_action_kv=True: detach KV cache → action/latent loss updates the
            #   action head only; the VLM is shaped by the language CE path alone.
            if self.detach_action_kv:
                vlm_past_key_values = [(k.detach(), v.detach()) for k, v in vlm_outputs.past_key_values]
            else:
                vlm_past_key_values = list(vlm_outputs.past_key_values)
            vlm_attention_mask = qwen_inputs["attention_mask"]

        # Step 4: Action + compressed-latent loss computation
        with torch.autocast("cuda", dtype=torch.float32):
            # Target actions: take last chunk
            actions = torch.tensor(np.array(actions), device=vlm_attention_mask.device, dtype=torch.float32)
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]

            # Slice action mask
            action_mask_target = None
            if action_masks is not None:
                action_mask_target = torch.tensor(
                    np.array(action_masks),
                    device=vlm_attention_mask.device,
                    dtype=torch.float32,
                )
                action_mask_target = action_mask_target[:, -(self.future_action_window_size + 1) :, :]

            # Prepare state tensor
            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(np.array(state), device=vlm_attention_mask.device, dtype=torch.float32)
                B, L, D = state_tensor.shape
                if L > 1:
                    state_tensor = state_tensor[:, -1, :].unsqueeze(dim=1)

            # Build per-sample action loss flag tensor
            action_loss_flags_tensor = None
            if compute_action_loss_flags is not None:
                action_loss_flags_tensor = torch.tensor(
                    compute_action_loss_flags,
                    device=vlm_attention_mask.device,
                    dtype=torch.bool,
                )  # [B]

            # Repeated diffusion steps for better Monte-Carlo coverage
            if self.repeated_diffusion_steps > 1:
                n = self.repeated_diffusion_steps
                actions_target = actions_target.repeat(n, 1, 1)
                vlm_attention_mask = vlm_attention_mask.repeat(n, 1)
                vlm_past_key_values = [(k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1)) for k, v in vlm_past_key_values]
                present_latents = present_latents.repeat(n, 1, 1)
                future_latents = future_latents.repeat(n, 1, 1)
                if state_tensor is not None:
                    state_tensor = state_tensor.repeat(n, 1, 1)
                if action_mask_target is not None:
                    action_mask_target = action_mask_target.repeat(n, 1, 1)
                if action_loss_flags_tensor is not None:
                    action_loss_flags_tensor = action_loss_flags_tensor.repeat(n)

            # Compute combined loss (action flow-matching + compressed latent prediction)
            loss_dict = self.action_model(
                vlm_past_key_values=vlm_past_key_values,
                vlm_attention_mask=vlm_attention_mask,
                actions=actions_target,
                state=state_tensor,
                action_mask=action_mask_target,
                present_latents=present_latents,
                future_latents=future_latents,
                compute_action_loss_flags=action_loss_flags_tensor,
            )

        # ── Combine all three losses ───────────────────────────────────────
        # loss_dict["total_loss"] already = action_loss_weight * action_loss
        #                                  + latent_loss_weight * latent_loss
        # so all three weights remain independently tunable.
        backward_loss = loss_dict["total_loss"] + self.lang_action_loss_weight * lang_loss

        return {
            "action_loss": backward_loss,  # backward target (keeps trainer compat)
            "action_fm_loss": loss_dict["action_loss"],  # monitoring: pure action flow-matching loss
            "latent_pred_loss": loss_dict["latent_loss"],  # monitoring: compressed latent prediction loss
            "lang_action_loss": lang_loss,  # monitoring: LAP language-action CE (0 when disabled)
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Predict actions using KV sharing and pooled present-frame features.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - state: np.ndarray shaped [1, state_dim] (optional)
                - present_image: List[PIL.Image] (single front-view, current frame)
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim].
        """
        if type(examples) is not list:
            examples = [examples]

        batch_images = [to_pil_preserve(example["image"]) for example in examples]
        instructions = [example["lang"] for example in examples]
        state = (
            [example["state"] for example in examples]
            if self.config.framework.get("use_state", False) and "state" in examples[0]
            else None
        )
        present_images = (
            [to_pil_preserve(example["present_image"]) for example in examples]
            if "present_image" in examples[0]
            else None
        )

        train_obs_image_size = getattr(self.config.datasets.vla_data, "image_size", None)
        if train_obs_image_size:
            batch_images = resize_images(batch_images, target_size=train_obs_image_size)

        # Step 1: VLM forward with KV cache
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,
                meta_queries=self.meta_queries,
            )

            vlm_past_key_values = vlm_outputs.past_key_values
            vlm_attention_mask = qwen_inputs["attention_mask"]
            last_hidden_states = vlm_outputs.hidden_states[-1]

        # Step 2: Extract present latents for visual context
        present_latents = None
        if present_images is not None:
            with torch.no_grad():
                present_latents = self.latent_extractor(present_images)  # [B, N, latent_dim]

        # Step 3: Prepare state
        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(last_hidden_states.device, dtype=torch.float32)
            B, L, D = state_tensor.shape
            if L > 1:
                state_tensor = state_tensor[:, -1, :].unsqueeze(dim=1)

        # Step 4: Action prediction with KV cache sharing + compressed latent context
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vlm_past_key_values=vlm_past_key_values,
                vlm_attention_mask=vlm_attention_mask,
                state=state_tensor,
                present_latents=present_latents,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}

    # ------------------------------------------------------------------
    # Language-action conversion and assistant-response label masking
    # ------------------------------------------------------------------

    def _action_to_language(self, actions: np.ndarray) -> str:
        """Convert action array to language string using the configured format.

        OXE-Mix 14D layout: SingleArmToDualArmTransform places single-arm data
        in the RIGHT arm (slots 7-13). The LEFT arm (slots 0-6) is always zeros
        + -1 gripper sentinel. We extract the right arm for text conversion.

        For verbose/compact formats, the extracted 7D action is then denormalized
        from [-1,1] to physical units (meters/radians) using global q01/q99 stats.
        For vla0, normalized [-1,1] values are used directly.
        """
        from vlaflow.dataloader.lap_action_converter import convert_action_to_language

        action_horizon = self.future_action_window_size + 1

        arr = np.asarray(actions, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[None, :]

        # Extract 7D right-arm action from 14D dual-arm layout:
        #   14D: [left_pos(0:3), left_euler(3:6), left_grip(6),
        #         right_pos(7:10), right_euler(10:13), right_grip(13)]
        # For verbose/compact/vla0: use right arm [pos3, euler3, grip1].
        if arr.shape[-1] == 14:
            act = np.concatenate([arr[..., 7:13], arr[..., 13:14]], axis=-1)  # [T, 7]
        else:
            act = arr[..., :7]

        if self.lang_action_format in {"verbose", "compact"} and self._global_norm_q01 is not None:
            act = self._denormalize_cont_actions(act)

        return convert_action_to_language(
            act,
            lang_action_format=self.lang_action_format,
            action_horizon=action_horizon,
            include_rotation=self.include_rotation,
            num_bins=self.vla0_num_bins,
        )

    @staticmethod
    def _load_global_norm_stats(path: str) -> tuple:
        """Load global norm stats JSON → (q01, q99) numpy arrays of shape [6].

        Key order: [x, y, z, roll, pitch, yaw] — maps to the 7D right-arm
        action [right_pos(3), right_euler(3), right_gripper(1)] after extracting
        the right arm from the 14D OXE-Mix layout.
        Uses the output format of scripts/precompute_oxemix_global_stats.py.

        Returns:
            (q01, q99): float32 arrays of shape [6], or (None, None) on failure.
        """
        p = Path(path)
        if not p.exists():
            logger.warning("[MindLWPI_Compressed] global_norm_stats_path not found: %s", path)
            return None, None
        try:
            with open(p) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("[MindLWPI_Compressed] Failed to load global_norm_stats_path %s: %s", path, e)
            return None, None

        action_stats = raw.get("action", {})
        key_order = ["x", "y", "z", "roll", "pitch", "yaw"]
        q01 = np.zeros(6, dtype=np.float32)
        q99 = np.ones(6, dtype=np.float32)
        for i, key in enumerate(key_order):
            s = action_stats.get(key, {})
            if "q01" in s and "q99" in s:
                q01[i] = float(s["q01"])
                q99[i] = float(s["q99"])

        logger.info("[MindLWPI_Compressed] Loaded global norm stats: q01=%s, q99=%s", q01, q99)
        return q01, q99

    def _denormalize_cont_actions(self, actions: np.ndarray) -> np.ndarray:
        """Denormalize 7D action [dx, dy, dz, droll, dpitch, dyaw, gripper] from [-1,1] to physical units.

        Applies the inverse of GR00T's q99/min_max normalization (identical formula):
            physical = (normalized + 1) / 2 * (q99 - q01) + q01

        Dims 0-5: translation (m) + rotation (rad) from the right arm.
        Dim  6:   gripper — kept as-is; the binary threshold g >= 0.5 works
                  correctly for open(1.0)/close(-1.0) in normalized space.
        """
        arr = np.asarray(actions, dtype=np.float64).copy()
        q01 = self._global_norm_q01.astype(np.float64)  # shape [6]
        q99 = self._global_norm_q99.astype(np.float64)  # shape [6]
        # Inverse of: normalized = 2 * (x - q01) / (q99 - q01) - 1
        arr[..., :6] = (arr[..., :6] + 1.0) / 2.0 * (q99 - q01) + q01
        return arr

    def _build_langact_inputs(self, images, instructions, lang_actions):
        """Build tokenized VLM inputs with language action as the assistant response.

        Label masking strategy:
          - Everything before the assistant's response → IGNORE_INDEX (-100)
          - Only the language action tokens → contribute to LM loss
        """
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=images,
            instructions=instructions,
            solutions=lang_actions,  # assistant response = language action text
        )
        # Supervise only the assistant's action-description response.
        qwen_inputs = self._create_langact_labels(qwen_inputs)
        return qwen_inputs

    def _create_langact_labels(self, batch_input: dict) -> dict:
        """Re-create labels so only the assistant's language action response is unmasked.

        Finds the last <|im_start|>assistant\\n marker and masks all prefix tokens.
        Works for Qwen2.5-VL, Qwen3-VL, and any VLM whose chat template follows the
        <|im_start|>role\\n pattern.
        """
        tokenizer = self.qwen_vl_interface.processor.tokenizer
        input_ids = batch_input["input_ids"]

        # Resolve <|im_start|> token id
        try:
            im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
            if im_start_id is None or im_start_id == tokenizer.unk_token_id:
                im_start_id = 151644
        except Exception:
            im_start_id = 151644  # Qwen2.5-VL default

        # Resolve \n token id (usually 198 in Qwen tokenizers)
        try:
            newline_id = tokenizer.encode("\n", add_special_tokens=False)[-1]
        except Exception:
            newline_id = 198

        pad_id = tokenizer.pad_token_id

        labels = input_ids.clone()

        for i in range(labels.size(0)):
            seq = labels[i]
            seq_len = seq.size(0)

            # Find all positions of <|im_start|>
            im_start_positions = (seq == im_start_id).nonzero(as_tuple=False).flatten()
            if im_start_positions.numel() == 0:
                seq[:] = IGNORE_INDEX
                continue

            # Actual non-pad length
            if pad_id is not None:
                actual_end = (seq != pad_id).sum().item()
                actual_start = seq_len - actual_end
            else:
                actual_start = 0
                actual_end = seq_len

            # Search RIGHT to LEFT for <|im_start|> with response content after it
            found_response_start = -1
            for pos_idx in range(len(im_start_positions) - 1, -1, -1):
                pos = im_start_positions[pos_idx].item()
                subseq = seq[pos + 1 :]
                nl_positions = (subseq == newline_id).nonzero(as_tuple=False).flatten()
                if nl_positions.numel() == 0:
                    continue
                response_start = pos + 1 + nl_positions[0].item() + 1
                if response_start < actual_start + actual_end - 2:
                    found_response_start = response_start
                    break

            if found_response_start == -1:
                seq[:] = IGNORE_INDEX
            else:
                seq[:found_response_start] = IGNORE_INDEX

        # Mask padding tokens
        if pad_id is not None:
            labels[labels == pad_id] = IGNORE_INDEX

        batch_input["labels"] = labels
        return batch_input
