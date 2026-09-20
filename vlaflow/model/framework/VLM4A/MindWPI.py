# Copyright (c) 2026 Li Auto. (VLAFlow modifications)
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
MindWPI Framework
MindVLM + Flow-matching head with KV cache sharing + Future-frame Latent World Prediction
"""
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
from vlaflow.model.modules.action_model.LayerwiseFM_LatentPredictActionHeader import get_latent_action_model
from vlaflow.model.modules.latent_extractor import get_latent_model
from vlaflow.model.modules.vlm import get_vlm_model
from vlaflow.model.tools import FRAMEWORK_REGISTRY
from vlaflow.training.trainer_utils.trainer_tools import resize_images


@FRAMEWORK_REGISTRY.register("MindWPI")
class Mind_WPI(baseframework):
    """
    MindWPI: Vision-Language-Action model with KV cache sharing + pseudo world model.

    Components:
      - MindVLM for multimodal understanding
      - Flow-matching action head with KV cache sharing
      - Latent extractor (VJEPA-2/DINOv3) for visual feature extraction
      - Future-frame latent prediction (single-pass parallel decode via DiT)

    The key idea: present-frame latent features are injected into the DiT as tokens.
    These tokens predict future-frame latents in parallel (not autoregressive) after
    a single pass through the DiT backbone, alongside action prediction.
    """

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

        print(
            f"MindWPI Action Model total parameters: "
            f"{sum(p.numel() for p in self.action_model.parameters()) / 1e6:.2f}M, "
            f"trainable parameters: "
            f"{sum(p.numel() for p in self.action_model.parameters() if p.requires_grad) / 1e6:.2f}M",
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

    def forward(
        self,
        examples: List[dict] = None,
        **kwargs,
    ) -> Tuple:
        """
        Forward pass with KV cache sharing + future-frame latent prediction.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
                - present_image: List[PIL.Image] (single front-view, current frame)
                - future_image: List[PIL.Image] (single front-view, future frame(s))
        Returns:
            dict:
                action_loss (torch.Tensor): Combined action + latent prediction loss.
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
        # Datasets that should not contribute action loss (e.g. low-quality action
        # labels during pretraining) set this to False/0 in the data pipeline.
        # Shape after conversion: [B] boolean tensor.
        compute_action_loss_flags = None
        if "compute_action_loss" in examples[0]:
            compute_action_loss_flags = [example["compute_action_loss"] for example in examples]

        # Step 1: Build VLM inputs
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

            vlm_past_key_values = vlm_outputs.past_key_values
            vlm_attention_mask = qwen_inputs["attention_mask"]

        # Step 4: Action + latent loss computation
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

            # Compute combined loss (action flow-matching + latent prediction)
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

        return {
            "action_loss": loss_dict["total_loss"],  # backward target (keeps trainer compat)
            "action_fm_loss": loss_dict["action_loss"],  # monitoring: pure action flow-matching loss
            "latent_pred_loss": loss_dict["latent_loss"],  # monitoring: latent prediction loss
        }

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: use diffusion sampling with KV cache sharing.
        Present latents provide visual context to the action head.

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
        # present_image arrives as np.ndarray (uint8) over msgpack-numpy from clients,
        # or as PIL.Image when called in-process. to_pil_preserve handles both
        # uniformly (PIL passes through, numpy → PIL) — same pattern as `image` above.
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

        # Step 4: Action prediction with KV cache sharing + latent context
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vlm_past_key_values=vlm_past_key_values,
                vlm_attention_mask=vlm_attention_mask,
                state=state_tensor,
                present_latents=present_latents,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
