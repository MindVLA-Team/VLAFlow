# Copyright (c) 2026 Li Auto Inc. (VLAFlow modifications)
# VLAFlow modifications are licensed under Apache-2.0; see LICENSE.
#
# Original upstream notice (terms retained in LICENSES/LicenseRef-StarVLA.txt):
# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License");
"""
MindPI Framework
MindVLM + Flow-matching head with KV cache sharing (MiRobot-style)
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
from vlaflow.model.modules.action_model.LayerwiseFM_KVShareActionHeader import get_kv_shared_action_model
from vlaflow.model.modules.vlm import get_vlm_model
from vlaflow.model.tools import FRAMEWORK_REGISTRY
from vlaflow.training.trainer_utils.trainer_tools import resize_images

####################################################
# ⚠️ Warning: This framework implements KV cache sharing
# architecture and is NOT compatible with previous
# checkpoints created before 2025-03-26.
####################################################


@FRAMEWORK_REGISTRY.register("MindPI")
class Mind_PI(baseframework):
    """
    MindPI: Vision-Language-Action model with KV cache sharing.

    Components:
      - MindVLM for multimodal understanding
      - Flow-matching action head with KV cache sharing
      - DiT can attend to VLM's KV cache directly (MiRobot-style)

    Focus: Predict future continuous actions conditioned on images + instruction.
    """

    def __init__(
        self,
        config: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Construct all submodules and cache key configuration values.

        Args:
            config: Hierarchical configuration (OmegaConf/dict) containing framework + trainer sections.
            **kwargs: Reserved for future overrides (unused).
        """

        super().__init__()
        self.config = config
        self.qwen_vl_interface = get_vlm_model(config=self.config)

        # Dynamically read LLM config — works for any Qwen2.5/3/3.5 or other backbone.
        # Qwen3VL / Qwen2.5VL nest language-model settings under .text_config;
        # plain LLM backbones expose them directly on the top-level config.
        _model_cfg = self.qwen_vl_interface.model.config
        llm_cfg = getattr(_model_cfg, "text_config", _model_cfg)
        num_vl_layers = llm_cfg.num_hidden_layers
        llm_hidden_size = llm_cfg.hidden_size
        self.config.framework.qwenvl.vl_hidden_dim = llm_hidden_size
        self.config.framework.qwenvl.num_vl_layers = num_vl_layers

        action_head_type = "dit"
        if getattr(self.config.framework.action_model, "action_head_type", "dit") != "dit":
            raise ValueError("MindPI supports the DiT action head in this release.")
        self.action_model = get_kv_shared_action_model(config=self.config)

        print(
            f"MindPI [{action_head_type}] Action Model total parameters: "
            f"{sum(p.numel() for p in self.action_model.parameters()) / 1e6:.2f}M, "
            f"trainable parameters: "
            f"{sum(p.numel() for p in self.action_model.parameters() if p.requires_grad) / 1e6:.2f}M",
            "current",
        )

        self.future_action_window_size = config.framework.action_model.future_action_window_size
        self.past_action_window_size = config.framework.action_model.past_action_window_size
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size

        # Number of times to replicate each batch element before the action-head
        # forward pass, giving the flow-matching loss N independent (t, noise)
        # samples per sequence.  Set to 1 (default) to disable.
        self.repeated_diffusion_steps = int(getattr(self.config.framework.action_model, "repeated_diffusion_steps", 1))

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
        Forward pass with KV cache sharing.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - action: np.ndarray or list shaped [T, action_dim]
        Returns:
            dict:
                action_loss (torch.Tensor): Scalar flow matching loss.
        """
        batch_images = [example["image"] for example in examples]  # [B，[PIL]]
        instructions = [example["lang"] for example in examples]  # [B, str]
        actions = [example["action"] for example in examples]  # label [B， len, 7]

        state = (
            [example["state"] for example in examples]
            if self.config.framework.get("use_state", False) and "state" in examples[0]
            else None
        )  # [B, 1, state_dim]

        # Per-element loss mask shipped by the dataloader (1=real, 0=padded /
        # sentinel). Used to keep single-arm samples from teaching the model
        # to predict zeros on the unused arm slots when training in the shared
        # 14-D dual-arm space.
        action_masks = [example["action_mask"] for example in examples] if "action_mask" in examples[0] else None

        # Step 1: Build VLM inputs
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(images=batch_images, instructions=instructions)

        # Step 2: VLM forward with KV cache enabled
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vlm_outputs = self.qwen_vl_interface(
                **qwen_inputs,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
                use_cache=True,  # Enable KV cache
                meta_queries=self.meta_queries,  # only support mindvlm now, other backbone won't affect
            )

            # Get KV cache from VLM
            vlm_past_key_values = vlm_outputs.past_key_values  # List of tuples (k, v) per layer
            vlm_attention_mask = qwen_inputs["attention_mask"]  # [B, vlm_seq_len]
            last_hidden_states = vlm_outputs.hidden_states[-1]

        # Step 3: Action loss computation
        with torch.autocast("cuda", dtype=torch.float32):
            # Target actions: take last chunk_len tokens
            actions = torch.tensor(
                np.array(actions), device=last_hidden_states.device, dtype=torch.float32
            )  # [B, T_full, action_dim]
            actions_target = actions[:, -(self.future_action_window_size + 1) :, :]

            # Slice the same trailing window from the action mask, if provided.
            action_mask_target = None
            if action_masks is not None:
                action_mask_target = torch.tensor(
                    np.array(action_masks),
                    device=last_hidden_states.device,
                    dtype=torch.float32,
                )
                action_mask_target = action_mask_target[:, -(self.future_action_window_size + 1) :, :]

            # Prepare state tensor
            state_tensor = None
            if state is not None:
                state_tensor = torch.tensor(np.array(state), device=last_hidden_states.device, dtype=torch.float32)

                B, L, D = state_tensor.shape
                if L > 1:
                    state_tensor = state_tensor[:, -1, :].unsqueeze(
                        dim=1
                    )  # shape [B, 1, D], only the last state will be used

            # ── Repeated diffusion steps ─────────────────────────────────────
            # Replicate each batch element N times so the action head draws N
            # independent (timestep, noise) samples per sequence, improving
            # Monte-Carlo coverage of the flow-matching timestep distribution.
            if self.repeated_diffusion_steps > 1:
                n = self.repeated_diffusion_steps
                actions_target = actions_target.repeat(n, 1, 1)
                vlm_attention_mask = vlm_attention_mask.repeat(n, 1)
                vlm_past_key_values = [(k.repeat(n, 1, 1, 1), v.repeat(n, 1, 1, 1)) for k, v in vlm_past_key_values]
                if state_tensor is not None:
                    state_tensor = state_tensor.repeat(n, 1, 1)
                if action_mask_target is not None:
                    action_mask_target = action_mask_target.repeat(n, 1, 1)

            # Compute action loss with KV cache sharing
            action_loss = self.action_model(
                vlm_past_key_values=vlm_past_key_values,
                vlm_attention_mask=vlm_attention_mask,
                actions=actions_target,
                state=state_tensor,
                action_mask=action_mask_target,
            )

        return {"action_loss": action_loss}

    @torch.inference_mode()
    def predict_action(
        self,
        examples: List[dict] = None,
        **kwargs: str,
    ) -> np.ndarray:
        """
        Inference: use diffusion sampling with KV cache sharing.

        Args:
            examples: List[dict], each dict requires:
                - image: List[PIL.Image] (multi-view)
                - lang: str instruction
                - state: np.ndarray shaped [1, state_dim] (optional)
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
                meta_queries=self.meta_queries,  # only support mindvlm now, other backbone won't affect
            )

            vlm_past_key_values = vlm_outputs.past_key_values
            vlm_attention_mask = qwen_inputs["attention_mask"]
            last_hidden_states = vlm_outputs.hidden_states[-1]

        # Step 2: Prepare state
        state_tensor = None
        if state is not None:
            state_tensor = torch.from_numpy(np.array(state)).to(last_hidden_states.device, dtype=torch.float32)

            B, L, D = state_tensor.shape
            if L > 1:
                state_tensor = state_tensor[:, -1, :].unsqueeze(
                    dim=1
                )  # shape [B, 1, D], only the last state will be used

        # Step 3: Action prediction with KV cache sharing
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                vlm_past_key_values=vlm_past_key_values,
                vlm_attention_mask=vlm_attention_mask,
                state=state_tensor,
            )

        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}
