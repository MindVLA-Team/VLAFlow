import os
from collections import deque
from pathlib import Path
from typing import Dict, Optional

import cv2 as cv
import numpy as np
from transforms3d.euler import euler2axangle

from deployment.model_server.checkpoint import read_mode_config
from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble=True,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10093,
        action_output_dim: int = 14,
    ) -> None:

        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "oxe_bridge" if unnorm_key is None else unnorm_key
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 7 for widowx_bridge to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "oxe_rt1" if unnorm_key is None else unnorm_key
            action_ensemble = action_ensemble
            adaptive_ensemble_alpha = adaptive_ensemble_alpha
            if action_ensemble_horizon is None:
                # Set 2 for google_robot to fix the window size of motion scale between each frame. see appendix in our paper for details
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(
                f"Policy setup {policy_setup} not supported for octo models. The other datasets can be found in the huggingface config.json file."
            )
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps

        self.cfg_scale = cfg_scale  # 1.5

        self.image_size = image_size
        self.action_scale = action_scale  # 1.0
        self.horizon = horizon  # 0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        # action_output_dim: set to 14 when evaluating a dual-arm 14-D model on this single-arm
        # benchmark. The right-arm slice [7:14] is extracted before unnormalization so that
        # all downstream index references ([:3], [3:6], [6:7]) remain correct.
        assert action_output_dim in (7, 14), f"action_output_dim must be 7 or 14, got {action_output_dim}"
        self.action_output_dim = action_output_dim

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str], task description; if different from previous task description, policy state is reset
        Output:
            raw_action: dict; raw policy action output
            action: dict; processed action to be sent to the maniskill2 environment, with the following keys:
                - 'world_vector': np.ndarray of shape (3,), xyz translation of robot end-effector
                - 'rot_axangle': np.ndarray of shape (3,), axis-angle representation of end-effector rotation
                - 'gripper': np.ndarray of shape (1,), gripper action
                - 'terminate_episode': np.ndarray of shape (1,), 1 if episode should be terminated, 0 otherwise
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        self._add_image_to_history(self._resize_image(image))
        # image: Image.Image = Image.fromarray(image)

        # ── Capture the raw observation for world-model latent extractors
        # (e.g. MindWPI's VJEPA-2/DINOv3). The latent extractor has its own image
        # processor (resizes to 256x256), so we keep the original-resolution np.uint8
        # to match the training pipeline (LeRobotWMSingleDataset feeds raw frames
        # straight from the video file). We send numpy (not PIL) because msgpack-numpy
        # is the wire format for websocket; the server (MindWPI.predict_action)
        # applies to_pil_preserve to convert numpy → PIL — same pattern as `image`.
        # Models that don't use this field (MindPI/QwenPI/QwenGR00T/etc.) simply
        # ignore it.
        present_image_np = image  # np.uint8, original resolution

        image = self._resize_image(image)
        example = {
            "image": [image],
            "lang": self.task_description,
            # Used by MindWPI's predict_action → self.latent_extractor(present_images).
            # Ignored by all other frameworks (forward-compatible extra key).
            "present_image": [present_image_np],
        }

        vla_input = {
            "examples": [example],
            "do_sample": False,
            "cfg_scale": self.cfg_scale,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        response = self.client.predict_action(vla_input)

        # unnormalize the action
        normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
        normalized_actions = normalized_actions[0]

        # For 14-D dual-arm models: extract right-arm slots [7:14] → [pos3, rot3, gripper]
        # so that all downstream index references remain valid for single-arm evaluation.
        if self.action_output_dim == 14:
            normalized_actions = normalized_actions[:, 7:14]

        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats
        )

        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),  # range [0, 1]; 1 = open; 0 = close
        }

        # process raw_action to obtain the action to be sent to the maniskill2 environment
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)

        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = axes * angles
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action
            # fix a bug in the SIMPLER code here
            # self.previous_gripper_action = current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )

        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        # unnorm_key = baseframework._check_unnorm_key(norm_stats, unnorm_key) # This is also very environment-specific
        return norm_stats[unnorm_key]["action"]

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image
