# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from enum import Enum


class EmbodimentTag(Enum):
    GR1 = "gr1"
    """
    The GR1 dataset.
    """

    OXE_DROID = "oxe_droid"
    """
    The OxE Droid dataset.
    """

    OXE_BRIDGE = "oxe_bridge"
    """
    The OxE Bridge dataset.
    """

    OXE_RT1 = "oxe_rt1"
    """
    The OxE RT-1 dataset.
    """

    AGIBOT_GENIE1 = "agibot_genie1"
    """
    The AgiBot Genie-1 with gripper dataset.
    """

    NEW_EMBODIMENT = "new_embodiment"
    """
    Any new embodiment for finetuning.
    """

    FRANKA = "franka"
    """
    The Franka Emika Panda robot.
    """

    AGILEX_COBOT_MAGIC = "agilex_cobot_magic"
    """
    The Agilex Cobot Magic dual-arm robot (RoboCOIN_SUB).
    """

    GALAXEA_R1_LITE = "galaxea_r1_lite"
    """
    The Galaxea R1 Lite dual-arm robot (RoboCOIN_SUB).
    """

    AGILEX_SPLIT_ALOHA = "agilex_split_aloha"
    """
    The Agilex Split Aloha dual-arm robot (RoboCOIN_SUB).
    """

    OXE_AUGE_WIDOWX = "oxe_auge_widowx"
    """
    OXE-AugE augmented dataset — WidowX robot.
    """

    OXE_AUGE_SAWYER = "oxe_auge_sawyer"
    """
    OXE-AugE augmented dataset — Sawyer robot.
    """

    OXE_AUGE_UR5E = "oxe_auge_ur5e"
    """
    OXE-AugE augmented dataset — UR5e robot.
    """

    OXE_AUGE_GOOGLE = "oxe_auge_google_robot"
    """
    OXE-AugE augmented dataset — Google robot.
    """

    OXE_AUGE_JACO = "oxe_auge_jaco"
    """
    OXE-AugE augmented dataset — Jaco robot.
    """

    OXE_AUGE_KINOVA3 = "oxe_auge_kinova3"
    """
    OXE-AugE augmented dataset — Kinova3 robot.
    """

    OXE_AUGE_KUKA = "oxe_auge_kuka_iiwa"
    """
    OXE-AugE augmented dataset — Kuka IIWA robot.
    """

    OXE_AUGE_XARM7 = "oxe_auge_xarm7"
    """
    OXE-AugE augmented dataset — xArm7 robot.
    """

    SO100_SINGLE = "so100_single"
    """
    SO100 single-arm robot (community_dataset_v2) — joint-space, 6-DOF (5 joints + gripper).
    """

    SO100_DUAL = "so100_dual"
    """
    SO100 dual-arm robot (community_dataset_v2) — joint-space, 12-DOF (2× 5 joints + gripper).
    """


# Embodiment tag string: to projector index in the Action Expert Module
EMBODIMENT_TAG_MAPPING = {
    EmbodimentTag.NEW_EMBODIMENT.value: 31,
    EmbodimentTag.OXE_DROID.value: 17,
    EmbodimentTag.OXE_BRIDGE.value: 18,
    EmbodimentTag.OXE_RT1.value: 19,
    EmbodimentTag.AGIBOT_GENIE1.value: 26,
    EmbodimentTag.GR1.value: 24,
    EmbodimentTag.FRANKA.value: 25,
    # RoboCOIN_SUB
    EmbodimentTag.AGILEX_COBOT_MAGIC.value: 8,
    EmbodimentTag.GALAXEA_R1_LITE.value: 9,
    EmbodimentTag.AGILEX_SPLIT_ALOHA.value: 10,
    # OXE_AugE — 8 robots
    EmbodimentTag.OXE_AUGE_WIDOWX.value: 11,
    EmbodimentTag.OXE_AUGE_SAWYER.value: 12,
    EmbodimentTag.OXE_AUGE_UR5E.value: 13,
    EmbodimentTag.OXE_AUGE_GOOGLE.value: 14,
    EmbodimentTag.OXE_AUGE_JACO.value: 15,
    EmbodimentTag.OXE_AUGE_KINOVA3.value: 16,
    EmbodimentTag.OXE_AUGE_KUKA.value: 20,
    EmbodimentTag.OXE_AUGE_XARM7.value: 21,
    # SO100 community data — joint-space robots
    EmbodimentTag.SO100_SINGLE.value: 22,
    EmbodimentTag.SO100_DUAL.value: 23,
}

# Robot type to embodiment tag mapping
ROBOT_TYPE_TO_EMBODIMENT_TAG = {
    "libero_franka": EmbodimentTag.FRANKA,
    "oxe_droid": EmbodimentTag.OXE_DROID,
    "oxe_bridge": EmbodimentTag.OXE_BRIDGE,
    "oxe_rt1": EmbodimentTag.OXE_RT1,
    "demo_sim_franka_delta_joints": EmbodimentTag.FRANKA,
    "custom_robot_config": EmbodimentTag.NEW_EMBODIMENT,
    "fourier_gr1_arms_waist": EmbodimentTag.GR1,
    # RoboCOIN_SUB
    "agilex_cobot_magic": EmbodimentTag.AGILEX_COBOT_MAGIC,
    "galaxea_r1_lite": EmbodimentTag.GALAXEA_R1_LITE,
    "agilex_split_aloha": EmbodimentTag.AGILEX_SPLIT_ALOHA,
    # OXE_AugE
    "oxe_auge_widowx": EmbodimentTag.OXE_AUGE_WIDOWX,
    "oxe_auge_sawyer": EmbodimentTag.OXE_AUGE_SAWYER,
    "oxe_auge_ur5e": EmbodimentTag.OXE_AUGE_UR5E,
    "oxe_auge_google_robot": EmbodimentTag.OXE_AUGE_GOOGLE,
    "oxe_auge_jaco": EmbodimentTag.OXE_AUGE_JACO,
    "oxe_auge_kinova3": EmbodimentTag.OXE_AUGE_KINOVA3,
    "oxe_auge_kuka_iiwa": EmbodimentTag.OXE_AUGE_KUKA,
    "oxe_auge_xarm7": EmbodimentTag.OXE_AUGE_XARM7,
    # SO100 community data — joint-space
    "so100_single_6dof": EmbodimentTag.SO100_SINGLE,
    "so100_single_7dof": EmbodimentTag.SO100_SINGLE,
    "so100_dual_12dof": EmbodimentTag.SO100_DUAL,
    "so100_dual_14dof": EmbodimentTag.SO100_DUAL,
}
