# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import stack_joint_pos_instance_randomize_env_cfg
from dataclasses import dataclass, field
##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip
import os


@configclass
class FrankaCubeStackInstanceRandomizeEnvCfg(
        stack_joint_pos_instance_randomize_env_cfg.
        FrankaCubeStackInstanceRandomizeEnvCfg):
    config_yaml: str = field(default=None)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(command_type="pose",
                                                   use_relative_mode=False,
                                                   ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.0]),
        )
