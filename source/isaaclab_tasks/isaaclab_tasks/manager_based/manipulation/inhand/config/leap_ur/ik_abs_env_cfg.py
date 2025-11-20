# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG  # isort: skip


@configclass
class LeapYCBEnvCfg(joint_pos_env_cfg.LeapYCBEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
            ],
            body_name="wrist_3_link",
            controller=DifferentialIKControllerCfg(command_type="pose",
                                                   use_relative_mode=False,
                                                   ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.0]),
        )


@configclass
class LeapCubeEnvCfg_PLAY(LeapYCBEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
