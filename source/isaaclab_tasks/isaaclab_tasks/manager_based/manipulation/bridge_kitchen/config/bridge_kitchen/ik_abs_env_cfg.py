# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import joint_pos_env_cfg

##
# Pre-defined configs
from scripts.workflows.utils.robot_cfg import WIDOWX_CFG


@configclass
class UWBridgeKitchenEnvCfg(joint_pos_env_cfg.UWBridgeKitchenEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # Set Franka as robot
        # We switch here to a stiffer PD controller for IK tracking to be better.
        self.scene.robot = WIDOWX_CFG

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "wrist_rotate",
                "elbow",
                "shoulder",
                "forearm_roll",
                "waist",
                "wrist_angle",
                "gripper",
            ],
            body_name="wx250s_ee_gripper_link",
            controller=DifferentialIKControllerCfg(command_type="pose",
                                                   use_relative_mode=False,
                                                   ik_method="dls",
                                                   customize_delta_pose=False),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.0]),
        )


@configclass
class UWBridgeKitchenEnvCfg_PLAY(UWBridgeKitchenEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
