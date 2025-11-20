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
class XarmYCBEnvCfg(joint_pos_env_cfg.XarmYCBEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        diffik = DifferentialInverseKinematicsActionCfg(
            asset_name="left_hand",
            joint_names=["joint.*"],
            body_name="base",
            controller=DifferentialIKControllerCfg(command_type="pose",
                                                   use_relative_mode=False,
                                                   ik_method="dls"),
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                pos=[0.0, 0.0, 0.0]),
        )
        if self.config_yaml["params"]["add_left_hand"]:

            left_diffik = diffik.copy()
            left_diffik.asset_name = "left_hand"
            self.actions.left_arm_action = left_diffik

        if self.config_yaml["params"]["add_right_hand"]:

            right_diffik = diffik.copy()
            right_diffik.asset_name = "right_hand"
            self.actions.right_arm_action = right_diffik


@configclass
class XarmCubeEnvCfg_PLAY(XarmYCBEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
