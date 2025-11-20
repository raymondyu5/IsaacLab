# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

from . import stack_joint_pos_env_cfg

##
# Pre-defined configs
##
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.configuration_robot import config_ik_arm


@configclass
class HandYCBEnvCfg(stack_joint_pos_env_cfg.HandYCBEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        config_ik_arm(self.config_yaml,
                      self.actions,
                      self.events,
                      use_relative_mode=True)


@configclass
class HandEnvCfg_PLAY(HandYCBEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
