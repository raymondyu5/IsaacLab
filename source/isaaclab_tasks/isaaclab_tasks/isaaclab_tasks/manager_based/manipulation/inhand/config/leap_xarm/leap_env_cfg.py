# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg
from scripts.workflows.utils.config_setting import set_from_config
##
# Pre-defined configs
##
from isaaclab_assets import Xarm_UR5_ARTICULATION
from dataclasses import dataclass, field
import os


@configclass
class XarmCubeEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    config_yaml: str = field(default=None)

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        env_cfg = self.config_yaml
        self.env_config = env_cfg
        current_path = os.getcwd()
        set_from_config(self.scene, env_cfg, current_path)

        # switch robot to Xarm hand
        self.scene.robot = Xarm_UR5_ARTICULATION.replace(
            prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class XarmCubeEnvCfg_PLAY(XarmCubeEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None


##
# Environment configuration with no velocity observations.
##


@configclass
class XarmCubeNoVelObsEnvCfg(XarmCubeEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # switch observation group to no velocity group
        self.observations.policy = inhand_env_cfg.ObservationsCfg.NoVelocityKinematicObsGroupCfg(
        )


@configclass
class XarmCubeNoVelObsEnvCfg_PLAY(XarmCubeNoVelObsEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove termination due to timeouts
        self.terminations.time_out = None
