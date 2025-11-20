# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
import os

from isaaclab.envs import mdp
from scripts.workflows.utils.config_setting import set_from_config
from isaaclab.managers import ObservationTermCfg as ObsTerm
##
# Pre-defined configs
##
from isaaclab_assets import IMPLICIT_LEFT_ABILIRY, IMPLICIT_RIGHT_ABILITY, ABILITY_HAND_ACTION

from dataclasses import MISSING, Field, dataclass, field, replace
import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg


@configclass
class AbilityYCBEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
    config_yaml: str = field(default=None)

    def __post_init__(self):
        # with open(self.config_yaml, 'r') as file:
        #     env_cfg = yaml.safe_load(file)
        env_cfg = self.config_yaml

        # post init of parent
        super().__post_init__()
        # self.config_yaml = env_cfg
        current_path = os.getcwd()

        set_from_config(self.scene, env_cfg, current_path)

        if env_cfg["params"]["add_left_hand"]:
            setattr(
                self.scene, "left_hand",
                IMPLICIT_LEFT_ABILIRY.replace(
                    prim_path="{ENV_REGEX_NS}/left_hand"))
            left_hand_action = ABILITY_HAND_ACTION.replace(
                asset_name="left_hand", )
            setattr(self.actions, "left_hand_action", left_hand_action)

        if env_cfg["params"]["add_right_hand"]:
            setattr(
                self.scene, "right_hand",
                IMPLICIT_RIGHT_ABILITY.replace(
                    prim_path="{ENV_REGEX_NS}/right_hand"))

            right_hand_action = ABILITY_HAND_ACTION.replace(
                asset_name="right_hand", )
            setattr(self.actions, "right_hand_action", right_hand_action)

        camera_setting = env_cfg["params"]["Camera"]

        if camera_setting["initial"]:
            setattr(self.observations.policy, "camera_obs", "camera_obs")
            self.observations.policy.camera_obs = ObsTerm(
                func=mdp.process_camera_data,
                params={
                    "whole_rgb": camera_setting["whole_rgb"],
                    "seg_rgb": camera_setting["seg_rgb"],
                    "whole_pc": camera_setting["whole_pc"],
                    "seg_pc": camera_setting["seg_pc"],
                    "bbox": camera_setting.get("bbox"),
                    "segmentation_name": camera_setting["segmentation_name"],
                    "max_length": camera_setting["max_length"],
                    "align_robot_base": camera_setting["align_robot_base"],
                    "whole_depth": camera_setting["whole_depth"],
                },
            )

        camera_setting = env_cfg["params"]["Camera"]


@configclass
class AbilityYCBEnvCfg_PLAY(AbilityYCBEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
