# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
import os
from isaaclab_tasks.manager_based.manipulation.cabinet import mdp

from scripts.workflows.utils.config_setting import set_from_config
from isaaclab.managers import ObservationTermCfg as ObsTerm
##
# Pre-defined configs
##
from isaaclab_assets import IMPLICIT_RIGHT_LEAP_XARM5, IMPLICIT_LEFT_LEAP_XARM5

from dataclasses import MISSING, Field, dataclass, field, replace
import isaaclab_tasks.manager_based.manipulation.inhand.inhand_env_cfg as inhand_env_cfg
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab_tasks.manager_based.manipulation.inhand.mdp as mdp
# from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.mdp.leap_xarm.reward_leap_xarm import LeapXarmApproachReward
import torch


@configclass
class XarmYCBEnvCfg(inhand_env_cfg.InHandObjectEnvCfg):
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

        # Set Actions for the specific robot type (franka)
        arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=[
                "joint1",
                "joint2",
                "joint3",
                "joint4",
                "joint5",
            ],
            # scale=1.0,
            use_default_offset=False,
            preserve_order=True)

        reset_robot_default_joints = EventTerm(
            func=mdp.set_hand_arm_joinit_pose,
            mode="reset",
            params={
                "joint_pos":
                torch.as_tensor([
                    -5.1955e-06, -4.0349e-02, -3.6284e-01, -1.1320e+00,
                    1.5718e+00, -7.5451e-07, -1.2334e-03, 1.9972e-05,
                    8.9328e-05, 8.7132e-04, -1.2139e-05, 3.0446e-04,
                    2.6327e-04, -5.6637e-06, -5.7940e-06, 4.5861e-07,
                    2.8385e-07, -5.6125e-11, -8.3479e-07, -5.7064e-08,
                    -5.6905e-08
                ]),
                "robot_name":
                "robot",
            },
        )

        if env_cfg["params"]["add_left_hand"]:
            setattr(
                self.scene, "left_hand",
                IMPLICIT_LEFT_LEAP_XARM5.replace(
                    prim_path="{ENV_REGEX_NS}/left_hand"))

            left_hand_action = mdp.JointPositionActionCfg(
                asset_name="left_hand",
                joint_names=[
                    "j0",
                    "j1",
                    "j2",
                    "j3",
                    "j4",
                    "j5",
                    "j6",
                    "j7",
                    "j8",
                    "j9",
                    "j10",
                    "j11",
                    "j12",
                    "j13",
                    "j14",
                    "j15",
                ],
            )
            setattr(self.actions, "left_hand_action", left_hand_action)
            arm_action.asset_name = "left_hand"
            setattr(self.actions, "left_arm_action", arm_action)
            # reset_robot_default_joints.params["robot_name"] = "left_hand"

            # setattr(self.events, "reset_left_robot_default_joints",
            #         reset_robot_default_joints)

        if env_cfg["params"]["add_right_hand"]:
            setattr(
                self.scene, "right_hand",
                IMPLICIT_RIGHT_LEAP_XARM5.replace(
                    prim_path="{ENV_REGEX_NS}/right_hand"))

            right_hand_action = mdp.JointPositionActionCfg(
                asset_name="right_hand",
                joint_names=[
                    "j0",
                    "j1",
                    "j2",
                    "j3",
                    "j4",
                    "j5",
                    "j6",
                    "j7",
                    "j8",
                    "j9",
                    "j10",
                    "j11",
                    "j12",
                    "j13",
                    "j14",
                    "j15",
                ],
            )
            setattr(self.actions, "right_hand_action", right_hand_action)
            arm_action.asset_name = "right_hand"
            setattr(self.actions, "right_arm_action", arm_action)
            # reset_robot_default_joints.params["robot_name"] = "right_hand"
            # setattr(self.events, "reset_right_robot_default_joints",
            #         reset_robot_default_joints)

        camera_setting = env_cfg["params"]["Camera"]
        # LeapXarmApproachReward(self.config_yaml, self.rewards)

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
class XarmYCBEnvCfg_PLAY(XarmYCBEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
