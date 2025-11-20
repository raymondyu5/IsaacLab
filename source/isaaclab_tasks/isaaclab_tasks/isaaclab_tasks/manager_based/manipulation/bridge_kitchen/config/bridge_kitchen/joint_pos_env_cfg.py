# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.cabinet import mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.bridge_kitchen.bridge_uw_kitchen_cfg import BridgeKitchenUWCfg
##
# Pre-defined configs
##
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip
from dataclasses import dataclass, field
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.assets import RigidObjectCfg
import sys
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG

sys.path.append(".")
from scripts.workflows.utils.config_setting import set_from_config
import os
from isaaclab.envs.mdp.actions.utils.widowx_reset_ik import WidowXResetIK
import torch
from scripts.workflows.utils.robot_cfg import WIDOWX_CFG
from isaaclab.markers.config import FRAME_MARKER_CFG
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.cabinet.cabinet_env_cfg import (  # isort: skip
    FRAME_MARKER_SMALL_CFG)

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.stack.mdp.obs_reward_buffer import RewardObsBuffer


@configclass
class UWBridgeKitchenEnvCfg(BridgeKitchenUWCfg):
    config_yaml: str = field(default=None)

    def __post_init__(self):
        # with open(self.config_yaml, 'r') as file:
        #     env_cfg = yaml.safe_load(file)
        env_cfg = self.config_yaml
        self.env_config = env_cfg

        # post init of parent
        super().__post_init__()
        # self.config_yaml = env_cfg
        current_path = os.getcwd()

        self.scene.robot = WIDOWX_CFG
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.03, 0.03, 0.03)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/robot/wx250s/wx250s_base_link",
            debug_vis=True,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=
                    "{ENV_REGEX_NS}/robot/wx250s/wx250s_ee_gripper_link",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0],
                                     rot=[0.0, -0.7, 0.0, 0.7]),
                ),
            ],
        )

        # # Set franka as robot
        # self.scene.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(
        #     prim_path="{ENV_REGEX_NS}/Robot")

        # Set Actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
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
            # scale=1.0,
            use_default_offset=False,
            preserve_order=True)
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["left_finger", "right_finger"],
            open_command_expr={
                "left_finger": 0.036,
                "right_finger": -0.036,
            },
            close_command_expr={
                "left_finger": 0.020,
                "right_finger": -0.020,
            },
        )

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
        set_from_config(self.scene, env_cfg, current_path)
        if env_cfg["params"]["GS_Camera"]["initial"]:
            from scripts.workflows.utils.gs_env.multi_gs_env import MultiGaussianEnv
            multi_gs_group = MultiGaussianEnv(env_config=env_cfg)
            setattr(self.observations.policy, "gs_image", "gs_image")

            self.observations.policy.gs_image = ObsTerm(
                func=multi_gs_group.extract_gaussians)

        setattr(self.events, "reset_robot", "reset_robot")
        reset_robot_function = WidowXResetIK(env_cfg)
        self.events.reset_robot = EventTerm(
            func=reset_robot_function.reset_ee_link_pose, mode="reset")
        for reset_object_name in env_cfg["params"]["Task"][
                "reset_object_names"]:

            reset_object = EventTerm(func=mdp.reset_rigid_articulation,
                                     params={
                                         "target_name":
                                         reset_object_name,
                                         "pose_range":
                                         env_cfg["params"]["RigidObject"]
                                         [reset_object_name]["pose_range"],
                                     },
                                     mode="reset")
            setattr(self.events, f"reset_{reset_object_name}", reset_object)

        if env_cfg["params"]["RL_Train"]["initial"]:
            for rigid_name in env_cfg["params"]["RL_Train"][
                    "rigid_object_names"]:
                state_info = ObsTerm(mdp.get_root_state,
                                     params={"name": rigid_name})
                setattr(self.observations.policy, f"{rigid_name}_state",
                        state_info)

        elif env_cfg["params"]["GS_Camera"]["initial"]:

            for rigid_name in env_cfg["params"]["RigidObject"].keys():
                state_info = ObsTerm(mdp.get_root_state,
                                     params={"name": rigid_name})
                setattr(self.observations.policy, f"{rigid_name}_state",
                        state_info)

        target_lift_object_pos = torch.as_tensor(
            env_cfg["params"]["Task"]["target_lift_object_pos"])
        target_lift_object_pos_function = ObsTerm(
            mdp.target_lift_object_pos,
            params={"target_lift_object_pos": target_lift_object_pos})
        setattr(self.observations.policy, "target_lift_object_pos",
                target_lift_object_pos_function)

        target_object_pose = mdp.UniformPoseCommandCfg(
            asset_name="robot",
            body_name="wx250s_ee_gripper_link",  # will be set by agent env cfg
            resampling_time_range=(20, 20),
            debug_vis=True,
            ranges=mdp.UniformPoseCommandCfg.Ranges(pos_x=(0.26, 0.38),
                                                    pos_y=(0.08, 0.13),
                                                    pos_z=(0.18, 0.25),
                                                    roll=(0.0, 0.0),
                                                    pitch=(0.0, 0.0),
                                                    yaw=(0.0, 0.0)),
        )
        setattr(self.commands, "target_object_pose", target_object_pose)

        RewardObsBuffer(self.rewards,
                        self.observations.policy,
                        self.config_yaml,
                        target_object_name="eggplant",
                        ee_frame_name="wx250s_ee_gripper_link",
                        left_finger_name="wx250s_left_finger_link",
                        right_finger_name="wx250s_right_finger_link",
                        use_gripper_offset=False)

        # if env_cfg["params"]["Task"]["use_residual"]:
        #     from scripts.workflows.open_policy.utils.residual_buffer import ResidualBuffer
        #     ResidualBuffer(self.observations.policy, self.commands,
        #                    env_cfg["params"]["Task"]["base_policy"])


@configclass
class UWBridgeKitchenEnvCfg_PLAY(UWBridgeKitchenEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 0.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
