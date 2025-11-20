# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg, DeformableObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, DeformableBodyPropertiesCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.lift_cube import mdp
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.lift_cube.lift_env_cfg import LiftEnvCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip
import isaaclab.sim as sim_utils
import os
from . import agents
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import yaml
from isaaclab.sensors import TiledCameraCfg, ContactSensorCfg
##
# Scene definition
##
from isaaclab.sensors.camera import Camera, CameraCfg
import isaaclab.utils.math as math_utils
import torch
import math
from isaaclab.assets.articulation import ArticulationCfg

import sys

sys.path.append(".")
from scripts.workflows.utils.config_setting import set_from_config

camera_list = {"camera": CameraCfg, "tiled_camera": TiledCameraCfg}


def obtain_target_quat_from_multi_angles(axis, angles):
    quat_list = []
    for index, cam_axis in enumerate(axis):
        euler_xyz = torch.zeros(3)
        euler_xyz[cam_axis] = angles[index]
        quat_list.append(
            math_utils.quat_from_euler_xyz(euler_xyz[0], euler_xyz[1],
                                           euler_xyz[2]))
    if len(quat_list) == 1:
        return quat_list[0]
    else:
        target_quat = quat_list[0]
        for index in range(len(quat_list) - 1):

            target_quat = math_utils.quat_mul(quat_list[index + 1],
                                              target_quat)
        return target_quat


from dataclasses import dataclass, field


@configclass
class FrankaDeformCubeLiftEnvCfg(LiftEnvCfg):

    # Add config_yaml as an optional field
    config_yaml: str = field(default=None)

    def __post_init__(self, config_yaml=None):
        # post init of parent
        current_path = os.getcwd()

        with open(self.config_yaml, 'r') as file:
            env_cfg = yaml.safe_load(file)
        setattr(self, "general_setting", env_cfg["params"]["General"])

        super().__post_init__()

        set_from_config(self.scene, env_cfg, current_path)

        gripper_id = 0
        for rigid_object_name in env_cfg["params"]["RigidObject"].keys():

            if "gripper" in rigid_object_name:
                if env_cfg["params"]["RigidObject"][rigid_object_name]["init"]:
                    if env_cfg["params"]["RigidObject"][rigid_object_name][
                            "motion"]:
                        setattr(self.actions,
                                f"float_gripper_action_{gripper_id}",
                                f"float_gripper_action_{gripper_id}")
                        setattr(
                            self.actions, f"float_gripper_action_{gripper_id}",
                            mdp.RigidPositionActionCfg(
                                asset_name=rigid_object_name,
                                debug_config={
                                    "attachment_type":
                                    env_cfg["params"]["RigidObject"]
                                    [rigid_object_name]["attachment_type"]
                                }))

                        gripper_id += 1

        if env_cfg["params"]["Robot"]["init"]:
            # Set Franka as robot
            setattr(self.scene, "robot", "robot")
            self.scene.robot = FRANKA_PANDA_CFG.replace(
                prim_path="{ENV_REGEX_NS}/Robot")
            self.scene.robot.articulation_cfg["robot_setting"] = env_cfg[
                "params"]["Robot"]["robot_setting"]

            # Set actions for the specific robot type (franka)
            setattr(self.actions, "arm_action", "arm_action")
            self.actions.arm_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                scale=0.5,
                use_default_offset=True)
            setattr(self.actions, "gripper_action", "gripper_action")
            self.actions.gripper_action = mdp.RelativeJointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_finger.*"],
                scale=0.1,
                # use_default_offset=True
                # open_command_expr={"panda_finger_.*": 0.04},
                # close_command_expr={"panda_finger_.*": 0.0},
            )
            # setattr(self.scene, "ee_frame", "ee_frame")
            # Listens to the required transforms
            marker_cfg = FRAME_MARKER_CFG.copy()
            marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
            marker_cfg.prim_path = "/Visuals/FrameTransformer"
            self.scene.ee_frame = FrameTransformerCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
                debug_vis=False,
                visualizer_cfg=marker_cfg,
                target_frames=[
                    FrameTransformerCfg.FrameCfg(
                        prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                        name="end_effector",
                        offset=OffsetCfg(pos=[0.0, 0.0, 0.1034], ),
                    ),
                ],
            )
            # Make the end effector less stiff to not hurt the poor teddy bear
            self.scene.robot.actuators["panda_hand"].effort_limit = 50.0
            self.scene.robot.actuators["panda_hand"].stiffness = 40.0
            self.scene.robot.actuators["panda_hand"].damping = 10.0


@configclass
class FrankaDeformCubeLiftEnvCfg_PLAY(FrankaDeformCubeLiftEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 128
        self.scene.env_spacing = 10
        # disable randomization for play
        self.observations.policy.enable_corruption = False
