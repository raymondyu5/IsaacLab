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

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.Tube import mdp
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.Tube.lift_env_cfg import LiftEnvCfg
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


@configclass
class FrankaTubeLiftEnvCfg(LiftEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        current_path = os.getcwd()

        with open(
                f"{current_path}/source/extensions/isaaclab_tasks/omni/isaac/lab_tasks/manager_based/manipulation/Tube/config/franka/agents/env_random.yaml",
                'r') as file:
            env_cfg = yaml.safe_load(file)

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot").replace(
                init_state=ArticulationCfg.InitialStateCfg(
                    joint_pos=env_cfg['params']["Robot"]["qpos"], ))

        #Set actions for the specific robot type (franka)
        self.actions.body_joint_pos = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            scale=0.5,
            use_default_offset=True)

        self.actions.finger_joint_pos = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        if env_cfg['params']['DeformableObject'] is not None:
            for deform_object in env_cfg['params']['DeformableObject'].keys():

                setattr(self.scene, deform_object, deform_object)
                deform_object_config = env_cfg['params']['DeformableObject'][
                    deform_object]
                deformable_env_cfg = DeformableObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + deform_object,
                    init_state=DeformableObjectCfg.InitialStateCfg(
                        pos=deform_object_config['pos'],
                        rot=tuple(
                            obtain_target_quat_from_multi_angles(
                                deform_object_config["rot"]["axis"],
                                deform_object_config["rot"]["angles"]))),
                    spawn=UsdFileCfg(
                        usd_path=f"{current_path}/" +
                        deform_object_config['path'],
                        scale=deform_object_config['scale'],
                        deformable_props=DeformableBodyPropertiesCfg(
                            simulation_hexahedral_resolution=
                            deform_object_config[
                                "simulation_hexahedral_resolution"],
                            vertex_velocity_damping=deform_object_config[
                                "vertex_velocity_damping"],
                            self_collision_filter_distance=0.001),
                    ),
                    deform_cfg=deform_object_config)
                setattr(self.scene, deform_object, deformable_env_cfg)

        if env_cfg['params']['RigidObject'] is not None:
            for rigid_object in env_cfg['params']['RigidObject'].keys():

                setattr(self.scene, rigid_object, rigid_object)

                rigid_object_config = env_cfg['params']['RigidObject'][
                    rigid_object]
                rigid_env_cfg = RigidObjectCfg(
                    prim_path="{ENV_REGEX_NS}/" + rigid_object,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=rigid_object_config['pos'],
                        rot=tuple(
                            obtain_target_quat_from_multi_angles(
                                rigid_object_config["rot"]["axis"],
                                rigid_object_config["rot"]["angles"]))),
                    spawn=UsdFileCfg(
                        usd_path=f"{current_path}/" +
                        rigid_object_config['path'],
                        scale=rigid_object_config['scale'],
                        rigid_props=RigidBodyPropertiesCfg(
                            solver_position_iteration_count=16,
                            solver_velocity_iteration_count=1,
                            max_angular_velocity=1000.0,
                            max_linear_velocity=1000.0,
                            max_depenetration_velocity=5.0,
                            disable_gravity=False,
                        ),
                    ),
                )
                setattr(self.scene, rigid_object, rigid_env_cfg)

        # inital camra
        if env_cfg['params']['Camera']["initial"]:
            for cameare_name in env_cfg['params']['Camera']["cameras"].keys():
                setattr(self.scene, cameare_name, cameare_name)
                camera_setting = env_cfg['params']['Camera']["cameras"][
                    cameare_name]
                camera_cfg = CameraCfg(
                    prim_path="{ENV_REGEX_NS}/" + cameare_name,
                    offset=CameraCfg.OffsetCfg(
                        pos=camera_setting["pos"],
                        rot=tuple(
                            obtain_target_quat_from_multi_angles(
                                camera_setting["rot"]["axis"],
                                camera_setting["rot"]["angles"]))),
                    update_period=camera_setting["update_period"],
                    height=camera_setting["height"],
                    width=camera_setting["width"],
                    data_types=camera_setting["data_types"],
                    spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1,
                                                                     1000)))
                setattr(self.scene, cameare_name, camera_cfg)

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


@configclass
class FrankaTubeLiftEnvCfg_PLAY(FrankaTubeLiftEnvCfg):

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 1
        self.scene.env_spacing = 5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
