# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg

from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg, DeformableBodyPropertiesCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform

from isaaclab.sim.spawners import GroundPlaneCfg, UsdFileCfg
import isaaclab.sim as sim_utils
from isaaclab_assets import FRANKA_PANDA_CFG  # isort: skip

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.sensors.camera import Camera, CameraCfg
import os
import yaml
import isaaclab.utils.math as math_utils
from isaaclab.sensors.camera.batch_utils import create_pointcloud_from_rgbd_batch
from isaacsim.core.utils.extensions import enable_extension

enable_extension("omni.isaac.robot_assembler")
from omni.isaac.robot_assembler import RobotAssembler, AssembledRobot


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


def load_yaml(current_path, yaml_path):
    with open(current_path + "/" + yaml_path, 'r') as file:
        object_cfg = yaml.safe_load(file)

    deformable_objects_settings = None
    rigid_objects_settings = None
    camera_settings = None

    if object_cfg['params']['DeformableObject'] is not None:
        deformable_objects_settings = {}

        for deform_object in object_cfg['params']['DeformableObject'].keys():

            deform_object_config = object_cfg['params']['DeformableObject'][
                deform_object]
            deformable_object_cfg = DeformableObjectCfg(
                prim_path="/World/envs/env_.*/" + deform_object,
                init_state=DeformableObjectCfg.InitialStateCfg(
                    pos=deform_object_config['pos'],
                    rot=tuple(
                        obtain_target_quat_from_multi_angles(
                            deform_object_config["rot"]["axis"],
                            deform_object_config["rot"]["angles"]))),
                spawn=UsdFileCfg(
                    usd_path=f"{current_path}/" + deform_object_config['path'],
                    scale=deform_object_config['scale'],
                    deformable_props=DeformableBodyPropertiesCfg(),
                ),
                deform_cfg=deform_object_config)
            deformable_objects_settings[deform_object] = deformable_object_cfg

    if object_cfg['params']['RigidObject'] is not None:
        rigid_objects_settings = {}
        for rigid_object in object_cfg['params']['RigidObject'].keys():

            rigid_object_config = object_cfg['params']['RigidObject'][
                rigid_object]
            rigid_object_cfg = RigidObjectCfg(
                prim_path="/World/envs/env_.*/" + rigid_object,
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=rigid_object_config['pos'],
                    rot=tuple(
                        obtain_target_quat_from_multi_angles(
                            rigid_object_config["rot"]["axis"],
                            rigid_object_config["rot"]["angles"]))),
                spawn=UsdFileCfg(
                    usd_path=f"{current_path}/" + rigid_object_config['path'],
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
            rigid_objects_settings[rigid_object] = rigid_object_cfg

    # inital camra
    if object_cfg['params']['Camera']["initial"]:
        camera_settings = {}
        for camera_name in object_cfg['params']['Camera']["cameras"].keys():

            camera_setting = object_cfg['params']['Camera']["cameras"][
                camera_name]
            camera_cfg = CameraCfg(
                prim_path="/World/envs/env_.*/" + camera_name,
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
                spawn=sim_utils.PinholeCameraCfg(clipping_range=(0.1, 1000)))

            camera_settings[camera_name] = camera_cfg

    return deformable_objects_settings, rigid_objects_settings, camera_settings
