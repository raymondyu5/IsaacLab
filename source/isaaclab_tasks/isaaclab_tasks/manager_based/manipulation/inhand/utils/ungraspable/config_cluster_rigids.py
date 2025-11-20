from isaaclab.managers import ObservationTermCfg as ObsTerm

import os

import isaaclab_tasks.manager_based.manipulation.inhand.mdp as mdp

from isaaclab.managers import EventTermCfg as EventTerm

import torch
import warnings
from typing import TYPE_CHECKING, Literal
from isaaclab.actuators import ImplicitActuator
from isaaclab.assets import Articulation, DeformableObject, RigidObject
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.terrains import TerrainImporter
import random
import math
from pxr import Gf, Sdf, UsdGeom, Vt
import isaaclab.sim as sim_utils
import omni.physics.tensors.impl.api as physx
import omni.usd
if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
import isaaclab.utils.math as math_utils
from isaaclab.sensors.camera.utils import obtain_target_quat_from_multi_angles

import numpy as np


class MultiRootStateCfg():

    def __init__(self,
                 object_name,
                 rigid_object_cfg,
                 object_config,
                 num_envs,
                 distractor_name=None):
        self.object_name = object_name

        self.reset_height = []

        self.reset_height = torch.zeros((num_envs)).to("cuda")
        env_ids = torch.arange(num_envs).to("cuda")

        self.default_root_state = torch.zeros((num_envs, 7)).to("cuda")
        self.default_scale = torch.zeros((num_envs, 3)).to("cuda")
        num_object = len(object_config["objects_list"])

        if object_config["name"] not in ["right_hand_object"]:
            return

        for index, object_name in enumerate(object_config["objects_list"]):

            if object_name in distractor_name:
                continue
            mask = (env_ids % num_object) == index
            object_height = rigid_object_cfg[object_name]["pos"][2]
            self.reset_height[mask] = object_height
            rot = torch.as_tensor(
                tuple(
                    obtain_target_quat_from_multi_angles(
                        object_config["rot"]["axis"],
                        object_config["rot"]["angles"])))
            pos = torch.as_tensor(object_config['pos'], dtype=torch.float32)

            self.default_scale[mask] = torch.as_tensor(
                object_config.get("scale", [1, 1, 1])).to("cuda")

            self.default_root_state[mask] = torch.cat(
                (pos, rot), dim=-1).to("cuda").to(dtype=torch.float32)
            self.default_root_state[mask, 2] = object_height

        self.pose_range = object_config.get("pose_range", {})

        self.velocity_range = object_config.get("velocity_range", {})
        self.init_scale = False

    def reset_default_scale(self, env, env_ids, relative_child_path=None):

        asset: RigidObject = env.scene[self.object_name]
        env_ids = env_ids.cpu()
        if relative_child_path is None:
            relative_child_path = ""
        elif not relative_child_path.startswith("/"):
            relative_child_path = "/" + relative_child_path

        # use sdf changeblock for faster processing of USD properties

        stage = omni.usd.get_context().get_stage()
        # resolve prim paths for spawning and cloning
        prim_paths = sim_utils.find_matching_prim_paths(asset.cfg.prim_path)
        with Sdf.ChangeBlock():
            for i, env_id in enumerate(env_ids):
                # path to prim to randomize
                prim_path = prim_paths[env_id] + relative_child_path
                # spawn single instance
                prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(),
                                                  prim_path)

                # get the attribute to randomize
                scale_spec = prim_spec.GetAttributeAtPath(prim_path +
                                                          ".xformOp:scale")
                # if the scale attribute does not exist, create it
                has_scale_attr = scale_spec is not None
                if not has_scale_attr:
                    scale_spec = Sdf.AttributeSpec(
                        prim_spec, prim_path + ".xformOp:scale",
                        Sdf.ValueTypeNames.Double3)

                # set the new scale

                scale_spec.default = Gf.Vec3f(self.default_scale[i])

    def reset_multi_root_state_uniform(
        self,
        env,
        env_ids,
    ):

        asset = env.scene[self.object_name]
        # get default root state

        asset.data.default_root_state[
            ..., :7] = self.default_root_state.clone()

        root_states = asset.data.default_root_state[env_ids].clone()
        root_states[:,
                    2] += env.scene["table_block"]._data.root_state_w[:,
                                                                      2].clone(
                                                                      )

        # poses
        range_list = [
            self.pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]

        ranges = torch.tensor(range_list, device='cuda:0')
        rand_samples = math_utils.sample_uniform(ranges[:, 0],
                                                 ranges[:, 1],
                                                 (len(env_ids), 6),
                                                 device=asset.device)

        positions = root_states[:, 0:3] + env.scene.env_origins[
            env_ids] + rand_samples[:, 0:3]

        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(orientations_delta,
                                           root_states[:, 3:7])

        # velocities
        range_list = [
            self.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0],
                                                 ranges[:, 1],
                                                 (len(env_ids), 6),
                                                 device=asset.device)

        velocities = root_states[:, 7:13] + rand_samples

        target_state = torch.cat([positions, orientations, velocities], dim=-1)

        target_state[:, 2] = self.reset_height.to(env.device)[env_ids].clone()

        target_state[:, 2] += env.scene[
            "table_block"].data.reset_root_state[:, 2] + 0.05

        table_block = env.scene["table_block"]

        table_block_state = table_block.data.default_root_state.clone()
        table_block_state[:, :2] = target_state[:, :2].clone()
        table_block.write_root_pose_to_sim(table_block_state[:, :7],
                                           env_ids=env_ids)
        table_block.write_root_velocity_to_sim(table_block_state[:, 7:],
                                               env_ids=env_ids)

        # set into the physics simulation
        asset.data.reset_root_state[..., :7] = target_state[:, :7].clone()

        asset.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(target_state[:, 7:], env_ids=env_ids)

        return positions, orientations

    def reset_multi_root_target_uniform(self, env, env_ids, target_pose_cfg):

        asset = env.scene[self.object_name]
        # get default root state

        asset.data.default_root_state[
            ..., :7] = self.default_root_state.clone()
        # asset.data.reset_root_state[..., :7] = self.default_root_state.clone()

        root_states = asset.data.default_root_state[env_ids].clone()

        root_states[:, 2] = target_pose_cfg["pos"][2]

        # poses
        range_list = [
            target_pose_cfg["pose_range"].get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]

        ranges = torch.tensor(range_list, device='cuda:0')
        rand_samples = math_utils.sample_uniform(ranges[:, 0],
                                                 ranges[:, 1],
                                                 (len(env_ids), 6),
                                                 device=asset.device)

        positions = root_states[:, 0:3] + rand_samples[:, 0:3]

        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        # orientations = math_utils.quat_mul(orientations_delta,
        #                                    root_states[:, 3:7])
        # # velocities
        range_list = [
            self.velocity_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0],
                                                 ranges[:, 1],
                                                 (len(env_ids), 6),
                                                 device=asset.device)

        orientation = math_utils.quat_mul(
            orientations_delta,
            asset.data.reset_root_state[..., 3:7].clone(),
        )

        # delta_quat = math_utils.quat_mul(
        #     orientation,
        #     math_utils.quat_inv(asset.data.reset_root_state[..., 3:7].clone()))
        # print(math_utils.euler_xyz_from_quat(delta_quat)[2])

        setattr(asset.data, "target_state",
                torch.cat([positions, orientation], dim=-1))

        return positions, orientation


def get_rigid_state(env, object_name):
    """
    Get the rigid state of the specified object in the environment.
    """
    object_state = env.scene[object_name]._data.root_state_w[:, :7]
    object_state[:, :3] -= env.scene.env_origins
    return object_state


def configure_reset_rigid_obs(object, object_name, env_cfg):

    reset = EventTerm(
        func=mdp.reset_rigid_articulation,
        mode="reset",
        params={
            "target_name":
            object_name,
            "pose_range":
            env_cfg["params"]["RigidObject"][object_name]["pose_range"]
        })
    setattr(object.events, f"reset_{object_name}", reset)


def configure_multi_cluster_rigid_obs(object, env_cfg):

    multi_cluster_object = env_cfg["params"]["multi_cluster_rigid"]
    num_envs = env_cfg["params"].get("num_envs", object.scene.num_envs)
    target_pose_cfg = env_cfg["params"]["target_manipulated_object_pose"]
    distractor_name = env_cfg["params"].get("distractor_name", [])

    if len(distractor_name) > 0:
        for ditractor in distractor_name:
            reset_func = EventTerm(
                func=mdp.reset_rigid_articulation,
                mode="reset",
                params={
                    "target_name":
                    ditractor,
                    "pose_range":
                    env_cfg["params"]["RigidObject"][ditractor]["pose_range"]
                })
            setattr(object.events, f"reset_{ditractor}", reset_func)

    for object_name, object_config in multi_cluster_object.items():

        handness = object_config.get("handness", None)
        if not env_cfg["params"][f"add_{handness}_hand"]:
            continue

        multi_state = MultiRootStateCfg(object_name,
                                        env_cfg["params"]["RigidObject"],
                                        object_config, num_envs,
                                        distractor_name)
        reset_func = EventTerm(func=multi_state.reset_multi_root_state_uniform,
                               mode="reset",
                               params={})
        reset_target_pose = EventTerm(
            func=multi_state.reset_multi_root_target_uniform,
            mode="reset",
            params={"target_pose_cfg": target_pose_cfg})
        #TODO: need to reset color

        # randomize_color = EventTerm(
        #     func=mdp.randomize_visual_color,
        #     mode="reset",
        #     params={
        #         "colors": {
        #             "r": (0.0, 1.0),
        #             "g": (0.0, 1.0),
        #             "b": (0.0, 1.0)
        #         },
        #         "asset_cfg": SceneEntityCfg(object_name),
        #         "mesh_name": "geometry/mesh",
        #         "event_name": f"rep_{object_name}_randomize_color",
        #     },
        # )
        # setattr(object.events, f"rep_{object_name}_randomize_color",
        #         randomize_color)

        setattr(object.events, f"reset_{object_name}", reset_func)
        setattr(object.events, f"reset_target_pose", reset_target_pose)

        ### object material

        object_physics_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            min_step_count_between_reset=800,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(object_name),
                "static_friction_range": (0.5, 2.5),
                "dynamic_friction_range": (0.5, 2.5),
                "restitution_range": (1.0, 1.0),
                "num_buckets": 1250,
            },
        )
        setattr(object.events, f"reset_{object_name}_material",
                object_physics_material)

        object_scale_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            min_step_count_between_reset=800,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg(object_name),
                "mass_distribution_params": (0.1, 1.0),
                "operation": "scale",
                "distribution": "uniform",
            },
        )
        setattr(object.events, f"reset_{object_name}_material",
                object_scale_mass)


def configure_rigid_obs(object, env_cfg):

    rigid_objects = env_cfg['params'].get('RigidObject', {})

    if rigid_objects is not None:
        for object_name, object_config in rigid_objects.items():
            import pdb
            pdb.set_trace()

            if object_config.get("load_from_folder", False):

                object_names = os.listdir(object_config["load_from_folder"])
                for name in object_names:
                    object_state = ObsTerm(func=get_rigid_state,
                                           params={"object_name": name})
                    setattr(object.observations.policy, name, object_state)
                    # configure_reset_rigid_obs(object, name, env_cfg)

            else:
                object_state = ObsTerm(func=get_rigid_state,
                                       params={"object_name": object_name})
                setattr(object.observations.policy, object_name, object_state)
                configure_reset_rigid_obs(object, object_name, env_cfg)
