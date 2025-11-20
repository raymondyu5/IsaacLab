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
from isaaclab.utils.math import obtain_target_quat_from_multi_angles

import numpy as np

import copy


class MultiRootStateCfg():

    def __init__(self, object_name, rigid_object_cfg, num_envs):
        self.object_name = object_name

        self.reset_height = []

        self.reset_height = torch.zeros((num_envs)).to("cuda")
        env_ids = torch.arange(num_envs).to("cuda")
        object_id = 0
        self.default_root_state = torch.zeros((num_envs, 7)).to("cuda")
        self.default_scale = torch.zeros((num_envs, 3)).to("cuda")

        for object_name, object_config in rigid_object_cfg.items():
            mask = (env_ids % len(rigid_object_cfg)) == object_id
            object_height = object_config["pos"][2]
            self.reset_height[mask] = object_height
            rot = torch.as_tensor(
                tuple(
                    obtain_target_quat_from_multi_angles(
                        object_config["rot"]["axis"],
                        object_config["rot"]["angles"])))
            pos = torch.as_tensor(object_config['pos'], dtype=torch.float32)
            pos = torch.round(pos * 100) / 100
            self.default_scale[mask] = torch.as_tensor(
                object_config.get("scale",
                                  [1, 1, 1]), dtype=torch.float32).to("cuda")

            self.default_root_state[mask] = torch.cat(
                (pos, rot), dim=-1).to("cuda").to(dtype=torch.float32)

            object_id += 1

        self.default_scale = np.round(self.default_scale.cpu().numpy() * 0.3,
                                      2).tolist()
        self.default_scale = [[round(x, 2) for x in scale]
                              for scale in self.default_scale]

        pose_range = object_config.get("pose_range", {})
        self.default_pose_range = np.array([
            pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ])
        self.pose_range = copy.deepcopy(self.default_pose_range)

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
        asset.data.reset_root_state[..., :7] = self.default_root_state.clone()

        root_states = asset.data.default_root_state[env_ids].clone()

        # poses
        range_list = copy.deepcopy(self.pose_range)

        ranges = torch.tensor(range_list, device=env.device)
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

        # set into the physics simulation

        asset.write_root_pose_to_sim(target_state[:, :7], env_ids=env_ids)
        asset.write_root_velocity_to_sim(target_state[:, 7:], env_ids=env_ids)

        return positions, orientations


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


def configure_multi_rigid_obs(object, env_cfg):

    num_rigid_objects = len(list(env_cfg["params"]["RigidObject"].keys()))
    num_envs = env_cfg["params"].get("num_envs", object.scene.num_envs)
    object_name = env_cfg["params"]["spawn_multi_assets_name"]

    multi_state = MultiRootStateCfg(object_name,
                                    env_cfg["params"]["RigidObject"], num_envs)
    reset_func = EventTerm(func=multi_state.reset_multi_root_state_uniform,
                           mode="reset",
                           params={})

    setattr(object.events, f"reset_{object_name}", reset_func)


def configure_rigid_obs(object, env_cfg):

    rigid_objects = env_cfg['params'].get('RigidObject', {})

    if rigid_objects is not None:
        for object_name, object_config in rigid_objects.items():

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
