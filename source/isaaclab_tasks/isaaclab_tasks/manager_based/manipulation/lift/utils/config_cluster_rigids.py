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

import random
import math


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    pick_range: dict[str, tuple[float, float]] = {},
    place_range: dict[str, tuple[float, float]] = {},
    max_sample_tries: int = 5000,
):
    pick_range_list = [
        pick_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    place_range_list = [
        place_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    range_list = [pick_range_list, place_range_list]
    pose_list = []

    for i in range(num_objects):
        sample_pose = None

        for _ in range(max_sample_tries):
            sample = [random.uniform(r[0], r[1]) for r in range_list[i]]

            # accept first pose unconditionally
            if len(pose_list) == 0:
                sample_pose = sample
                break

            # check distance constraint
            separation_ok = all(
                math.dist(sample[:3], pose[:3]) > min_separation
                for pose in pose_list)

            if separation_ok:
                sample_pose = sample
                break

        # Fallback: if no valid pose found after max tries
        if sample_pose is None:

            pose_list[0][1] = range_list[0][1][0]

            fallback_sample = [
                random.uniform(*place_range.get("x", (0.0, 0.0))),
                place_range.get("y", (0.0, 0.0))[-1],
                random.uniform(*place_range.get("z", (0.0, 0.0))),
                random.uniform(*place_range.get("roll", (0.0, 0.0))),
                random.uniform(*place_range.get("pitch", (0.0, 0.0))),
                random.uniform(*place_range.get("yaw", (0.0, 0.0))),
            ]
            sample_pose = fallback_sample

        pose_list.append(sample_pose)

    return pose_list


def randomize_object_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    pick_range: dict[str, tuple[float, float]] = {},
    place_range: dict[str, tuple[float, float]] = {},
    reset_height: list[float] = [0.0, 0.0],
    max_sample_tries: int = 50,
):
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():
        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            pick_range=pick_range,
            place_range=place_range,
            max_sample_tries=max_sample_tries,
        )

        # Randomize pose for each object
        for i in range(len(asset_cfgs)):
            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env,
                                                                    0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])

            positions[:, 2] = reset_height[i][cur_env]

            # asset.data.reset_root_state[cur_env, :7] = torch.cat(
            #     [positions, orientations], dim=-1)

            asset.write_root_pose_to_sim(torch.cat([positions, orientations],
                                                   dim=-1),
                                         env_ids=torch.tensor(
                                             [cur_env], device=env.device))
            asset.write_root_velocity_to_sim(torch.zeros(1,
                                                         6,
                                                         device=env.device),
                                             env_ids=torch.tensor(
                                                 [cur_env], device=env.device))


class MultiRootStateCfg():

    def __init__(
        self,
        pick_object,
        place_object,
        object_config,
        num_envs,
    ):
        self.pick_object = pick_object
        self.place_object = place_object

        num_pick_object = len(pick_object["objects_list"])
        num_place_object = len(place_object["objects_list"])
        env_ids = torch.arange(num_envs).to("cuda")

        self.define_objects(env_ids=env_ids,
                            num_object=num_pick_object,
                            object_config=object_config,
                            objects_list=pick_object["objects_list"],
                            name="pick")
        self.define_objects(env_ids=env_ids,
                            num_object=num_place_object,
                            object_config=object_config,
                            objects_list=place_object["objects_list"],
                            name="place")
        self.pick_object_name = pick_object["name"]
        self.place_object_name = place_object["name"]

    def define_objects(self,
                       env_ids,
                       num_object,
                       object_config,
                       objects_list,
                       name=None):

        setattr(self, f"default_{name}_root_state",
                torch.zeros((len(env_ids), 7)).to("cuda"))
        setattr(self, f"default_{name}_scale",
                torch.zeros((len(env_ids), 3)).to("cuda"))
        setattr(self, f"reset_{name}_height",
                torch.zeros((len(env_ids), )).to("cuda"))

        for index, object_name in enumerate(objects_list):
            mask = (env_ids % num_object) == index

            object_height = object_config[object_name]["pos"][2]
            getattr(self, f"reset_{name}_height")[mask] = object_height

            rot = torch.as_tensor(
                tuple(
                    obtain_target_quat_from_multi_angles(
                        object_config[object_name]["rot"]["axis"],
                        object_config[object_name]["rot"]["angles"])))
            pos = torch.as_tensor(object_config[object_name]['pos'],
                                  dtype=torch.float32)

            getattr(self, f"default_{name}_scale")[mask] = torch.as_tensor(
                object_config[object_name].get("scale", [1, 1, 1])).to("cuda")
            getattr(self, f"default_{name}_root_state")[mask] = torch.cat(
                (pos, rot), dim=-1).to("cuda").to(dtype=torch.float32)
            getattr(self, f"default_{name}_root_state")[mask,
                                                        2] = object_height

        self.velocity_range = object_config.get("velocity_range", {})
        self.init_scale = False

    def reset_multi_root_state_uniform(
        self,
        env,
        env_ids,
        min_separation: float = 0.20,
    ):

        pick_asset = env.scene[self.pick_object_name]
        # get default root state

        pick_asset.data.default_root_state[
            ..., :7] = self.default_pick_root_state.clone()

        # poses
        randomize_object_pose(
            env,
            env_ids,
            asset_cfgs=[
                SceneEntityCfg(self.pick_object_name),
                SceneEntityCfg(self.place_object_name),
            ],
            min_separation=min_separation,
            pick_range=self.pick_range,
            place_range=self.place_range,
            reset_height=[self.reset_pick_height, self.reset_place_height],
            max_sample_tries=5000,
        )

    def reset_multi_root_target_uniform(self, env, env_ids, handness,
                                        target_object_pose):

        asset = env.scene[self.pick_object_name]
        target_state = env.scene[
            f"{handness}_hand_place_object"]._data.root_state_w.clone()
        target_state[:, 2] += torch.rand(len(target_state)).to(
            env.device) * (target_object_pose[1] -
                           target_object_pose[0]) + target_object_pose[0]

        target_state[:, 3:7] = env.scene[
            f"{handness}_hand_object"]._data.root_state_w[:, 3:7].clone()

        setattr(asset.data, "target_state", target_state[:, :7].clone())


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

    num_envs = env_cfg["params"].get("num_envs", object.scene.num_envs)

    multi_cluster_rigid = env_cfg["params"]["multi_cluster_rigid"]

    pick_object = multi_cluster_rigid.get(f"pick_object", None)
    place_object = multi_cluster_rigid.get(f"place_object", None)

    all_rigid_list = list(env_cfg["params"]["RigidObject"].keys())
    for rigid_object in all_rigid_list:
        if rigid_object not in (pick_object["objects_list"] +
                                place_object["objects_list"]):
            env_cfg["params"]["RigidObject"].pop(rigid_object)

    multi_state = MultiRootStateCfg(
        pick_object,
        place_object,
        env_cfg["params"]["RigidObject"],
        num_envs,
    )

    setattr(multi_state, "pick_range",
            multi_cluster_rigid["pick_object"]["pose_range"])
    setattr(multi_state, "place_range",
            multi_cluster_rigid["place_object"]["pose_range"])
    reset_func = EventTerm(func=multi_state.reset_multi_root_state_uniform,
                           mode="reset",
                           params={
                               "min_separation":
                               env_cfg["params"].get("min_separation", 0.20)
                           })

    pick_object_name = multi_cluster_rigid["pick_object"]["name"]

    setattr(object.events, f"reset_{pick_object_name}", reset_func)

    ### object material

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=800,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(pick_object_name),
            "static_friction_range": (0.5, 2.5),
            "dynamic_friction_range": (0.5, 2.5),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 1250,
        },
    )
    setattr(object.events, f"reset_{pick_object_name}_material",
            object_physics_material)

    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=800,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(pick_object_name),
            "mass_distribution_params": (0.1, 1.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    setattr(object.events, f"reset_{pick_object_name}_material",
            object_scale_mass)
    return env_cfg
