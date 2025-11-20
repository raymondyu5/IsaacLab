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


def sample_object_poses(
    num_objects: int,
    min_separation: float = 0.0,
    max_sample_tries: int = 5000,
    range_list: list = None,
):

    pose_list = []
    pick_range_list = [
        range_list[0].get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]

    for i in range(num_objects):
        sample_pose = None

        for _ in range(max_sample_tries):
            if i == 0:
                range_sample_list = pick_range_list
            else:
                range_sample_list = range_list[1].tolist()

            sample = [random.uniform(r[0], r[1]) for r in range_sample_list]

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
        if i == 1:
            sample_pose = None
        if sample_pose is None:

            y_min, y_max = range_list[0].get("y", (-0.2, 0.2))

            first_index = random.choice([-1, 1])

            pose_list[0][1] = first_index * y_max
            articulation_pose = np.array(range_list[1].tolist())

            fallback_sample = [
                random.uniform(*articulation_pose[0]),
                first_index * range_list[1].tolist()[1][0],
                random.uniform(*articulation_pose[2]),
                random.uniform(*articulation_pose[3]),
                random.uniform(*articulation_pose[4]),
                random.uniform(*articulation_pose[5]),
            ]
            sample_pose = fallback_sample

        pose_list.append(sample_pose)

    return pose_list


def randomize_object_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfgs: list[SceneEntityCfg],
    min_separation: float = 0.0,
    reset_height: list[float] = [0.0, 0.0],
    max_sample_tries: int = 50,
    range_list: list = None,
):
    if env_ids is None:
        return

    # Randomize poses in each environment independently
    for cur_env in env_ids.tolist():

        pose_list = sample_object_poses(
            num_objects=len(asset_cfgs),
            min_separation=min_separation,
            max_sample_tries=max_sample_tries,
            range_list=[range_list[0], range_list[1][cur_env]])

        # Randomize pose for each object
        for i in range(len(asset_cfgs) - 1, -1, -1):

            asset_cfg = asset_cfgs[i]
            asset = env.scene[asset_cfg.name]

            # Write pose to simulation
            pose_tensor = torch.tensor([pose_list[i]], device=env.device)
            positions = pose_tensor[:, 0:3] + env.scene.env_origins[cur_env,
                                                                    0:3]
            orientations = math_utils.quat_from_euler_xyz(
                pose_tensor[:, 3], pose_tensor[:, 4], pose_tensor[:, 5])

            positions[:, 2] = reset_height[i][cur_env]
            default_root_state = asset.data.default_root_state.clone()

            orientations = math_utils.quat_mul(
                orientations, default_root_state[cur_env, 3:7].reshape(-1, 4))
            # asset.data.reset_root_state[cur_env, :7] = torch.cat(
            #     [positions, orientations], dim=-1)

            asset.write_root_pose_to_sim(torch.cat([positions, orientations],
                                                   dim=-1).to(torch.float32),
                                         env_ids=torch.tensor(
                                             [cur_env], device=env.device))
            asset.write_root_velocity_to_sim(torch.zeros(1,
                                                         6,
                                                         device=env.device),
                                             env_ids=torch.tensor(
                                                 [cur_env], device=env.device))


def define_articulation_objects(
    self,
    env_ids,
    num_object,
    articulation_object_setting,
):

    index = 0
    name = "articulation"

    setattr(self, f"default_{name}_root_state",
            torch.zeros((len(env_ids), 7)).to("cuda"))
    setattr(self, f"default_{name}_scale",
            torch.zeros((len(env_ids), 3)).to("cuda"))
    setattr(self, f"reset_{name}_height",
            torch.zeros((len(env_ids), )).to("cuda"))
    self.articulation_pose_range = []

    for index, object_name in enumerate(self.articulation_objects):
        mask = (env_ids % num_object) == index

        object_height = articulation_object_setting[object_name]["pos"][2]
        getattr(self, f"reset_{name}_height")[mask] = object_height

        rot = torch.as_tensor(
            tuple(
                obtain_target_quat_from_multi_angles(
                    articulation_object_setting[object_name]["rot"]["axis"],
                    articulation_object_setting[object_name]["rot"]
                    ["angles"])))
        pos = torch.as_tensor(articulation_object_setting[object_name]['pos'],
                              dtype=torch.float32)

        getattr(self, f"default_{name}_scale")[mask] = torch.as_tensor(
            articulation_object_setting[object_name].get("scale",
                                                         [1, 1, 1])).to("cuda")
        getattr(self, f"default_{name}_root_state")[mask] = torch.cat(
            (pos, rot), dim=-1).to("cuda").to(dtype=torch.float32)
        getattr(self, f"default_{name}_root_state")[mask, 2] = object_height

        pose_range = articulation_object_setting[object_name].get(
            "pose_range", {})
        articulation_pose_range = torch.tensor([
            pose_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]).to("cuda")

        self.articulation_pose_range.append(articulation_pose_range)


def define_rigid_objects(self,
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
            object_config[object_name].get("scale", [1, 1, 1]),
            dtype=torch.float32).to("cuda")
        getattr(self, f"default_{name}_root_state")[mask] = torch.cat(
            (pos, rot), dim=-1).to("cuda").to(dtype=torch.float32)
        getattr(self, f"default_{name}_root_state")[mask, 2] = object_height


class MultiRootStateCfg():

    def __init__(
        self,
        pick_object,
        articulation_object_setting,
        object_config,
        articulation_config,
        num_envs,
    ):
        self.pick_object = pick_object
        self.articulation_objects = list(articulation_object_setting.keys())

        num_pick_object = len(pick_object["objects_list"])

        num_articulation_object = len(self.articulation_objects)
        env_ids = torch.arange(num_envs).to("cuda")

        define_rigid_objects(self,
                             env_ids=env_ids,
                             num_object=num_pick_object,
                             object_config=object_config,
                             objects_list=pick_object["objects_list"],
                             name="pick")

        define_articulation_objects(
            self,
            env_ids=env_ids,
            num_object=num_articulation_object,
            articulation_object_setting=articulation_object_setting,
        )
        self.pick_object_name = pick_object["name"]

        self.articulation_object_name = articulation_config["name"]

    def reset_multi_root_state_uniform(
        self,
        env,
        env_ids,
        min_separation: float = 0.30,
    ):

        pick_asset = env.scene[self.pick_object_name]
        # get default root state

        pick_asset.data.default_root_state[
            ..., :7] = self.default_pick_root_state.clone()
        articulation = env.scene[self.articulation_object_name]
        articulation.data.default_root_state[
            ..., :7] = self.default_articulation_root_state.clone()

        # poses
        randomize_object_pose(
            env,
            env_ids,
            asset_cfgs=[
                SceneEntityCfg(self.pick_object_name),
                SceneEntityCfg(self.articulation_object_name),
            ],
            min_separation=min_separation,
            reset_height=[
                self.reset_pick_height, self.reset_articulation_height
            ],
            max_sample_tries=5000,
            range_list=[self.pose_pick_range, self.articulation_pose_range])

    def reset_multi_root_target_uniform(self, env, env_ids, handness,
                                        target_object_pose):

        asset = env.scene[self.pick_object_name]
        target_state = env.scene[
            f"{handness}_hand_place_object"]._data.root_state_w.clone()
        target_state[:, 2] += torch.rand(len(target_state)).to(
            env.device) * (target_object_pose[1] -
                           target_object_pose[0]) + target_object_pose[0]
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

    for handness in ["right", "left"]:
        if not env_cfg["params"][f"add_{handness}_hand"]:
            continue

        pick_object = multi_cluster_rigid.get(f"{handness}_hand_object", None)

        multi_state = MultiRootStateCfg(
            pick_object,
            env_cfg["params"].get("ArticulationObject"),
            env_cfg["params"]["RigidObject"],
            env_cfg["params"]["spawn_multi_articulation"],
            num_envs,
        )

        setattr(multi_state, "pose_pick_range",
                multi_cluster_rigid["right_hand_object"]["pose_range"])
        reset_func = EventTerm(
            func=multi_state.reset_multi_root_state_uniform,
            mode="reset",
            params={"min_separation": env_cfg.get("min_separation", 0.3)},
        )

        reset_articulation_joint = EventTerm(
            func=mdp.reset_joints_by_values,
            mode="reset",
            params={
                "joint_pose": torch.zeros(1),
                "asset_name": multi_state.articulation_object_name
            })

        object_name = f"{handness}_hand_object"
        setattr(object.events, f"reset_{object_name}", reset_func)
        setattr(object.events, f"reset_articulation_joint",
                reset_articulation_joint)

        joint_friction = EventTerm(
            func=mdp.randomize_joint_parameters,
            min_step_count_between_reset=200,
            mode="reset",
            params={
                "asset_cfg":
                SceneEntityCfg(multi_state.articulation_object_name),
                "friction_distribution_params": (10, 100),
                "operation": "add",
                "distribution": "uniform",
            },
        )
        setattr(object.events, f"reset_{handness}_hand_articulation_friction",
                joint_friction)

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
