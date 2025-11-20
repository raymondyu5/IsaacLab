from isaaclab_assets import *

import isaaclab.sim as sim_utils

from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

from isaaclab.managers import EventTermCfg as EventTerm
import torch

from isaaclab.managers import ObservationTermCfg as ObsTerm

import isaaclab.utils.math as math_utils
from isaaclab.managers import RewardTermCfg as RewTerm
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.inhand.utils.configuration_robot import config_robot_contact_sensor

import trimesh


class FrankaBucketObs:

    def __init__(
        self,
        env_cfg,
        obs_cfg,
        scene_cfg,
    ):
        self.env_cfg = env_cfg

        self.obs_cfg = obs_cfg
        self.scene_cfg = scene_cfg
        self.add_right_hand = False
        self.add_left_hand = False
        self.fingers_name_list = [
            "fingertip",
            "thumb_fingertip",
            "fingertip_2",
            "fingertip_3",
        ]

        if env_cfg["params"]["add_right_hand"]:
            self.add_right_hand = True
            self.hand_side = "right"
        if env_cfg["params"]["add_left_hand"]:
            self.add_left_hand = True
            self.hand_side = "left"

        self.num_envs = env_cfg["params"].get("num_envs",
                                              self.scene_cfg.num_envs)
        self.target_manipulated_object_pose = torch.as_tensor(
            env_cfg["params"]["target_manipulated_object_pose"]).unsqueeze(
                0).repeat_interleave(self.num_envs, dim=0).to("cuda")
        self.init_height = torch.zeros((self.num_envs)).to("cuda")
        self.env_ids = torch.arange(self.num_envs).to("cuda")
        self.num_object = len(
            list(env_cfg["params"]["ArticulationObject"].keys()))
        for index, object_name in enumerate(
                list(env_cfg["params"]["ArticulationObject"].keys())):
            mask = (self.env_ids % self.num_object) == index
            object_height = env_cfg["params"]["ArticulationObject"][
                object_name]["pos"][2]
            self.init_height[mask] = object_height
        self.handle_name = env_cfg["params"]["articulation_link"].split(
            "/")[-1]
        self.object_name = env_cfg["params"]["spawn_multi_assets_name"]

        self.setup_obs()

    def init_object_mesh(self):

        object_mesh_dir = self.env_cfg["params"]["spawn_rigid_objects"][
            "object_mesh_dir"]
        num_envs = self.env_cfg["params"].get("num_envs",
                                              self.scene_cfg.num_envs)
        num_points = self.env_cfg["params"]["spawn_rigid_objects"][
            "num_points"]
        self.object_points = torch.zeros((num_envs, num_points, 3)).to("cuda")
        for index, object_name in enumerate(
                list(self.env_cfg["params"]["RigidObject"].keys())):
            mesh = trimesh.load(object_mesh_dir + "/" + object_name + ".obj")
            mesh = trimesh.util.concatenate(mesh)
            mask = (self.env_ids % self.num_object) == index

            self.object_points[mask] = math_utils.fps_points(
                torch.as_tensor(mesh.vertices).to("cuda"),
                num_points)[0].to(dtype=torch.float32)

    def config_robot_contact_obs(self):

        contact_sensor_name = self.env_cfg["params"]["contact_sensor"][
            "spawn_contact_list"]

        self.sensor_name = []
        for name in contact_sensor_name:
            self.sensor_name.append(f"{self.hand_side}_{name}")

        contact_obs = ObsTerm(func=self.get_contact_obs, )
        setattr(self.obs_cfg, f"{self.hand_side}_contact_obs", contact_obs)

    def get_contact_obs(self, env):

        return self.contact_or_not

    def setup_obs(self):

        target_object_pose = ObsTerm(func=self.target_object_pose, )
        setattr(self.obs_cfg, "target_object_pose", target_object_pose)

        if self.add_left_hand:
            config_robot_contact_sensor(self.scene_cfg, self.env_cfg, "left")
            self.config_robot_contact_obs()
        if self.add_right_hand:
            config_robot_contact_sensor(self.scene_cfg, self.env_cfg, "right")
            self.config_robot_contact_obs()

        object_in_tip = ObsTerm(func=self.object_in_tip, )
        setattr(self.obs_cfg, "object_in_tip", object_in_tip)
        target_handle_pose = ObsTerm(func=self.target_handle_pose)
        setattr(self.obs_cfg, "target_handle_pose", target_handle_pose)
        # if self.spawn_object_mesh:
        #     object_pcd = ObsTerm(func=self.get_object_points, )
        #     setattr(self.obs_cfg, "object_pcd", object_pcd)

    def get_object_points(self, env):

        init_object_points = self.object_points.clone()
        transformed_points = math_utils.transform_points(
            init_object_points, self.handle_pose[:, :3], self.handle_pose[:,
                                                                          3:7])
        # from tools.visualization_utils import vis_pc, visualize_pcd
        # pc_list = vis_pc(transformed_points[0].cpu().numpy())
        # visualize_pcd([pc_list])
        return transformed_points

    def object_in_tip(self, env):

        return self.finger_object_dev.reshape(env.num_envs, -1)

    def manipulated_object_pose(self, env):

        return self.handle_pose

    def target_object_pose(self, env):
        self.get_object_info(env)
        self.get_finger_info(env)
        self.get_contact_info(env)
        self.target_manipulated_object_pose[:, 3:7] = env.scene[
            self.object_name]._data.default_root_state[:, 3:7].clone()

        target_pose = self.target_manipulated_object_pose.clone().to(
            env.device)

        target_pose[:, 3:7] = env.scene[
            self.object_name]._data.root_state_w[:, 3:7].clone()

        return target_pose

    def target_handle_pose(self, env):
        return self.handle_pose[..., :3]

    def get_finger_info(self, env):
        self.finger_pose = []
        for name in self.fingers_name_list:
            finger = env.scene[f"{self.hand_side}_{name}"]
            finger_pose = finger._data.root_state_w[:, :7].clone()
            finger_pose[:, :3] -= env.scene.env_origins
            self.finger_pose.append(finger_pose.unsqueeze(1))

        self.finger_pose = torch.cat(self.finger_pose, dim=1)

        finger_hand_pose = self.handle_pose.clone().unsqueeze(
            1).repeat_interleave(len(self.fingers_name_list), dim=1)
        self.finger_object_dev = (finger_hand_pose[..., :3] -
                                  self.finger_pose[..., :, :3])

    def get_object_info(self, env):

        self.handle_pose = env.scene[
            self.handle_name]._data.root_state_w[:, :7].clone()

        self.handle_pose[:, :3] -= env.scene.env_origins
        target_pose = self.target_manipulated_object_pose.clone().to(
            env.device)

        self.object_pose = env.scene[
            self.object_name]._data.root_state_w[:, :7].clone()
        self.object_pose[:, :3] -= env.scene.env_origins

    def get_contact_info(self, env):
        sensor_data = []

        for name in self.sensor_name:
            sensor = env.scene[f"{name}_contact"]

            force_data = torch.linalg.norm(sensor._data.force_matrix_w.reshape(
                env.num_envs, 3),
                                           dim=1).unsqueeze(1)

            sensor_data.append(force_data)

        self.contact_or_not = (torch.cat(sensor_data, dim=1) > 4.0).int()
