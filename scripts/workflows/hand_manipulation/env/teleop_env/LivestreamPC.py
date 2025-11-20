import viser
import numpy as np
import time
import torch
from scipy.spatial.transform import Rotation as R
import isaaclab.utils.math as math_utils

import trimesh

import os


class LiveStreamPCViewer:

    def __init__(self,
                 env,
                 env_cfg,
                 host='0.0.0.0',
                 port=8080,
                 load_table=True):
        self.server = viser.ViserServer(host=host, port=port)
        self.env = env
        self.env_cfg = env_cfg
        self.arm_type = self.env_cfg["params"]["arm_type"]
        self.load_table = load_table

        self.meshes = {}
        self.spawn_hand_list = []
        self.spawn_arm_list = []
        self.viser_camera_pose = np.array(
            self.env_cfg["params"]["viser_camera_pose"])
        self.init_setting()

    def init_setting(self):

        if self.env_cfg["params"]["add_right_hand"]:
            self.hand_side = "right"

        if self.env_cfg["params"]["add_left_hand"]:
            self.hand_side = "left"

        if self.env_cfg["params"]["spawn_rigid_objects"]["init"]:

            self.rigid_object_list = self.env_cfg["params"][
                "spawn_rigid_objects"]["spawn_list"].copy()
            self.synthesize_rigid_objects = True
            self.load_mesh(
                self.env, self.env_cfg["params"]["spawn_rigid_objects"]
                ["object_mesh_dir"], self.rigid_object_list)
        else:
            self.synthesize_rigid_objects = False
            self.rigid_object_list = []
        self.init_robot_setting()

        if self.load_table:
            self.process_mesh(
                self.env,
                "source/assets/table/vention_table",
                "table_instanceable",
                "table",
                position=(self.env.scene["table"].get_local_poses()
                          [0].cpu().numpy()[0]),
                wxyz=self.env.scene["table"].get_local_poses()
                [1].cpu().numpy()[0],
                skip=True)

    def init_robot_setting(self, ):

        self.spawn_hand_list += self.env_cfg["params"]["spawn_robot"][
            "spawn_hand_list"].copy()
        self.hand_mesh_path = self.env_cfg["params"]["hand_mesh_dir"]
        self.load_mesh(self.env, self.hand_mesh_path, self.spawn_hand_list)

        if self.env_cfg["params"]["arm_type"] is not None and self.env_cfg[
                "params"]["spawn_robot"]["spawn_arm"]:
            self.spawn_arm_list += self.env_cfg["params"]["spawn_robot"][
                f"spawn_arm_list"].copy()

            self.arm_mesh_path = self.env_cfg["params"]["arm_mesh_dir"]
            self.load_mesh(self.env, self.arm_mesh_path, self.spawn_arm_list)

    def process_mesh(self,
                     env,
                     mesh_dir,
                     mesh_name,
                     link_name,
                     position=(0.0, 0.0, 0.0),
                     wxyz=(1.0, 0.0, 0.0, 0.0),
                     texture_name=None,
                     skip=False):
        if texture_name is not None:

            mesh = trimesh.load(f"{mesh_dir}/{texture_name}.obj")
        else:
            mesh = trimesh.load(f"{mesh_dir}/{mesh_name}.obj")

        mesh = self.server.scene.add_mesh_trimesh(
            name=link_name.replace(".*", self.hand_side),
            mesh=mesh,
            position=position,
            wxyz=wxyz,
            scale=1,
        )

        if not skip:
            self.meshes[link_name.replace(".*", self.hand_side)] = mesh

    def load_mesh(self, env, mesh_dir, spawn_list):

        self.mesh_dict = {}
        texture_name = None

        for index, link_name in enumerate(spawn_list):

            if "palm" in link_name:
                mesh_name = f"{self.hand_side}_" + link_name
            elif "thumb" in link_name:

                mesh_name = "_".join(link_name.split("_"))
            elif link_name in self.rigid_object_list:

                mesh_name = link_name
                mesh_dir = os.path.join(
                    self.env_cfg["params"]["spawn_rigid_objects"]
                    ["object_mesh_dir"], mesh_name)
                texture_name = "textured"

            else:
                if self.arm_type == "ur5e" and link_name in self.spawn_arm_list:
                    mesh_name = "_".join(link_name.split("_")[1:])
                else:
                    parts = link_name.split("_")
                    if parts[-1].isdigit():
                        mesh_name = "_".join(
                            parts[:-1])  # Drop the last numeric part
                    else:
                        mesh_name = "_".join(parts)

            self.process_mesh(env,
                              mesh_dir,
                              mesh_name.replace(".*", self.hand_side),
                              link_name if link_name in self.rigid_object_list
                              else f"{self.hand_side}_" + link_name,
                              texture_name=texture_name)

    def run(self):
        while True:
            actions = torch.rand(self.env.action_space.shape,
                                 device=self.env.unwrapped.device) * 0.0
            obs, reward, terminate, time_out, info = self.env.step(actions)
            for i, link_name in enumerate(self.meshes.keys()):
                # Just for demo: make them orbit in a circle
                pose = self.env.scene[link_name]._data.root_state_w[0, :7]
                self.meshes[link_name].position = pose[:3].cpu().numpy()
                self.meshes[link_name].wxyz = pose[3:7].cpu().numpy()

            viser_camera_pose = np.array(
                self.env_cfg["params"]["viser_camera_pose"])
            # Set camera pose for all connected clients
            for client in list(self.server.get_clients().values()):
                client.camera.look_at = viser_camera_pose[3:]
                client.camera.position = viser_camera_pose[:3]

    def update(self, update_camera=True):
        for i, link_name in enumerate(self.meshes.keys()):
            # Just for demo: make them orbit in a circle
            pose = self.env.scene[link_name]._data.root_state_w[0, :7]
            self.meshes[link_name].position = pose[:3].cpu().numpy()
            self.meshes[link_name].wxyz = pose[3:7].cpu().numpy()
        if update_camera:

            # Set camera pose for all connected clients
            for client in list(self.server.get_clients().values()):
                client.camera.look_at = self.viser_camera_pose[3:]
                client.camera.position = self.viser_camera_pose[:3]
