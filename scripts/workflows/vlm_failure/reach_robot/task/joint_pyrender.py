import numpy as np
import h5py
import torch
import os
import cv2
import imageio
import argparse
import pickle
from tools.trash.vlm_failure.render_data import JointMarker
import isaaclab.utils.math as math_utils


class RenderJoint:

    def __init__(self,
                 args_cli,
                 save_path,
                 load_path,
                 save_video=False,
                 panda_link_names=None,
                 num_camera=3):
        self.args_cli = args_cli
        self.save_path = save_path
        self.load_path = load_path
        self.save_video = save_video
        self.panda_link_names = panda_link_names
        self.num_panda_link = len(panda_link_names)
        self.num_camera = num_camera
        self.action_marker = JointMarker(
            image_width=self.args_cli.resolution[0],
            image_height=self.args_cli.resolution[1],
            camera_scales=[self.args_cli.camera_scale] * self.num_panda_link,
            sphere_radius=1.0,
            znear=0.00001,
            zfar=100.0,
        )
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        if self.save_video:
            os.makedirs(f"{self.save_path}/videos", exist_ok=True)

    def extract(self, obs):
        rgb_images = []
        links_state_name = {}

        for link_name in self.panda_link_names:
            if "panda" not in link_name:
                continue
            panda_state = obs[f"{link_name}_state"].cpu().numpy()[0]
            panda_quat = panda_state[3:7]

            panda_matricies = np.identity(4)

            panda_matricies[:3, :3] = math_utils.matrix_from_quat(
                torch.as_tensor(panda_quat)).cpu().numpy()
          
            panda_matricies[:3, 3] = panda_state[:3]
            links_state_name[link_name] = panda_matricies

        for index_cam in range(self.num_camera):

            joint_matrices = []
            sphere_colors = []

            color_list = ["red", "green", "purple"]

            for color_id, link_name in enumerate(self.panda_link_names):
                if "panda" not in link_name:
                    continue
                joint_matrices.append(links_state_name[link_name])
                sphere_colors.append(color_list[color_id % (len(color_list))])
            intrinsic = obs["intrinsic_params"][0].cpu().numpy()
            extrinsic = obs["extrinsic_params"][0].cpu().numpy()

            rendered_img = self.action_marker.render_action(
                cam_intrinsic=intrinsic[index_cam],
                cam_extrinsic=extrinsic[index_cam],
                joint_matrices=joint_matrices,
                camera_scale=self.args_cli.camera_scale,
                joint_opens=[0] * self.num_panda_link,
                sphere_colors=sphere_colors)
            rgb_images.append(rendered_img)

        return np.array(rgb_images)

    def convert(self):
        for file in os.listdir(f"{self.load_path}"):
            if file.endswith(".npz"):
                data = torch.load(os.path.join(f"{self.load_path}", file),
                                  pickle_module=pickle)
                demo_key = file.split(".")[0]

                obs_buffer = data["obs"]
                num_steps = len(obs_buffer)

                if self.save_video:

                    video_writer = imageio.get_writer(
                        f"{self.save_path}/videos/{demo_key}.mp4", fps=30)

                for index_step in range(num_steps):
                    rgb_images = []
                    links_state_name = {}
                    obs = obs_buffer[index_step]["policy"]

                    for link_name in self.panda_link_names:
                        panda_state = obs[f"{link_name}_state"]
                        panda_quat = panda_state[:, 3:7]

                        panda_matricies = np.identity(4)

                        panda_matricies[:3, :3] = math_utils.matrix_from_quat(
                            panda_quat).cpu().numpy()[0]
                        panda_matricies[:3,
                                        3] = panda_state[0, :3].cpu().numpy()
                        links_state_name[link_name] = panda_matricies

                    for index_cam in range(self.num_camera):

                        joint_matrices = []
                        sphere_colors = []
                        color_list = ["red", "green", "purple"]

                        for color_id, link_name in enumerate(
                                self.panda_link_names):
                            joint_matrices.append(links_state_name[link_name])
                            sphere_colors.append(color_list[color_id %
                                                            (len(color_list))])

                        intrinsic = obs["intrinsic_params"][0].cpu().numpy()
                        extrinsic = obs["extrinsic_params"][0].cpu().numpy()

                        rendered_img = self.action_marker.render_action(
                            cam_intrinsic=intrinsic[index_cam],
                            cam_extrinsic=extrinsic[index_cam],
                            joint_matrices=joint_matrices,
                            camera_scale=self.args_cli.camera_scale,
                            joint_opens=[0] * self.num_panda_link,
                            sphere_colors=sphere_colors)

                        rgb_image = obs["rgb"][0][index_cam].cpu().numpy()

                        mask = np.all(rendered_img == [255, 255, 255], axis=-1)

                        # Replace white pixels in rendered_img with corresponding pixels from rgb_image
                        combined_img = np.where(mask[..., None], rgb_image,
                                                rendered_img)
                        rgb_images.append(combined_img)

                    if self.save_video:
                        stitched_image = np.concatenate(rgb_images, axis=1)
                        video_writer.append_data(stitched_image[:, :, ::-1])


# renderer = RenderJoint(
#     args_cli,
#     args_cli.save_path,
#     args_cli.load_path,
#     args_cli.save_video,
#     args_cli.panda_link_names,
# )
# renderer.convert()
