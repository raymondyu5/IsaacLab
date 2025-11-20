import os
from collections import defaultdict

import numpy as np

import copy

from tqdm import tqdm
import cv2
import torch
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.real_dataset.real_residual_image_dataset_loader import ResidualRealImageDatasetLoader
from torchvision.transforms import v2 as T


class DataLoader:

    def __init__(self,
                 replay_buffer,
                 visual_keys=[],
                 lowdim_keys=[],
                 data_path=None,
                 target_cam_id=None,
                 load_list=["spider4"],
                 num_demo=100,
                 downsample_points=10240,
                 use_latent_noise_only=False,
                 use_latent_noise=False):
        self.replay_buffer = replay_buffer
        self.visual_keys = visual_keys
        self.lowdim_keys = lowdim_keys
        self.target_cam_id = target_cam_id
        self.load_list = load_list
        self.data_path = data_path
        self.num_demo = num_demo
        self.downsample_points = downsample_points
        self.use_latent_noise = use_latent_noise
        self.use_latent_noise_only = use_latent_noise_only

        self.image_tf = T.Compose([
            T.Resize((128, 128),
                     interpolation=T.InterpolationMode.BICUBIC,
                     antialias=True),
            T.ToDtype(torch.float32, scale=True),  # replaces ToTensor in v2
        ])
        self.load_data()

    def load_data(self, ):

        if "all" in self.load_list:
            self.load_list = os.listdir(self.data_path)
            self.load_list.sort()

        print("Load List", self.load_list)
        demo_count = 0

        for demo_id, path_key in enumerate(self.load_list):

            if not os.path.isdir(self.data_path + "/" + path_key):
                continue
            npy_pathes = os.listdir(self.data_path + "/" + path_key)
            npy_pathes.sort()

            for _, cam_id in enumerate(self.target_cam_id):
                print("loading camer id ", cam_id)

                for file_id in tqdm(range(min([self.num_demo,
                                               len(npy_pathes)])),
                                    desc="Loading demos"):

                    lowdim_obs = []
                    rewards = []
                    cartiesian_postion = []
                    npy_path = os.path.join(
                        self.data_path + "/" + path_key,
                        f"episode_{file_id}/episode_{file_id}.npy")

                    if not os.path.exists(npy_path):
                        continue
                    print(f"episode_{file_id}")

                    # Load the numpy file
                    env_info = np.load(npy_path, allow_pickle=True).item()
                    state_info = env_info['obs']
                    action_info = np.array(env_info['actions'])

                    # low level obs
                    num_horiozon = len(state_info)

                    for index in range(num_horiozon):
                        step_state_info = state_info[index]

                        low_obs = []

                        for obs_name in self.lowdim_keys:
                            if obs_name == "base_action":
                                continue

                            value = np.array(step_state_info[obs_name])
                            low_obs.append(value)

                        lowdim_obs.append(
                            np.concatenate(low_obs, axis=0)[None])
                        rewards.append(step_state_info["reward"])
                        cartesian_position = np.array(
                            step_state_info["cartesian_position"])
                        gripper_position = np.array(
                            step_state_info["gripper_position"])

                        cartiesian_postion.append(
                            np.concatenate(
                                [cartesian_position, gripper_position])[None])

                    lowdim_obs = np.concatenate(lowdim_obs, axis=0)
                    robot_actions = action_info[:, :23]

                    base_action = action_info[:, 23:-22]
                    if "base_action" in self.lowdim_keys:

                        lowdim_obs = np.concatenate([lowdim_obs, base_action],
                                                    axis=-1)
                    rewards = np.array(rewards).reshape(-1, 1)
                    residual_action = action_info[:, -22:]
                    dones = np.zeros_like(rewards)
                    dones[-1] = 1.0

                    if "seg_pc" in self.visual_keys:

                        visual_data = self.load_pcd(path_key, file_id, cam_id)
                        visual_key = "seg_pc"
                    elif "rgb" in self.visual_keys:

                        visual_data = self.load_rgb(path_key, file_id, cam_id)
                        visual_key = "rgb"

                    num_steps = len(action_info)

                    assert len(visual_data) == num_steps

                    visual_data = np.concatenate(visual_data, axis=0)

                    if self.use_latent_noise:

                        latent_noise = np.array(
                            np.load(
                                os.path.join(self.data_path + "/" + path_key,
                                             f"episode_{file_id}") +
                                f"/latent_noise_{file_id}.npy",
                                allow_pickle=True).item()["latent_noise"])
                        assert len(visual_data) == len(lowdim_obs) == len(
                            robot_actions) == len(rewards) == len(latent_noise)
                        if self.use_latent_noise_only:
                            residual_action = latent_noise.reshape(-1, 22)

                        else:

                            residual_action = np.concatenate([
                                residual_action,
                                latent_noise.reshape(-1, 22)
                            ],
                                                             axis=-1)
                    cartiesian_postion = np.concatenate(cartiesian_postion,
                                                        axis=0)
                    assert len(visual_data) == len(lowdim_obs) == len(
                        robot_actions) == len(rewards)

                    for step in range(num_steps - 1):
                        self.replay_buffer.add(
                            {
                                visual_key: visual_data[step],
                                "state": lowdim_obs[step]
                            }, {
                                visual_key: visual_data[step + 1],
                                "state": lowdim_obs[step + 1]
                            },
                            residual_action[step],
                            rewards[step],
                            dones[step + 1],
                            robot_action=robot_actions[step],
                            infos=[{}],
                            base_action=base_action[step],
                            cartesian_action=cartiesian_postion[step])

                    demo_count += 1

    def load_rgb(self, path_key, file_id, cam_id):
        image_list = sorted([
            self.data_path + "/" + path_key + f"/episode_{file_id}/" + f
            for f in os.listdir(self.data_path + "/" + path_key +
                                f"/episode_{file_id}/") if f.endswith(".png")
        ])
        rgb_data = []

        for image_path in image_list:
            img = cv2.imread(image_path,
                             cv2.IMREAD_COLOR)[..., ::-1]  # BGR to RGB

            img_tensor = torch.as_tensor(np.array(img)).permute(
                2, 0, 1)  # (H,W,C) -> (C,H,W)
            out = (
                self.image_tf(img_tensor).permute(1, 2, 0).numpy()[..., ::-1] *
                255).astype(np.uint8)

            rgb_data.append(out[None])
        return rgb_data

    def load_pcd(self, path_key, file_id, cam_id):

        pcd_list = sorted([
            self.data_path + "/" + path_key + f"/episode_{file_id}/" + f
            for f in os.listdir(self.data_path + "/" + path_key +
                                f"/episode_{file_id}/")
            if f.endswith(".npy") and "episode" not in f and cam_id in f
        ])
        pcd_data = []

        for pcd_path in pcd_list:

            proccessed_pcd = np.array(np.load(pcd_path)).astype(
                np.float32).reshape(-1, 3)

            shuffled_indices = np.arange(proccessed_pcd.shape[0])
            np.random.shuffle(shuffled_indices)

            shuffle_pcd_value = (proccessed_pcd[
                shuffled_indices[:self.downsample_points], :][None])
            pcd_data.append(shuffle_pcd_value)

        return pcd_data
