import numpy as np

import torch
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper

from scripts.workflows.hand_manipulation.utils.dataset_utils.pca_utils import reconstruct_hand_pose_from_normalized_action
import time


class ReplayDexTrajDataset:

    def __init__(
        self,
        args_cli,
        env_config,
        env,
    ):
        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.unwrapped.device

        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]

        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_config,
            load_path=args_cli.load_path,
            save_path=args_cli.save_path,
        )
        if self.args_cli.save_path is not None:
            self.collector_interface.init_collector_interface()
        self.num_count = 0
        if self.add_left_hand:
            self.hand_side = "left"
        elif self.add_right_hand:
            self.hand_side = "right"
        self.raw_data = self.collector_interface.raw_data["data"]

        self.num_data = len(self.raw_data)
        self.demo_count = 0
        self.load_pca_data()

    def load_pca_data(self):

        eigengrasps_info = np.load(
            f"logs/grab_hand/{self.hand_side}_pca.npz.npy",
            allow_pickle=True).item()
        self.eigen_vectors = torch.as_tensor(
            eigengrasps_info["eigen_vectors"]).to(self.device)
        self.min_values = torch.as_tensor(eigengrasps_info["min_values"]).to(
            self.device)
        self.max_values = torch.as_tensor(eigengrasps_info["max_values"]).to(
            self.device)
        self.D_mean = torch.as_tensor(eigengrasps_info["D_mean"]).to(
            self.device)
        self.D_std = torch.as_tensor(eigengrasps_info["D_std"]).to(self.device)
        # Expand min/max to the full original space

        min_orig_norm = torch.matmul(self.min_values, self.eigen_vectors)
        max_orig_norm = torch.matmul(self.max_values, self.eigen_vectors)
        # Now pick the true min and max
        final_min = torch.minimum(min_orig_norm, max_orig_norm)
        final_max = torch.maximum(min_orig_norm, max_orig_norm)
        # Step 2: Denormalize
        self.min_orig = final_min * self.D_std + self.D_mean
        self.max_orig = final_max * self.D_std + self.D_mean

    def verify(self, ):

        self.env.reset()
        print("new demo")
        demo_actions = torch.as_tensor(
            np.array(self.raw_data[f"demo_{self.demo_count}"]["actions"])).to(
                self.device)[::10]
        normazlied_demo_actions = (demo_actions[:, -self.num_hand_joints:] -
                                   self.D_mean) / self.D_std

        eigengrasp_demo_actions = torch.matmul(normazlied_demo_actions,
                                               self.eigen_vectors.T)
        normalized_action = (eigengrasp_demo_actions - self.min_values) / (
            self.max_values - self.min_values) * 2 - 1
        reconstructed_actions = reconstruct_hand_pose_from_normalized_action(
            normalized_action, self.eigen_vectors, self.min_values,
            self.max_values, self.D_mean, self.D_std)

        # x_norm = torch.matmul(eigengrasp_demo_actions, self.eigen_vectors)

        # reconstructed_actions = x_norm * self.D_std + self.D_mean  # (B, D)

        for index, hand_action in enumerate(reconstructed_actions):
            action = torch.zeros(self.env.action_space.shape).to(self.device)
            action[..., :6] = 0.0
            action[..., 2] = 0.10
            action[0, 6:] = hand_action  #hand_action
            action[1, 6:] = demo_actions[index, -self.num_hand_joints:]

            self.env.step(action)
            time.sleep(0.1)
        self.demo_count += 1

    def random_play(self):
        for i in range(200):

            raw_actions = torch.rand(
                (self.env.num_envs, self.eigen_vectors.shape[0])).to(
                    self.device) * 2 - 1
            reconstructed_actions = reconstruct_hand_pose_from_normalized_action(
                raw_actions, self.eigen_vectors, self.min_values,
                self.max_values, self.D_mean, self.D_std)
            actions = torch.zeros(
                (self.env.num_envs, self.env.action_space.shape[-1]),
                device=self.device)

            actions[..., 6:] = reconstructed_actions
            self.env.step(actions)
            time.sleep(0.1)
