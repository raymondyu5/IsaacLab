import torch
import numpy as np

from scripts.offline_rl.utils.dataloader_utils import populate_data_store_from_zarr, ReplayBuffer


class OfflineEnvWrapper:

    def __init__(self, env, agent):
        self.env = env
        self.agent = agent
        self.obs_keys = [
            'right_hand_joint_pos', "right_manipulated_object_pose",
            "right_target_object_pose"
        ]
        self.joint_limits = torch.as_tensor(
            [[-0.03, 0.03], [-0.03, 0.03], [-0.03, 0.03], [-0.05, 0.05],
             [-0.05, 0.05], [-0.05, 0.05], [-0.314, 2.23], [-0.349, 2.094],
             [-0.314, 2.23], [-0.314, 2.23], [-1.047, 1.047],
             [-0.46999997, 2.4429998], [-1.047, 1.047], [-1.047, 1.047],
             [-0.5059999, 1.8849999], [-1.2, 1.8999999],
             [-0.5059999, 1.8849999], [-0.5059999, 1.8849999],
             [-0.366, 2.0419998], [-1.34, 1.8799999], [-0.366, 2.0419998],
             [-0.366, 2.0419998]], ).to(torch.float32).to(self.env.device)

        obs_keys = [
            'right_hand_joint_pos', "right_manipulated_object_pose",
            "right_target_object_pose"
        ]
        demo_path = "logs/data_0705/retarget_visionpro_data/rl_data/data/ours/image/bunny"

        self.demo_data, use_image, low_dim_obs_shape, capacity = populate_data_store_from_zarr(
            demo_path, obs_keys, num_demos=100)

    def reset(self):
        obs = self.env.reset()
        for i in range(20):
            action = torch.as_tensor(self.env.action_space.sample() * 0.0)
            obs, reward, _, _, info = self.env.step(action)
        return obs, info

    def replay(self):
        last_obs, _ = self.reset()

        actions = self.demo_data["actions"]
        for i in range(len(actions)):

            action = torch.as_tensor(actions[i]).to(
                self.env.device).unsqueeze(0)
            action = (action + 1) / 2 * (
                self.joint_limits[:, 1] -
                self.joint_limits[:, 0]) + self.joint_limits[:, 0]

            self.env.step(action)
            if i % 159 == 0:
                self.reset()

    def step(self, ):
        last_obs, _ = self.reset()
        for i in range(150):

            lowdim_obs = []
            for key in self.obs_keys:
                lowdim_obs.append(last_obs["policy"][key])
            lowdim_obs = torch.cat(lowdim_obs, dim=-1)
            action, _ = self.agent(lowdim_obs)
            action = (action + 1) / 2 * (
                self.joint_limits[:, 1] -
                self.joint_limits[:, 0]) + self.joint_limits[:, 0]

            last_obs, reward, _, _, info = self.env.step(action)
        return last_obs
