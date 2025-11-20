import numpy as np

import torch


class EvaluateSymmetryEnv:

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

        self.joint_offset = torch.ones((self.env.num_envs, 7)).to(self.device)
        self.joint_offset[..., [0, 2, 4, 6]] = -1.0

    def test(self):
        for i in range(200):
            left_actions = torch.zeros((self.env.num_envs, 23)).to(self.device)
            left_actions[..., 4] = 0.1

            right_actions = left_actions.clone()
            # right_actions[:,:6] = right_actions[:,:6]*self.joint_offset
            right_actions[..., 4] = 0.1

            actions = torch.cat((left_actions, right_actions), dim=1)
            obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)
