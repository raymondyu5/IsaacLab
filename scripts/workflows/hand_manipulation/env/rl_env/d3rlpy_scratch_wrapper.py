from scripts.workflows.hand_manipulation.env.rl_env.rl_step_wrapper import RLStepWrapper

import torch
import gymnasium as gym

import numpy as np
from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import Sb3VecEnvWrapper

from scripts.workflows.hand_manipulation.env.rl_env.rl_wrapper import RLDatawrapperEnv


class D3RLpyScratchWrapper:

    def __init__(self, env, args_cli, env_config, target_obs_key=[]):

        rl_env = RLDatawrapperEnv(
            env,
            env_config,
            args_cli=args_cli,
            use_relative_pose=True if "Rel" in args_cli.task else False,
            use_joint_pose=True if "Joint-Rel" in args_cli.task else False,
        )

        # wrap around environment for stable baselines
        self.wrapper = Sb3VecEnvWrapper(rl_env,
                                        gpu_buffer=False,
                                        args_cli=args_cli)

        self.env = env
        self.hand_side = "right" if args_cli.add_right_hand else "left"
        self.num_hand_joint = env_config["params"]["num_hand_joints"]
        self.device = self.env.unwrapped.device
        self.num_envs = self.env.unwrapped.num_envs
        self.env_ids = torch.arange(self.num_envs).to(self.device)
        self.env_cfg = env_config
        self.target_obs_key = target_obs_key

        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(
                self.env.unwrapped.num_envs,
                self.wrapper.action_space.shape[-1],
            ),
            dtype=np.float32)

        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.wrapper.observation_space.shape[-1], ),
            dtype=np.float32)
        self.rewards  = 0.0

    def process_obs(self, obs):
        per_obs = []

        for obs_key in self.target_obs_key:
            per_obs.append(obs['policy'][obs_key])

        per_obs = torch.cat(per_obs, dim=-1)
        return per_obs.cpu().numpy()

    def reset(self):
        obs = self.wrapper.reset()
        print("Episode reward after reset:", self.rewards)

        self.rewards  = 0.0

        return obs, {}

    def step(self, action):
        self.wrapper.step_async(action)

        next_obs, rewards, dones, infos, raw_action, diffusion_obs = self.wrapper.step_wait(
        )
        self.rewards += rewards.mean().item()

        return next_obs, rewards, dones, dones, infos
