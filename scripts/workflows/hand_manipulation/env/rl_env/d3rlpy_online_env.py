import gymnasium as gym
import numpy as np
import torch


class D3RLpyOnlineEnv:

    def __init__(self, env, args_cli, save_config, target_obs_key, max_action,
                 min_action):
        self.env = env
        self.args_cli = args_cli
        self.env_cfg = save_config
        self.target_obs_key = target_obs_key
        self.device = env.device
        self.max_action = max_action
        self.min_action = min_action

        self.action_space = gym.spaces.Box(low=-1,
                                           high=1,
                                           shape=(
                                               self.env.num_envs,
                                               self.env.action_space.shape[-1],
                                           ),
                                           dtype=np.float32)
        self.action_space = self.env.unwrapped.action_space

        obs_dim = 0

        if len(target_obs_key) == 0:
            for key in self.env.observation_space['policy'].spaces.keys():
                obs_dim += self.env.observation_space['policy'][key].shape[-1]

        else:
            for key in target_obs_key:
                obs_dim += self.env.observation_space['policy'][key].shape[-1]
        self.observation_space = gym.spaces.Box(low=-np.inf,
                                                high=np.inf,
                                                shape=(obs_dim, ),
                                                dtype=np.float32)
        self.use_delta_pose = True if "Rel" in args_cli.task else False
        self.num_envs = env.num_envs
        self.env_ids = torch.arange(self.num_envs).to(self.device)
        self.hand_side = "right" if args_cli.add_right_hand else "left"
        self.num_hand_joint = self.env_cfg["params"]["num_hand_joints"]

    def reset_robot_joints(self, ):

        init_joint_pose = self.env_cfg["params"][
            f"{self.hand_side}_reset_joint_pose"] + [0] * self.num_hand_joint

        self.env.unwrapped.scene[
            f"{self.hand_side}_hand"].root_physx_view.set_dof_positions(
                torch.as_tensor(init_joint_pose).unsqueeze(0).to(
                    self.device).repeat_interleave(self.num_envs, dim=0),
                indices=self.env_ids)

    def process_obs(self, obs):
        per_obs = []

        for obs_key in self.target_obs_key:
            per_obs.append(obs['policy'][obs_key])

        per_obs = torch.cat(per_obs, dim=-1)
        return per_obs.cpu().numpy()

    def reset(self):
        last_obs, _ = self.env.unwrapped.reset()
        for i in range(10):
            self.reset_robot_joints()

            if self.use_delta_pose:

                actions = torch.zeros(self.env.unwrapped.action_space.shape,
                                      dtype=torch.float32,
                                      device=self.device)

            else:

                actions = torch.as_tensor(
                    self.env_cfg["params"].get("init_ee_pose")).to(
                        self.device).unsqueeze(0).repeat_interleave(
                            self.env.unwrapped.num_envs, dim=0)
                actions = torch.concat([
                    actions,
                    torch.zeros((self.env.unwrapped.num_envs, 16),
                                dtype=torch.float32,
                                device=self.device)
                ],
                                       dim=-1)
            last_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
                actions)
        return self.process_obs(last_obs), {}

    def step(self, action):

        action = action.clip(-1, 1)

        torch_action = torch.as_tensor(action).to(self.device)
        env_action = self.max_action * (
            (torch_action + 1) / 2) + self.min_action

        next_obs, rewards, terminated, time_outs, extras = self.env.step(
            env_action)
        next_obs = self.process_obs(next_obs)

        if self.env.unwrapped.episode_length_buf[
                0] == self.env.unwrapped.max_episode_length - 1:
            # next_obs, _ = self.reset()
            terminated = torch.ones_like(terminated).bool()

        return next_obs, rewards.cpu().numpy(), terminated.cpu().numpy(
        ), time_outs.cpu().numpy(), extras
