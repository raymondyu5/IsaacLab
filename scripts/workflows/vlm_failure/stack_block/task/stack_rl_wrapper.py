import torch
import numpy as np

import copy

import imageio
from tools.curobo_planner import IKPlanner
from tools.curobo_planner import MotionPlanner

from scripts.workflows.automatic_articulation.utils.process_action import get_robottip_pose
from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.stack.mdp.obs_reward_buffer import RewardObsBuffer
import matplotlib.pyplot as plt


class RLDatawrapperStack():

    def __init__(
        self,
        env,
        device,
        env_config,
        args_cli,
        use_relative_pose=False,
    ):
        self.env = env
        self.device = device

        self.use_relative_pose = use_relative_pose
        self.env_config = env_config
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.demo_index = 0
        self.action_range = self.env_config["params"]["Task"]["action_range"]
        self.reset_actions = 0 * torch.rand(env.action_space.shape,
                                            device=self.device)
        self.reward_buffer = RewardObsBuffer(env_cfg=self.env_config,
                                             target_object_name="green_cube")
        self.init_planner()

    def init_planner(self):
        self.curo_ik_planner = IKPlanner(self.env)
        target_pose = torch.as_tensor(
            self.env_config["params"]["Task"]["init_ee_pose"]).to(self.device)
        self.target_robot_jpos = self.curo_ik_planner.plan_motion(
            target_pose[:3],
            target_pose[3:7])[0].repeat_interleave(self.env.num_envs, dim=0)
        # self.target_robot_jpos = torch.tensor(
        #     [
        #         0.3147, 0.0430, -0.3113, -2.5660, 0.0258, 2.6067, 0.7668,
        #         0.0400, 0.0400
        #     ],
        #     device=self.device)[None].repeat_interleave(self.env.num_envs, 0)

        self.env.scene["robot"].reset_joint_pos = self.target_robot_jpos
        # print("target_robot_jpos", self.target_robot_jpos)

        action_bound = torch.as_tensor(
            self.env_config["params"]["Task"]["action_range"]).to(self.device)

        self.lower_bound = -action_bound
        self.upper_bound = action_bound
        self.horizon = self.env_config["params"]["Task"]["horizon"]

    def reset(self):
        obs, info = self.env.reset()

        self.env.scene["robot"]

        for i in range(self.env_config["params"]["Task"]["reset_horizon"]):
            self.env.scene["robot"].root_physx_view.set_dof_positions(
                self.target_robot_jpos,
                indices=torch.arange(self.env.num_envs).to(self.device))

            obs, rewards, terminated, time_outs, extras = self.env.step(
                self.reset_actions)
        # self.env.episode_length_buf *= 0

        return obs, info

    def vis_obs(self, obs):
        # Create a figure and subplots
        plt.figure(figsize=(12, 8))
        obs = obs["policy"].cpu().numpy()

        for i in range(obs.shape[1]):
            plt.plot(obs[:, i], label=f'Plot {i+1}')

        # Add labels, legend, and title
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Visualization of N Plots')
        plt.legend()
        plt.show()

    def step(self, actions):

        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()
        clip_actions = torch.clamp(clip_actions, -1, 1)
        if self.use_relative_pose:

            # clip_actions = torch.clamp(clip_actions,
            #                            min=self.lower_bound,
            #                            max=self.upper_bound)

            clip_actions = (clip_actions + 1) / 2 * (
                self.upper_bound - self.lower_bound) + self.lower_bound
            clip_actions[:, :6] *= self.env.step_dt * 2

        clip_actions[:, -1] = torch.sign(clip_actions[:, -1])

        obs, rewards, terminated, time_outs, extras = self.env.step(
            clip_actions)

        # self.vis_obs(obs)
        # self.env.episode_length_buf -= 1
        # rewards = self.reward_buffer.sum_rewards(
        #     self.env,
        #     object_name="green_cube",
        #     ee_frame_name="panda_hand",
        # )
        # self.env.episode_length_buf += 1

        return obs, rewards, terminated, time_outs, extras
