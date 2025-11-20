import torch
import numpy as np

import copy

from tools.curobo_planner import IKPlanner

import matplotlib.pyplot as plt
from isaaclab_tasks.utils.data_collector import RobomimicDataCollector
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import os

from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import tqdm

from scripts.workflows.open_policy.utils.criterion import criterion_pick_place
import isaaclab.envs.mdp as mdp


class RLDatawrapperEnv():

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        use_relative_pose=False,
        robot_type="franka",
    ):
        self.env = env.env
        self.device = self.env.device

        self.use_relative_pose = use_relative_pose
        self.env_config = env_config
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.demo_index = 0
        self.action_range = self.env_config["params"]["Task"]["action_range"]
        self.reset_actions = 0 * torch.rand(env.action_space.shape,
                                            device=self.device)

        self.robot_type = robot_type
        self.args_cli = args_cli
        if args_cli.save_path is not None:
            self.collector_interface = MultiDatawrapper(
                args_cli,
                env_config,
                filter_keys=[],
                load_path=args_cli.load_path,
                save_path=args_cli.save_path,
                use_fps=False,
                use_joint_pos=False if "joint" not in args_cli.task else True,
                normalize_action=False,
                load_normalize_action=False)
            reset_buffer(self)

        else:
            self.collector_interface = None

        self.init_planner()
        self.agent = None
        self.episode_counter = 0
        self.eval_freq = self.args_cli.eval_freq

        self.target_object_name = self.env_config["params"]["Task"][
            "target_object"]
        self.placement_object_name = self.env_config["params"]["Task"][
            "placement"]["placement_object"]
        self.bbox_region = self.env_config["params"]["Task"][
            "success_condition"]["bbox_region"]
        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

    def init_planner(self):
        if self.robot_type == "franka":
            self.curo_ik_planner = IKPlanner(self.env)
            target_pose = torch.as_tensor(
                self.env_config["params"]["Task"]["init_ee_pose"]).to(
                    self.device)
            self.target_robot_jpos = self.curo_ik_planner.plan_motion(
                target_pose[:3],
                target_pose[3:7])[0].repeat_interleave(self.env.num_envs,
                                                       dim=0)
            self.env.scene[
                "robot"].data.reset_joint_pos = self.target_robot_jpos

        action_bound = torch.as_tensor(
            self.env_config["params"]["Task"]["action_range"]).to(self.device)

        self.lower_bound = -action_bound
        self.upper_bound = action_bound

        self.horizon = self.env_config["params"]["Task"]["horizon"]

    def reset(self):
        if self.episode_counter % self.eval_freq == 0 and self.agent is not None:
            self.ready_eval = True
            last_obs, info = self.reset_env()
            with torch.no_grad():
                while True:
                    if isinstance(last_obs["policy"], dict):
                        last_obs = self.process_dict_obs(last_obs)
                    else:
                        last_obs = last_obs["policy"]
                    rollout_action, _ = self.agent.policy.predict(
                        last_obs.cpu().numpy(), deterministic=True)
                    new_obs, rewards, terminated, time_outs, extras, _ = self.step(
                        torch.as_tensor(rollout_action).to(self.device))
                    _, self.eval_success_rate = criterion_pick_place(
                        self.env, self.target_object_name,
                        self.placement_object_name, self.bbox_region)

                    if self.env.episode_length_buf[
                            0] == self.env.max_episode_length - 1:

                        break
                    last_obs = copy.deepcopy(new_obs)

            print(f"Success rate for evaluation: {self.eval_success_rate}")
            obs, info = self.reset_env()

        else:
            self.ready_eval = False
            obs, info = self.reset_env()
        return obs, info

    def reset_env(self, random_open=True):
        obs, info = self.env.reset()

        for reset_object_name in self.env_config["params"]["Task"][
                "reset_object_names"]:
            if reset_object_name not in self.env_config["params"]["RL_Train"][
                    "rigid_object_names"]:
                continue

            mdp.reset_rigid_articulation(
                self.env,
                self.env_ids,
                target_name=reset_object_name,
                pose_range=self.env_config["params"]["RigidObject"]
                [reset_object_name]["pose_range"])

        for i in range(self.env_config["params"]["Task"]["reset_horizon"]):
            self.env.scene["robot"].root_physx_view.set_dof_positions(
                self.env.scene["robot"].data.reset_joint_pos,
                indices=torch.arange(self.env.num_envs).to(self.device))

            obs, rewards, terminated, time_outs, extras = self.env.step(
                self.reset_actions)
        # self.env.episode_length_buf *= 0

        return obs, info

    def vis_obs(self, obs):
        # Create a figure and subplots
        plt.figure(figsize=(12, 8))
        obs = obs.cpu().numpy()

        for i in range(obs.shape[1]):
            plt.plot(obs[:, i], label=f'Plot {i+1}')

        # Add labels, legend, and title
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title('Visualization of N Plots')
        plt.legend()
        plt.show()

    def step(self, actions, base_action=None):

        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()

        clip_actions = torch.clamp(clip_actions, -1, 1)
        if self.use_relative_pose:

            clip_actions = (clip_actions + 1) / 2 * (
                self.upper_bound - self.lower_bound) + self.lower_bound
            clip_actions[:, :6] *= self.env.step_dt * 1

        clip_actions[:, -1] = torch.sign(clip_actions[:, -1])
        self.clip_actions = clip_actions.clone()

        if base_action is not None:

            obs, rewards, terminated, time_outs, extras = self.env.step(
                clip_actions + base_action)
        else:

            obs, rewards, terminated, time_outs, extras = self.env.step(
                clip_actions)

        THRESHOLD = 1e2  # Adjust based on your specific needs

        # Check for NaN or excessively large values in observations
        obs_tensor = obs["policy"]

        # if torch.isnan(obs_tensor).any():
        #     print("Warning: obs contains NaN values!")
        # if (torch.abs(obs_tensor) > THRESHOLD).any():

        #     print(f"Warning: obs contains values larger than {THRESHOLD}!")

        # Check for NaN or excessively large values in rewards
        rewards_tensor = rewards
        if torch.isnan(rewards_tensor).any():
            print("Warning: rewards contain NaN values!")
        # if (torch.abs(rewards_tensor) > THRESHOLD).any():
        #     print(f"Warning: rewards contain values larger than {THRESHOLD}!")

        if self.env.episode_length_buf[0] == self.env.max_episode_length - 1:
            self.episode_counter += 1
        return obs, rewards, terminated, time_outs, extras, clip_actions

    def process_dict_obs(self, obs):

        proccess_action = []
        for key, value in obs["policy"].items():
            proccess_action.append(value)

        return torch.cat(proccess_action, dim=1)

    def eval_checkpoint(self, agent, last_obs):

        self.rew_buffer = []
        self.obs_buffer = []
        self.actions_buffer = []

        last_obs, _ = self.reset()

        total_frame = 80  #self.horizon
        i = 0
        reset_index = self.horizon
        reset_alread = False

        while i < total_frame:

            if isinstance(last_obs["policy"], dict):
                proccess_last_obs = self.process_dict_obs(last_obs)
            else:
                proccess_last_obs = last_obs["policy"]

            actions, _ = agent.predict(proccess_last_obs.cpu().numpy(),
                                       deterministic=True)

            # if i > 150:
            #     actions[:, -1] = 1

            next_obs, rewards, terminated, time_outs, extras, clip_actions = self.step(
                torch.as_tensor(actions).to(self.device))
            success_or_not, success_rate = criterion_pick_place(
                self.env, self.target_object_name, self.placement_object_name,
                self.bbox_region)

            last_obs["policy"]["success_or_not"] = success_or_not.unsqueeze(1)
            # if success_rate > 0.70 and not reset_alread:
            #     print("stop early at frame: ", i)
            #     reset_alread = True
            #     total_frame = np.min([i + 10, total_frame])

            if self.collector_interface is not None:

                update_buffer(self, next_obs, last_obs, clip_actions, rewards,
                              terminated, time_outs)

            last_obs = copy.deepcopy(next_obs)

            i += 1

        if self.args_cli.save_path is not None:

            self.collector_interface.add_demonstraions_to_buffer(
                self.obs_buffer, self.action_buffer, self.rewards_buffer,
                self.does_buffer)

        reset_buffer(self)

        return success_or_not

    def eval_residual_checkpoint(self, openvla_rl_env, agent, last_obs):

        self.rew_buffer = []
        self.obs_buffer = []
        self.actions_buffer = []

        last_obs = openvla_rl_env.reset()
        raw_last_obs = openvla_rl_env.raw_obs_dict

        for i in range(self.horizon):

            actions, _ = agent.predict(last_obs, deterministic=True)
            openvla_rl_env.step_async(actions)

            new_obs, rewards, dones, infos = openvla_rl_env.step_wait()

            if self.collector_interface is not None:

                for key, value in raw_last_obs["policy"].items():
                    self.collector_interface.add(f"obs/{key}", value)

                # for key, value in new_obs["policy"].items():
                #     self.collector_interface.add(f"next_obs/{key}", value)

                # -- actions
                self.collector_interface.add("actions", self.clip_actions)

                self.collector_interface.add("rewards", rewards)

                self.collector_interface.add("dones", dones)
                if i == self.horizon - 1:
                    dones = torch.as_tensor([True] * self.env.num_envs,
                                            dtype=torch.bool)
                else:
                    dones = torch.as_tensor([False] * self.env.num_envs,
                                            dtype=torch.bool)

                reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)

                self.collector_interface.flush(reset_env_ids)
            last_obs = copy.deepcopy(new_obs)
            raw_last_obs = openvla_rl_env.raw_obs_dict

        return rewards > 10
