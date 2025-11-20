import torch
import numpy as np

import copy

import imageio
from tools.curobo_planner import IKPlanner
from tools.curobo_planner import MotionPlanner

from source.isaaclab_tasks.isaaclab_tasks.manager_based.manipulation.stack.mdp.obs_reward_buffer import RewardObsBuffer
import matplotlib.pyplot as plt

from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import os
from scripts.workflows.open_policy.utils.criterion import criterion_pick_place

from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import tqdm

THRESHOLD = 1e2  # Adjust based on your specific needs


class ReverseRLDatawrapperEnv():

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
        self.reward_buffer = RewardObsBuffer(env_cfg=self.env_config,
                                             target_object_name="green_cube")
        self.robot_type = robot_type
        self.args_cli = args_cli
        if args_cli.save_path is not None or args_cli.load_path is not None:
            import copy

            copy_args = copy.deepcopy(args_cli)
            copy_args.log_dir = args_cli.load_path_dir
            self.collector_interface = MultiDatawrapper(
                copy_args,
                env_config,
                filter_keys=[],
                load_path=copy_args.load_path,
                save_path=copy_args.save_path,
                use_fps=False,
                use_joint_pos=False if "joint" not in args_cli.task else True,
                normalize_action=False,
                load_normalize_action=False)
            reset_buffer(self)

        else:
            self.collector_interface = None

        self.init_planner()
        self.init_demo_buffer()
        self.success_rate = 0.0
        self.total_frames = self.env.max_episode_length - self.env_config[
            "params"]["Task"]["reset_horizon"]
        self.policy = None
        self.eval_freq = self.args_cli.eval_freq
        self.episode_counter = 0

        self.success_threshold = self.args_cli.success_threshold
        self.reset_count = self.args_cli.reset_count
        self.lower_bound_count = 0

        self.target_object_name = self.env_config["params"]["Task"][
            "target_object"]
        self.placement_object_name = self.env_config["params"]["Task"][
            "placement"]["placement_object"]
        self.bbox_region = self.env_config["params"]["Task"][
            "success_condition"]["bbox_region"]

    def init_demo_buffer(self):
        self.buffer_data = self.collector_interface.raw_data["data"]
        self.reset_object_states = {}

        for object_name in self.env.scene.rigid_objects.keys():
            self.reset_object_states[object_name] = []

        self.buffer_action = []
        for demo_key in self.buffer_data.keys():
            obs = self.buffer_data[demo_key]["obs"]

            for object_name in self.env.scene.rigid_objects.keys():
                if f"{object_name}_state" not in obs.keys():
                    if object_name in self.reset_object_states.keys():
                        self.reset_object_states.pop(object_name)

                    continue

                self.reset_object_states[object_name].append(
                    np.array(obs[f"{object_name}_state"])[5])
            self.buffer_action.append(
                np.array(self.buffer_data[demo_key]["actions"]))

        for object_name in self.reset_object_states.keys():

            self.reset_object_states[object_name] = torch.as_tensor(
                np.array(self.reset_object_states[object_name]))

        self.buffer_action = torch.as_tensor(np.array(self.buffer_action))
        # self.buffer_action[:, -20:-1] = -1
        self.num_demos = len(self.buffer_action)
        self.demo_traj = self.buffer_action.shape[1]
        self.env_ids = torch.arange(self.env.num_envs, device=self.device)
        self.demo_begin_frame = self.args_cli.begin_timestep
        self.reset_demo_begin_frame = False
        self.sample_steps = self.args_cli.sample_steps
        self.sample_window = self.args_cli.sample_window
        self.agent = None
        self.save_model = False

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

    def sanity_check(self):
        agent = self.agent
        self.agnet = None

        last_obs = self.reset()

    def reset(self):
        if self.episode_counter % self.eval_freq == 0 and self.agent is not None:

            self.ready_eval = True
            last_obs, info = self.reset_env()
            with torch.no_grad():
                while True:
                    last_obs = self.process_dict_obs(last_obs)
                    rollout_action, _ = self.agent.policy.predict(
                        last_obs.cpu().numpy(), deterministic=True)

                    new_obs, rewards, terminated, time_outs, extras, clip_actions = self.step(
                        torch.as_tensor(rollout_action).to(self.device))
                    _, self.eval_success_rate = criterion_pick_place(
                        self.env,
                        self.target_object_name,
                        self.placement_object_name,
                        self.bbox_region,
                    )
                    if self.eval_success_rate > self.success_threshold[1]:
                        self.save_model = True
                        self.save_model_path = f"eval_model_{self.demo_begin_frame}.pt"
                        self.reset_demo_begin_frame = True
                        break
                    else:
                        self.save_model = False

                    if self.env.episode_length_buf[
                            0] == self.end_frame - 1 and self.eval_success_rate > self.success_threshold[
                                0]:
                        self.lower_bound_count += 1
                        if self.lower_bound_count == self.reset_count:
                            self.save_model = True
                            self.save_model_path = f"train_model_{self.demo_begin_frame}.pt"
                            self.reset_demo_begin_frame = True
                            self.lower_bound_count = 0

                    if self.env.episode_length_buf[
                            0] == self.end_frame - 1 or self.eval_success_rate > self.success_threshold[
                                1]:

                        break
                    last_obs = copy.deepcopy(new_obs)

            print(f"Success rate for evaluation: {self.eval_success_rate}")

            obs, info = self.reset_env()

        else:
            self.ready_eval = False
            obs, info = self.reset_env()

        return obs, info

    def reset_env(self, random_open=True):

        if self.reset_demo_begin_frame:

            self.reset_demo_begin_frame = False
            self.demo_begin_frame -= self.args_cli.sample_steps
            self.demo_begin_frame = np.clip(self.demo_begin_frame, 0, 100000)
            print(
                f"Resetting demo_begin_sample_step to {self.demo_begin_frame}")
        obs, info = self.env.reset()

        self.env.scene["robot"]

        # Sample with replacement
        self.sample_id = torch.randint(0, self.num_demos,
                                       (self.env.num_envs, ))
        for object_name in self.reset_object_states.keys():

            reset_state = self.reset_object_states[object_name][
                self.sample_id].to(self.device).clone()
            reset_state[:, :3] += self.env.scene.env_origins
            self.env.scene[object_name].write_root_link_pose_to_sim(
                reset_state, env_ids=self.env_ids)

        for i in range(self.env_config["params"]["Task"]["reset_horizon"]):
            self.env.scene["robot"].root_physx_view.set_dof_positions(
                self.env.scene["robot"].data.reset_joint_pos,
                indices=torch.arange(self.env.num_envs).to(self.device))

            last_obs, rewards, terminated, time_outs, extras = self.env.step(
                self.reset_actions)

        obs, info = self.forward_traj()
        if obs is not None:
            return obs, info
        else:
            return last_obs, info

    def forward_traj(self):

        sample_end_frames = torch.randint(
            np.clip(self.demo_begin_frame - self.sample_window, 0, 100000),
            np.clip(self.demo_begin_frame, 1, 10000), (self.env.num_envs, ))
        self.forward_actions = self.buffer_action[
            self.sample_id][:, :self.demo_begin_frame]

        mask = torch.ones_like(self.forward_actions)

        # Zero out values after sample_end_frames for each batch
        for i in range(self.forward_actions.shape[0]):
            mask[i, sample_end_frames[i]:] = 0
            try:
                mask[i, sample_end_frames[i]:,
                     -1] = self.forward_actions[i, sample_end_frames[i], -1]
            except:
                pass

        # Apply mask
        self.forward_actions *= mask

        for step_id in range(self.forward_actions.shape[1]):
            obs, rewards, terminated, time_outs, info = self.env.step(
                self.forward_actions[:, step_id])

        self.end_frame = torch.clip(
            int((self.demo_traj - self.demo_begin_frame)) +
            self.env.episode_length_buf[0], 0, self.env.max_episode_length)
        if self.forward_actions.shape[1] > 0:
            return obs, info
        else:
            return None, None

    def step(self, actions, base_action=None):

        if isinstance(actions, np.ndarray):
            actions = torch.as_tensor(actions).to(self.device)

        clip_actions = actions.clone()

        # clip_actions = torch.clamp(clip_actions, -1, 1)

        if self.use_relative_pose:

            clip_actions = (clip_actions + 1) / 2 * (
                self.upper_bound - self.lower_bound) + self.lower_bound
            clip_actions[:, :6] *= self.env.step_dt * 1

        self.clip_actions = clip_actions.clone()
        if base_action is not None:

            obs, rewards, terminated, time_outs, extras = self.env.step(
                clip_actions + base_action)
        else:

            obs, rewards, terminated, time_outs, extras = self.env.step(
                clip_actions)

        # Check for NaN or excessively large values in rewards
        rewards_tensor = rewards
        if torch.isnan(rewards_tensor).any():
            print("Warning: rewards contain NaN values!")
        if (torch.abs(rewards_tensor) > THRESHOLD).any():
            print(f"Warning: rewards contain values larger than {THRESHOLD}!")
        if self.env.episode_length_buf[0] == self.end_frame:
            time_outs[:] = True

            _, self.success_rate = criterion_pick_place(
                self.env, self.target_object_name, self.placement_object_name,
                self.bbox_region, self.args_cli)
            if self.success_rate > self.success_threshold[1]:
                self.save_model = True
                self.save_model_path = f"train_model_{self.demo_begin_frame}.pt"
                self.reset_demo_begin_frame = True
            elif self.success_rate > self.success_threshold[0]:
                self.lower_bound_count += 1
                if self.lower_bound_count == self.reset_count:
                    self.save_model = True
                    self.save_model_path = f"train_model_{self.demo_begin_frame}.pt"
                    self.reset_demo_begin_frame = True
                    self.lower_bound_count = 0
            else:
                self.save_model = False

            print("Stochastic success rate: ", self.success_rate)
            self.episode_counter += 1

        else:
            self.success_rate = 0.0

        return obs, rewards, terminated, time_outs, extras, clip_actions

    def process_dict_obs(self, obs):

        proccess_action = []
        for key, value in obs["policy"].items():
            proccess_action.append(value)

        return torch.cat(proccess_action, dim=1)
