from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
import numpy as np
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer

import copy

import imageio
import sys

sys.path.append("submodule/stable-baselines3/")

from scripts.workflows.open_policy.task.BCPPOBuffer import OnlineBCBuffer


class BCReplayDatawrapper:

    def __init__(self,
                 env,
                 env_config,
                 raw_args_cli,
                 use_relative_pose=False,
                 use_all_data=False):

        self.env = env.env.env
        self.sb3_env = env
        self.use_all_data = use_all_data

        self.device = self.env.device
        self.args_cli = raw_args_cli
        args_cli = copy.deepcopy(raw_args_cli)

        args_cli.log_dir = args_cli.demo_dir
        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_config,
            filter_keys=[],
            load_path=args_cli.load_replay_path,
            save_path=args_cli.save_replay_path,
            use_fps=False,
            use_joint_pos=False if "joint" not in args_cli.task else True,
            normalize_action=False)

        if args_cli.save_replay_path is not None:
            self.collector_interface.init_collector_interface()
        self.use_relative_pose = use_relative_pose
        self.env_config = env_config

        self.num_collected_demo = len(
            self.collector_interface.raw_data["data"])
        from scripts.workflows.hand_manipulation.utils.rl_env.rl_init_utils import init_rl_setting
        init_rl_setting(self)
        self.init_demo_data()
        self.demo_index = 0
        reset_buffer(self)

    def init_demo_data(self):
        raw_data = self.collector_interface.raw_data["data"]
        action_buffer = []
        init_object_pose = []

        for df in raw_data:
            actions = torch.as_tensor(np.asarray(raw_data[df]["actions"])).to(
                self.device)

            actions = actions.reshape(-1, actions.shape[-1])

            raw_actions = torch.zeros(
                (self.horizon, actions.shape[-1])).to(self.device)

            raw_actions[:len(actions)] = actions

            action_buffer.append(raw_actions.unsqueeze(0).clone())

            init_object_pose.append(
                torch.as_tensor(
                    np.asarray(raw_data[df]["obs"]
                               [self.target_manipulated_object][4])).reshape(
                                   -1, 7).to(self.device))

        self.raw_actions_buffer = torch.cat(action_buffer, dim=0)
        self.all_actions_buffer = torch.zeros(
            (self.env.num_envs, self.raw_actions_buffer.shape[1],
             self.raw_actions_buffer.shape[-1])).to(self.device)
        self.all_actions_buffer[:len(
            self.raw_actions_buffer)] = self.raw_actions_buffer[:min(
                len(self.raw_actions_buffer), self.env.num_envs)].clone()

        self.all_actions_buffer[:, len(
            actions
        ):, -self.num_hand_joints:] = self.all_actions_buffer[:,
                                                              len(actions) - 1,
                                                              -self.
                                                              num_hand_joints:].clone(
                                                              ).unsqueeze(
                                                                  1
                                                              ).repeat_interleave(
                                                                  self.horizon
                                                                  -
                                                                  len(actions),
                                                                  1)

        init_object_pose = torch.cat(init_object_pose, dim=0)
        self.init_object_pose = torch.zeros(
            (self.env.num_envs, init_object_pose.shape[-1])).to(self.device)
        self.init_object_pose[:len(init_object_pose)] = init_object_pose[:min(
            len(init_object_pose), self.env.num_envs)].clone()
        self.init_object_pose[:, :3] += self.env.scene.env_origins[:, :3]

    def reset(self):
        obs, info = self.env.reset()

        for i in range(10):
            obs = self.env.step(
                torch.as_tensor(self.env.action_space.sample()).to(self.device)
                * 0.0)[0]

            self.env.scene[
                self.target_manipulated_object].write_root_pose_to_sim(
                    self.init_object_pose,
                    torch.arange(self.env.num_envs).to(self.device))

        self.pre_finger_action = self.env.scene[
            f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                     num_hand_joints:].clone()

        return obs

    def proccess_actions(self, actions):

        # actions[:, -self.num_hand_joints:] -= self.pre_finger_action.clone()

        actions = actions / self.env.step_dt

        clip_actions = ((actions - self.lower_bound) /
                        (self.upper_bound - self.lower_bound)) * 2 - 1

        return clip_actions

    def clip_actions(self, clip_actions):
        raw_actions = (clip_actions + 1) / 2 * (
            self.upper_bound - self.lower_bound) + self.lower_bound
        raw_actions[:, :6] *= self.env.step_dt * 1
        return raw_actions

    def update_buffer(self, obs_buffer, next_obs_buffer, actions_buffer,
                      rew_buffer, does_buffer, success_or_not):

        obs = np.concatenate(obs_buffer, 0).transpose(1, 0, 2)[success_or_not]
        next_obs = np.concatenate(next_obs_buffer,
                                  0).transpose(1, 0, 2)[success_or_not]

        dones = np.concatenate(does_buffer, 0).transpose(
            1, 0)[success_or_not].astype(np.float16)
        actions = np.concatenate(actions_buffer,
                                 0).transpose(1, 0, 2)[success_or_not]

        rews = np.concatenate(rew_buffer, 0).transpose(1, 0)[success_or_not]

        terminates = np.zeros_like(rews)
        terminates[:, -1] = 1

        self.rollout_buffer = OnlineBCBuffer(
            self.env,
            rews.shape[0],
            self.env.observation_space["policy"],
            self.env.action_space,
            self.device,
            rews.shape[0],
        )

        self.rollout_buffer.add(obs, next_obs, actions, rews, terminates,
                                dones)

    def step(self):
        self.rew_buffer = []
        self.obs_buffer = []
        self.actions_buffer = []
        self.next_obs_buffer = []
        self.does_buffer = []

        last_obs = self.reset()

        for i in range(self.raw_actions_buffer.shape[1]):

            robot_actions = self.all_actions_buffer[:, i].clone()
            clip_actions = self.proccess_actions(robot_actions)

            # self.sb3_env.step_async(delta_actions)
            # new_obs, rew, dones, infos = self.sb3_env.step_wait()
            # clip_actions = torch.clamp(clip_actions, -1, 1)

            gt_actions = (clip_actions.clone() + 1) / 2 * (
                self.upper_bound - self.lower_bound) + self.lower_bound
            gt_actions *= self.env.step_dt * 1

            gt_actions[:,
                       -self.num_hand_joints:] += self.pre_finger_action.clone(
                       )
            new_obs, rewards, terminated, time_outs, extras = self.env.step(
                gt_actions)

            self.rew_buffer.append(rewards[None].cpu().numpy())

            self.obs_buffer.append(last_obs["policy"][None].cpu().numpy())
            self.actions_buffer.append(clip_actions[None].cpu().numpy())
            self.next_obs_buffer.append(
                copy.deepcopy(new_obs["policy"][None]).cpu().numpy())
            self.does_buffer.append(time_outs[None].cpu().numpy())

            last_obs = copy.deepcopy(new_obs)
            self.pre_finger_action = self.env.scene[
                f"{self.hand_side}_hand"].data.joint_pos[:, -self.
                                                         num_hand_joints:].clone(
                                                         )

        height = self.env.scene[
            self.target_manipulated_object].data.root_state_w[:, 2]
        success_or_not = height > 0.20
        success_rate = success_or_not.sum().float() / success_or_not.shape[0]

        print("Success rate of the demonstration: ", success_rate)

        self.update_buffer(self.obs_buffer, self.next_obs_buffer,
                           self.actions_buffer, self.rew_buffer,
                           self.does_buffer,
                           success_or_not.cpu().numpy())

    def test_demo(self, test_env):
        print("Testing demo")
        self.reset()
        test_env.reset()
        self.reset()

        for action in self.actions_buffer:

            new_obs, rew, terminate, timeouts, infos, _ = test_env.step(
                torch.as_tensor(action[0]).to(self.device))

        height = self.env.scene[
            self.target_manipulated_object].data.root_state_w[:, 2]
        success_or_not = height > 0.20
        success_rate = success_or_not.sum().float() / success_or_not.shape[0]

        print("Success rate of the demonstration: ", success_rate)
