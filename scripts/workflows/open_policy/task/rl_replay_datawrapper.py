from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
import numpy as np
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer

import copy

import imageio
import sys

sys.path.append("submodule/stable-baselines3/")
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
import matplotlib.pyplot as plt

from scripts.workflows.open_policy.task.BCPPOBuffer import OnlineBCBuffer
from scripts.workflows.open_policy.utils.criterion import criterion_pick_place


class RLReplayDatawrapper:

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
        args_cli = copy.deepcopy(raw_args_cli)

        args_cli.log_dir = args_cli.load_openvla_dir
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

        if self.use_relative_pose:
            self.reset_actions = 0 * torch.rand(env.action_space.shape,
                                                device=self.device)

        self.num_collected_demo = len(
            self.collector_interface.raw_data["data"])
        self.init_setting()
        self.demo_index = 0
        reset_buffer(self)

    def init_setting(self):
        self.rigbid_objects_list = []
        # for name in self.env.scene.rigid_objects.keys():
        #     if "wx250s" in name:
        #         continue
        #     self.rigbid_objects_list.append(name)
        self.rigbid_objects_list = self.env_config["params"]["Task"][
            "reset_object_names"]
        self.env_ids = torch.arange(self.env.num_envs).to(self.device)
        self.horizon = self.env_config["params"]["Task"]["horizon"]

        raw_data = self.collector_interface.raw_data["data"]
        action_buffer = []
        self.rigid_objects_buffer = {}
        for rigbid_object_name in self.rigbid_objects_list:
            self.rigid_objects_buffer[rigbid_object_name] = []
        length = 0
        self.target_object_name = self.env_config["params"]["Task"][
            "target_object"]
        self.placement_object_name = self.env_config["params"]["Task"][
            "placement"]["placement_object"]
        self.bbox_region = self.env_config["params"]["Task"][
            "success_condition"]["bbox_region"]
        for df in raw_data:
            actions = torch.as_tensor(np.asarray(raw_data[df]["actions"])).to(
                self.device)

            actions = actions.reshape(-1, actions.shape[-1])

            length = np.max([length, len(actions)])

            raw_actions = torch.zeros(
                (self.horizon, actions.shape[-1])).to(self.device)

            raw_actions[:len(actions)] = actions

            # raw_actions[len(actions):, :-1] = actions[-1, :-1]

            action_buffer.append(raw_actions.unsqueeze(0).clone())

            for rigbid_object_name in self.rigbid_objects_list:

                self.rigid_objects_buffer[rigbid_object_name].append(
                    torch.as_tensor(
                        np.asarray(raw_data[df]["obs"]
                                   [f"{rigbid_object_name}_state"][0])).to(
                                       self.device).unsqueeze(0))

        for rigbid_object_name in self.rigbid_objects_list:
            raw_state = torch.zeros((self.env.num_envs, 7)).to(self.device)
            obs_data = torch.cat(self.rigid_objects_buffer[rigbid_object_name],
                                 dim=0)
            raw_state[:np.min([len(obs_data), self.env.num_envs]
                              )] = obs_data[:self.env.num_envs]

            raw_state[:, :3] += self.env.scene.env_origins
            self.rigid_objects_buffer[rigbid_object_name] = raw_state

        self.actions_buffer = torch.cat(action_buffer, dim=0)

        self.all_actions = torch.zeros(
            (self.env.num_envs, self.horizon,
             self.actions_buffer.shape[-1])).to(self.device)

        self.all_actions[:np.min([len(obs_data), self.
                                  env.num_envs])] = self.actions_buffer.clone(
                                  )[:self.env.num_envs]

        self.all_actions = self.all_actions

        action_bound = torch.as_tensor(
            self.env_config["params"]["Task"]["action_range"]).to(self.device)

        self.lower_bound = -action_bound
        self.upper_bound = action_bound
        self.horizon = self.env_config["params"]["Task"]["horizon"]

        self.target_object_name = self.env_config["params"]["Task"][
            "target_object"]
        self.placement_object_name = self.env_config["params"]["Task"][
            "placement"]["placement_object"]
        self.bbox_region = self.env_config["params"]["Task"][
            "success_condition"]["bbox_region"]

    def reset(self):
        self.env.reset()
        self.cur_obs = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["obs"]

        init_robot_joint_pos = torch.as_tensor(
            self.cur_obs["joint_pos"][0]).unsqueeze(0).to(
                self.device).repeat_interleave(self.env.num_envs, dim=0)
        for name in self.rigbid_objects_list:
            asset = self.env.scene.rigid_objects[name]

            raw_state = self.rigid_objects_buffer[name][:self.env.num_envs]

            asset.data.reset_root_state[:, :7] = raw_state[:, :7]

            # set into the physics simulation
            asset.write_root_link_pose_to_sim(raw_state[:, :7],
                                              env_ids=self.env_ids)

        for i in range(self.env_config["params"]["Task"]["reset_horizon"]):
            need_dof = self.env.scene[
                "robot"].root_physx_view.get_dof_positions()
            need_dof[:, :8] = init_robot_joint_pos[:, :8]
            self.env.scene["robot"].root_physx_view.set_dof_positions(
                need_dof, indices=self.env_ids)

            self.sb3_env.step_async(
                self.reset_actions.unsqueeze(0).repeat_interleave(
                    self.env.num_envs, dim=0))

            obs, rew, dones, infos = self.sb3_env.step_wait()

        return obs

    def proccess_actions(self, actions):

        actions[:, :6] /= self.env.step_dt

        clip_actions = ((actions - self.lower_bound) /
                        (self.upper_bound - self.lower_bound)) * 2 - 1

        return clip_actions

    def clip_actions(self, clip_actions):
        raw_actions = (clip_actions + 1) / 2 * (
            self.upper_bound - self.lower_bound) + self.lower_bound
        raw_actions[:, :6] *= self.env.step_dt * 1
        return raw_actions

    def update_buffer(self, obs_buffer, next_obs_buffer, actions_buffer,
                      rew_buffer, does_buffer, satistied_index):

        obs = np.concatenate(obs_buffer, 0).transpose(1, 0, 2)[satistied_index]
        next_obs = np.concatenate(next_obs_buffer,
                                  0).transpose(1, 0, 2)[satistied_index]

        dones = np.concatenate(does_buffer, 0).transpose(
            1, 0)[satistied_index].astype(np.float16)
        actions = np.concatenate(actions_buffer,
                                 0).transpose(1, 0, 2)[satistied_index]

        rews = np.concatenate(rew_buffer, 0).transpose(1, 0)[satistied_index]

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

        for i in range(self.all_actions.shape[1]):

            delta_actions = self.all_actions[:, i].clone()

            delta_actions[:, -1] = torch.sign(delta_actions[:, -1])

            self.sb3_env.step_async(delta_actions)
            new_obs, rew, dones, infos = self.sb3_env.step_wait()

            self.rew_buffer.append(rew[None])
            self.obs_buffer.append(last_obs[None])
            self.actions_buffer.append(
                self.proccess_actions(delta_actions)[None].cpu().numpy())
            self.next_obs_buffer.append(copy.deepcopy(new_obs[None]))
            self.does_buffer.append(dones[None])

            last_obs = copy.deepcopy(new_obs)

        if not self.use_all_data:
            success_or_not, success_rate = criterion_pick_place(
                self.env, self.target_object_name, self.placement_object_name,
                self.bbox_region)
            satistied_index = success_or_not.cpu().numpy()
            print("Success rate of the demonstration: ", success_rate)
        else:

            num_demo = len(self.collector_interface.raw_data["data"]) - 10
            satistied_index = np.zeros(self.env.num_envs)
            satistied_index[:num_demo] = 1
            satistied_index = satistied_index.astype(bool)
        self.update_buffer(self.obs_buffer, self.next_obs_buffer,
                           self.actions_buffer, self.rew_buffer,
                           self.does_buffer, satistied_index)

    def test_demo(self, test_env):
        print("Testing demo")
        self.reset()
        for action in self.actions_buffer:

            new_obs, rew, terminate, timeouts, infos, _ = test_env.step(
                torch.as_tensor(action[0]).to(self.device))

        success_or_not, success_rate = criterion_pick_place(
            self.env, self.target_object_name, self.placement_object_name,
            self.bbox_region)
        print("Success rate of the demonstration: ", success_rate)
