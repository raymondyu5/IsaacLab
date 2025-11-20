import torch
import numpy as np

import copy

import imageio

from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer, filter_out_data
import tqdm

import matplotlib.pyplot as plt
import os
import math
from scripts.workflows.hand_manipulation.utils.visualizer.plot_eval import viz_object_success_rate, viz_result

from collections import defaultdict
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper


class ResidualEvalRLWrapper:

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        hand_side='right',
    ):
        self.env = env
        self.device = self.env.device
        self.args_cli = args_cli

        self.env_config = env_config
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.eval_success = []
        self.eval_dev = []

        self.rollout_reward = []
        self.eval_iter = 0
        self.collector_interface = None
        self.hand_side = hand_side

        if self.args_cli.save_path is not None:

            self.collector_interface = MultiDatawrapper(
                self.args_cli,
                self.env_config,
                save_path=self.args_cli.save_path,
                load_path=self.args_cli.load_path,
                filter_keys=[
                    "segmentation", "seg_rgb", 'extrinsic_params',
                    'intrinsic_params', 'id2lables'
                ],
            )
        self.init_eval_setting()

    def init_eval_setting(self):

        rigid_object_list = self.env_config['params']["multi_cluster_rigid"][
            f"{self.hand_side}_hand_object"]["objects_list"]
        self.env_ids = torch.arange(self.env.num_envs, ).to(self.device)
        self.rigid_object_success_rate = []
        self.rigid_object_reward = []
        self.rigid_object_dev = []

        self.object_success_rate = {}
        self.object_rewards = {}
        self.object_dev = {}
        if len(rigid_object_list) > 50:

            class_to_objects = defaultdict(list)
            for index, name in enumerate(rigid_object_list):
                object_class = "_".join(
                    name.split('_')[:-1])  # e.g., "apple_2" → "apple"
                class_to_objects[object_class].append(index)
            env_target_ids = self.env_ids % len(rigid_object_list)

            self.rigid_object_list = list(class_to_objects.keys())
            for cls_index, object_class in enumerate(self.rigid_object_list):

                object_index_list = torch.as_tensor(
                    class_to_objects[object_class]).to(self.device)
                mask = torch.isin(env_target_ids, object_index_list)
                env_target_ids[mask] = cls_index

            self.env_target_ids = env_target_ids

        else:
            self.rigid_object_list = rigid_object_list

            self.env_target_ids = self.env_ids % len(rigid_object_list)
            for obj in self.rigid_object_list:
                self.object_success_rate[obj] = []
                self.object_rewards[obj] = []
                self.object_dev[obj] = []
            self.env_rigid_object_name = [[
                self.rigid_object_list[i]
            ] for i in self.env_target_ids.cpu().numpy()]

            self.env_rigid_object_name = []

            for i in self.env_target_ids.cpu().numpy():
                self.env_rigid_object_name.append(self.rigid_object_list[i])

    def init_eval_result_folder(self):
        result_folder = "/".join(
            self.args_cli.checkpoint.split("/")[:4]) + f'/eval_results'
        os.makedirs(result_folder, exist_ok=True)
        num_eval_result = len(os.listdir(result_folder))
        name = self.args_cli.checkpoint.split("/")[3]
        self.save_result_path = f"{result_folder}/{name}_{num_eval_result}"
        os.makedirs(self.save_result_path, exist_ok=True)

    def reset_env(self):
        next_obs, _ = self.reset()
        for i in range(10):
            next_obs, rewards, terminated, time_outs, extras = self.env.step(
                torch.zeros(self.env.action_space.shape, device=self.device))

        return next_obs

    def save_data_to_buffer(self, next_obs, last_obs, hand_arm_actions,
                            rewards, terminated, time_outs):

        # ee_quat_des = self.env.action_manager._terms[
        #     f"{self.hand_side}_arm_action"]._ik_controller.ee_quat_des.clone()
        # ee_pos_des = self.env.action_manager._terms[
        #     f"{self.hand_side}_arm_action"]._ik_controller.ee_pos_des.clone()
        # joint_pos_des = self.env.action_manager._terms[
        #     f"{self.hand_side}_arm_action"].joint_pos_des.clone()
        # finger_pos_des = self.env.action_manager._terms[
        #     f"{self.hand_side}_hand_action"].processed_actions.clone()
        # last_obs["policy"]["ee_control_action"] = torch.cat(
        #     [ee_pos_des, ee_quat_des, finger_pos_des], dim=-1)
        # last_obs["policy"]["joint_control_action"] = torch.cat(
        #     [joint_pos_des, finger_pos_des], dim=-1)
        # last_obs["policy"]["delta_ee_control_action"] = torch.cat([
        #     hand_arm_actions[, :self.num_arm_actions].clone(), finger_pos_des
        # ],
        #                                                           dim=-1)
        # last_obs["policy"]["object_name"] = self.env_rigid_object_name

        update_buffer(
            self,
            next_obs,
            last_obs,
            hand_arm_actions,
            rewards,
            terminated,
            time_outs,
        )

    def eval_symmetry(self, agent, rl_agent_env, bc_eval=False):

        last_obs = self.reset()[0]

        done = False
        reward = torch.zeros(self.env.num_envs, device=self.device)

        while not done:
            if isinstance(last_obs["policy"], dict):

                proccess_last_obs = rl_agent_env._process_obs(last_obs)
            else:
                proccess_last_obs = last_obs["policy"]

            actions, _ = agent.predict(proccess_last_obs, deterministic=True)

            next_obs, spare_rewards, terminated, time_outs, extras, clip_actions = self.step(
                torch.as_tensor(actions).reshape(self.env.num_envs,
                                                 -1).to(self.device),
                bc_eval=bc_eval)

            last_obs = copy.deepcopy(next_obs)
            done = time_outs.any()

            reward += spare_rewards

        success = (last_obs["policy"]
                   [f"{self.hand_side}_manipulated_object_pose"][:, 2]) > 0.20

        np.savez(
            f"{self.save_result_path}/eval_{self.eval_iter}.npz",
            reward=reward.cpu().numpy(),
            success=success.cpu().numpy(),
        )
        self.eval_iter += 1
        return success.cpu().numpy().mean(), reward.cpu().numpy().mean()

    def eval_checkpoint(self, agent, rl_agent_env, bc_eval=False):

        success, reward = self.eval_symmetry(agent,
                                             rl_agent_env,
                                             bc_eval=bc_eval)

        return success, reward

    def eval_all_checkpoint(self, agent, rl_agent_env, model_path):
        reset_buffer(self)

        agent = agent.load(self.args_cli.checkpoint + f"/{model_path}",
                           rl_agent_env,
                           print_system_info=False)

        success, reward = self.eval_symmetry(agent, rl_agent_env)

        return success, reward
