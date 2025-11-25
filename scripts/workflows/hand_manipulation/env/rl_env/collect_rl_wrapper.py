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

import time

from tools.visualization_utils import *


class CollectRLWrapper:

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        wrapper,
        use_relative_pose=False,
        use_joint_pose=False,
        hand_side='right',
    ):
        self.env = env
        self.device = self.env.device
        self.args_cli = args_cli
        self.wrapper = wrapper

        self.use_relative_pose = use_relative_pose
        self.env_config = env_config
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

        self.eval_success = []
        self.eval_dev = []

        self.rollout_reward = []
        self.eval_iter = 0
        self.collector_interface = None
        self.hand_side = hand_side
        self.total_success = 0
        self.task = "place" if "Place" in args_cli.task else "grasp"
        self.horizon = self.env_config["params"]["Task"]["horizon"]
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]

        self.horizon = self.env_config["params"]["Task"]["horizon"]

        if self.args_cli.save_path is not None:
            filter_keys = [
                "segmentation", "seg_rgb", 'extrinsic_params',
                'intrinsic_params', 'id2lables'
            ]
            if self.args_cli.use_failure:
                filter_keys += ["seg_pc"]

            self.collector_interface = MultiDatawrapper(
                self.args_cli,
                self.env_config,
                save_path=self.args_cli.save_path,
                # load_path=self.args_cli.load_path,
                filter_keys=[
                    "segmentation", "seg_rgb", 'extrinsic_params',
                    'intrinsic_params', 'id2lables'
                ],
                save_zarr=True)
            setattr(self.wrapper, "save_data", True)
            delattr(self.wrapper, "collector_interface")
            setattr(self.wrapper, "collector_interface",
                    self.collector_interface)

    def process_dict_obs(self, obs):

        proccess_action = []
        for key, value in obs["policy"].items():

            if key in [
                    'seg_rgb', 'segmentation', 'rgb', 'whole_pc', 'seg_pc',
                    'extrinsic_params', 'intrinsic_params', 'id2lables'
            ]:
                continue

            proccess_action.append(value)

        return torch.cat(proccess_action, dim=1)

    def reset_robot_joints(self, ):

        # Get arm joint pose from config
        arm_joint_pose = self.env_config["params"][
            f"{self.hand_side}_reset_joint_pose"]

        # Get hand joint pose from config, default to zeros if not specified
        hand_joint_pose = self.env_config["params"].get(
            f"{self.hand_side}_reset_hand_joint_pose", [0] * self.num_hand_joints)

        init_joint_pose = arm_joint_pose + hand_joint_pose

        self.env.scene[
            f"{self.hand_side}_hand"].root_physx_view.set_dof_positions(
                torch.as_tensor(init_joint_pose).unsqueeze(0).to(
                    self.device).repeat_interleave(self.env.num_envs, dim=0),
                indices=torch.arange(self.env.num_envs).to(self.device))

    # def reset_env(self):

    #     next_obs, _ = self.env.reset()
    #     for i in range(10):

    #         if "Rel" in self.args_cli.task:
    #             next_obs, rewards, terminated, time_outs, extras = self.env.step(
    #                 torch.zeros((self.env.num_envs, 22), device=self.device))
    #         else:

    #             link7_pose = self.env.scene[
    #                 f"{self.hand_side}_panda_link7"]._data.root_state_w[:, :
    #                                                                     7].clone(
    #                                                                     )
    #             link7_pose[:, :3] -= self.env.scene.env_origins

    #             final_ee_pose = torch.cat([
    #                 link7_pose,
    #                 torch.zeros((self.env.num_envs, 16)).to(self.device)
    #             ],
    #                                       dim=-1)
    #             next_obs, rewards, terminated, time_outs, extras = self.env.step(
    #                 final_ee_pose)

    # if self.args_cli.save_path is not None:
    #     reset_buffer(self.wrapper)
    #     setattr(self.wrapper, "last_obs", next_obs)

    #     return next_obs

    def collect_data(self, agent):

        # wrapper.reset() now handles forced joint reset + 10 warmup steps internally
        next_obs, _ = self.wrapper.reset()

        if self.args_cli.save_path is not None:
            reset_buffer(self.wrapper)
            setattr(self.wrapper, "last_obs", next_obs)
        start_time = time.time()

        # Number of initial steps to send zero actions (warmup period)
        warmup_steps = getattr(self.args_cli, 'warmup_steps', 0)

        for i in range(self.horizon):

            last_obs = copy.deepcopy(next_obs)

            if isinstance(last_obs["policy"], dict):
                proccess_last_obs = self.process_dict_obs(last_obs)
            else:
                proccess_last_obs = last_obs["policy"]

            # During warmup: bypass wrapper and send zero actions directly to env
            if i < warmup_steps:
                # Use raw action space dimension (arm + hand joints)
                action_dim = self.wrapper.raw_action_space if hasattr(self.wrapper, 'raw_action_space') else 22
                zero_action = torch.zeros(self.env.num_envs, action_dim,
                                          dtype=torch.float32, device=self.device)
                next_obs, rewards, terminated, time_outs, extras = self.env.step(zero_action)

                # Update wrapper state for data collection
                setattr(self.wrapper, "last_obs", next_obs)
                if hasattr(self.wrapper, 'add_obs_to_buffer'):
                    self.wrapper.add_obs_to_buffer(self.env.episode_length_buf[0])

                hand_arm_actions = zero_action
            else:
                # Normal operation: use agent + wrapper
                actions = torch.as_tensor(
                    agent.predict(proccess_last_obs.cpu().numpy(),
                                  deterministic=True)[0]).to(self.device)
                setattr(self.wrapper, "eval_iter", i)

                next_obs, rewards, terminated, time_outs, extras, hand_arm_actions = self.wrapper.step(
                    actions)[:6]

            # cam_o3d = vis_pc(next_obs["policy"]["seg_pc"][0][0].cpu().numpy())
            # visualize_pcd([cam_o3d])

            dones = terminated | time_outs
            if dones[0]:
                break

        # if self.args_cli.save_path is not None:
        success_flag = self.lift_or_not(last_obs)
        self.eval_iter += 1
        self.total_success += success_flag.sum().item()
        print("Colleting data time: ",
              time.time() - start_time, "success: ",
              self.total_success / (self.env.num_envs * self.eval_iter))

        return success_flag

    def lift_or_not(self, last_obs):
        success_flag = self.wrapper.eval_success(last_obs)

        # Save all data if flag is set (for testing with action clipping)
        if hasattr(self.args_cli, 'save_all_data') and self.args_cli.save_all_data:
            if self.args_cli.save_path is not None:
                # Save all environments regardless of success/failure
                index = torch.arange(self.env.num_envs)
                filter_out_data(self.wrapper, index.cpu())
            return success_flag

        if success_flag.sum() > 0 or self.args_cli.use_failure:
            if self.args_cli.save_path is not None:

                if self.args_cli.use_failure and float(
                        self.args_cli.failure_ratio) > 0.1:

                    success_index = torch.nonzero(success_flag,
                                                  as_tuple=True)[0]
                    failure_index = torch.nonzero(~success_flag,
                                                  as_tuple=True)[0]
                    total_data = min([
                        len(failure_index) /
                        float(self.args_cli.failure_ratio), self.env.num_envs
                    ])
                    # Step 4: concatenate final indices
                    num_fail = int(total_data *
                                   float(self.args_cli.failure_ratio))
                    num_succ = int(total_data - num_fail)

                    if num_fail <= len(failure_index):
                        chosen_fail = failure_index[torch.randperm(
                            len(failure_index))[:num_fail]]
                    else:
                        # not enough failures, so just take all of them
                        chosen_fail = failure_index

                    # Successes
                    if num_succ <= len(success_index):
                        chosen_succ = success_index[torch.randperm(
                            len(success_index))[:num_succ]]
                    else:
                        chosen_succ = success_index

                    # final index
                    final_index = torch.cat([chosen_fail, chosen_succ]).cpu()

                    filter_out_data(self.wrapper, final_index)
                elif self.args_cli.use_failure:
                    index = torch.arange(self.env.num_envs)

                    filter_out_data(self.wrapper, index.cpu())

                else:

                    index = torch.nonzero(success_flag, as_tuple=True)[0]

                    filter_out_data(self.wrapper, index.cpu())

        return success_flag
