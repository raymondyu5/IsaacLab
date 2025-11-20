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
# from tools.draw.mesh_visualizer.batch_collision_checker import SDFCompuation

from isaaclab.envs.mdp.events import apply_external_force_torque


class EvalRLWrapper:

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        use_relative_pose=False,
        use_joint_pose=False,
        hand_side='right',
    ):
        self.env = env
        self.device = self.env.device
        self.args_cli = args_cli
        self.compute_sdf = args_cli.compute_sdf
        if self.compute_sdf:
            self.sdf_computer = SDFCompuation()

        self.use_relative_pose = use_relative_pose
        self.env_config = env_config
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.task = "place" if "Place" in args_cli.task else "grasp"

        self.eval_success = []
        self.eval_dev = []

        self.rollout_reward = []
        self.eval_iter = 0
        self.collector_interface = None
        self.hand_side = hand_side
        self.num_env_actions = 22

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
            self.args_cli.checkpoint.split("/")[:-2]) + f'/eval_results'
        os.makedirs(result_folder, exist_ok=True)
        # num_eval_result = len(os.listdir(result_folder))

        self.save_result_path = f"{result_folder}"
        os.makedirs(self.save_result_path, exist_ok=True)

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

    def eval_disturbance(self, agent, last_obs, checkpoint_path, *kwargs):

        save_dir = "/".join(
            self.args_cli.checkpoint.split("/")[:-2]) + f'/eval_disturbance'
        os.makedirs(save_dir, exist_ok=True)
        eval_result = []

        for i in range(0, 50, 1):
            self.env.event_manager._mode_term_cfgs["interval"][0].params[
                "force_range"] = [-0.05 * i, 0.05 * i]
            self.env.event_manager._mode_term_cfgs["interval"][0].params[
                "torque_range"] = [-0.05 * i, 0.05 * i]
            success_or_not, stop = self.eval_checkpoint(agent, last_obs)
            eval_result.append(success_or_not.sum().item() / self.env.num_envs)
            np.save(save_dir + f"/disturbance_eval.npz", np.array(eval_result))
            print("Success rate: ",
                  success_or_not.sum().item() / self.env.num_envs, i)

    def eval_checkpoint(self, agent, video_writer=None, *kwargs):
        reset_buffer(self)

        next_obs, _ = self.reset()

        raw_new_obs = copy.deepcopy(next_obs)
        body_state_w = self.env.scene[
            f"{self.hand_side}_hand"]._data.body_state_w
        body_names = self.env.scene[f"{self.hand_side}_hand"].body_names
        raw_new_obs["policy"][
            f"{self.hand_side}_hand_link_pose"] = body_state_w[:, :, :7]
        raw_new_obs["policy"][f"{self.hand_side}_body_names"] = [body_names]
        total_reward = torch.zeros(self.env.num_envs).to(self.device)

        for i in range(160):

            last_obs = copy.deepcopy(next_obs)
            raw_last_obs = copy.deepcopy(raw_new_obs)

            if isinstance(last_obs["policy"], dict):
                proccess_last_obs = self.process_dict_obs(last_obs)
            else:
                proccess_last_obs = last_obs["policy"]

            actions = torch.as_tensor(
                agent.predict(proccess_last_obs.cpu().numpy(),
                              deterministic=True)[0]).to(self.device)

            next_obs, rewards, terminated, time_outs, extras, hand_arm_actions, _ = self.step(
                actions)
            if video_writer is not None:

                frame = self.env.render()
                video_writer.append_data(frame)

            total_reward += rewards

            body_state_w = self.env.scene[
                f"{self.hand_side}_hand"]._data.body_state_w
            body_names = self.env.scene[f"{self.hand_side}_hand"].body_names
            raw_new_obs = copy.deepcopy(next_obs)
            raw_new_obs["policy"][
                f"{self.hand_side}_hand_link_pose"] = body_state_w[:, :, :7]
            raw_new_obs["policy"][f"{self.hand_side}_body_names"] = [
                body_names
            ]

            raw_new_obs["policy"][f"{self.hand_side}_hand_link_pose"][
                ..., :3] -= self.env.scene.env_origins[:, None, :3]

            if self.args_cli.save_path is not None:
                self.save_data_to_buffer(raw_new_obs, raw_last_obs,
                                         hand_arm_actions, rewards, terminated,
                                         time_outs)
            dones = terminated | time_outs
            if dones[0]:
                break

        self.eval_iter += 1
        print("reward: ", total_reward.sum().item() / self.env.num_envs)

        if self.task == "grasp":
            success = (last_obs["policy"]
                       [f"{self.hand_side}_manipulated_object_pose"][:,
                                                                     2]) > 0.3
        elif self.task == "place":

            pick_object_state = (
                last_obs["policy"][f"{self.hand_side}_manipulated_object_pose"]
                [:, :3])
            place_target_state = (
                last_obs["policy"][f"{self.hand_side}_hand_place_object_pose"]
                [:, :3])
            success = torch.linalg.norm(pick_object_state[:, :2] -
                                        place_target_state[:, :2],
                                        dim=1) < 0.10
        print("success rate: ", success.sum().item() / self.env.num_envs)

        return success, total_reward

    def lift_or_not(self, ):

        target_object_state = self.env.scene[
            f"{self.hand_side}_hand_object"].data.root_state_w[..., :7]
        success_flag = target_object_state[:, 2] > 0.3

        if success_flag.sum() > 0:
            if self.args_cli.save_path is not None:

                index = torch.nonzero(success_flag, as_tuple=True)[0]

                filter_out_data(self, index)

                demo_id = self.collector_interface.traj_count - 1

                np.savetxt(
                    self.args_cli.log_dir + "/" + self.args_cli.save_path +
                    f"/object_name_{demo_id}.txt", [
                        obj_name for index_ in index.cpu().numpy()
                        for obj_name in [self.env_rigid_object_name[index_]]
                    ],
                    fmt='%s')
        return success_flag

    def eval_all_checkpoint(self, agent, last_obs, rl_agent_env):
        reset_buffer(self)

        all_ckpt_list = os.listdir(self.args_cli.checkpoint)
        all_ckpt_list = [ckpt for ckpt in all_ckpt_list if "zip" in ckpt]
        all_ckpt_list = sorted(
            all_ckpt_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        for ckpt in all_ckpt_list:
            video_writer = None

            if self.args_cli.render_video:
                video_writer = imageio.get_writer(
                    os.path.join(self.save_result_path,
                                 ckpt.replace('.zip', '_eval.mp4')),
                    fps=30,
                )

            agent = agent.load(
                self.args_cli.checkpoint + f"/{ckpt}",
                rl_agent_env,
                print_system_info=False,
            )

            success, total_reward = self.eval_checkpoint(
                agent, video_writer=video_writer)
            # visualize_latent_space(next_obs["policy"])
            self.eval_iter += 1

            self.eval_success.append(success.sum().item() / self.env.num_envs)

            self.rigid_object_success_rate.append(success.to(torch.int))
            self.rigid_object_reward.append(total_reward)

            self.rollout_reward.append(
                (total_reward.sum().item() / self.env.num_envs))
            viz_result(self)

            if video_writer is not None:
                video_writer.close()
            # viz_object_success_rate(self)

        return success
