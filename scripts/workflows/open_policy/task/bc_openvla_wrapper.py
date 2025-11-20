from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
import numpy as np
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer

import copy

import imageio
from tqdm import tqdm
from scripts.workflows.utils.client.openvla_client import resize_image


class BCDatawrapperOpenPolicy:

    def __init__(
        self,
        env,
        env_config,
        args_cli,
        use_relative_pose=False,
    ):
        self.env = env
        self.device = env.device

        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_config,
            filter_keys=[],
            load_path=args_cli.load_path,
            save_path=args_cli.save_path,
            use_fps=False,
            use_joint_pos=False if "joint" not in args_cli.task else True,
            normalize_action=args_cli.normalize_action,
            load_normalize_action=args_cli.normalize_action)

        self.use_relative_pose = use_relative_pose
        self.env_config = env_config
        self.args_cli = args_cli

        if self.use_relative_pose:
            self.reset_actions = 0 * torch.rand(env.action_space.shape,
                                                device=self.device)

        self.num_collected_demo = len(
            self.collector_interface.raw_data["data"])
        self.init_setting()
        self.demo_index = 0
        reset_buffer(self)

    def init_setting(self):
        self.rigbid_objects_list = self.env_config["params"]["Task"][
            "reset_object_names"]
        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

    def reset(self):
        self.env.reset()
        self.cur_obs = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["obs"]

        if not self.args_cli.eval:
            # while True:
            #     last_rewards = self.collector_interface.raw_data["data"][
            #         f"demo_{self.demo_index}"]["rewards"][-1]
            #     if last_rewards > 10:
            #         break
            #     self.demo_index += 1

            for name in self.rigbid_objects_list:
                asset = self.env.scene.rigid_objects[name]

                init_pose = torch.as_tensor(
                    self.cur_obs[f"{name}_state"][0]).unsqueeze(0).to(
                        self.device)
                asset.data.reset_root_state[:, :7] = init_pose[:, :7]

                # set into the physics simulation
                asset.write_root_link_pose_to_sim(init_pose[:, :7],
                                                  env_ids=self.env_ids)
                # asset.write_root_com_velocity_to_sim(init_pose[:, 7:],
                #                                      env_ids=self.env_ids)

        init_robot_joint_pos = torch.as_tensor(
            self.cur_obs["joint_pos"][0]).unsqueeze(0).to(
                self.device).repeat_interleave(self.env.num_envs, dim=0)

        for i in range(self.env_config["params"]["Task"]["reset_horizon"]):

            need_dof_pos = self.env.scene[
                "robot"].root_physx_view.get_dof_positions()
            need_dof_pos[:, :init_robot_joint_pos.
                         shape[-1]] = init_robot_joint_pos
            self.env.scene["robot"].root_physx_view.set_dof_positions(
                need_dof_pos, indices=self.env_ids)

            if self.use_relative_pose:
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    self.reset_actions)
            else:
                init_ee_pose = torch.as_tensor(
                    self.cur_obs["ee_pose"][0]).unsqueeze(0).to(self.device)
                init_ee_pose = torch.cat(
                    [init_ee_pose, torch.zeros((1, 1))], dim=-1)
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    init_ee_pose)
        return obs

    def replay_policy(self, policy):
        self.reset()

        actions = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["actions"]
        actions = np.array(actions)

        total_frame = actions.shape[0]

        for i in range(total_frame):
            if self.args_cli.mode == "replay":
                action = actions[i]
            else:
                noramlized_action = actions[i]
                action = self.collector_interface.unnormalize(
                    noramlized_action,
                    self.collector_interface.action_stats["action"])
            action = torch.as_tensor(action).to(self.device)

            next_obs, rewards, terminated, time_outs, extras = self.env.step(
                action.unsqueeze(0))

        self.demo_index += 1

        return rewards > 10

    def open_loop_policy(self, policy):
        last_obs = self.reset()

        actions = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["actions"]
        actions = np.array(actions)
        obs_buffer = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["obs"]

        total_frame = actions.shape[0]
        if self.args_cli.model_type == "bc":
            pass

        else:
            obs_dict = {}

            for key in obs_buffer.keys():
                per_obs = obs_buffer[key][0][None]
                batch_obs = np.concatenate([per_obs, per_obs], axis=0)

                obs_dict[key] = torch.as_tensor(batch_obs).to(self.device)

            if "gs_image" in obs_buffer.keys():
                obs_dict["gs_image"] = obs_dict["gs_image"].permute(
                    0, 3, 1, 2) / 255

        for i in range(total_frame):

            if self.args_cli.model_type == "bc":
                obs_dict = {}
                for key in obs_buffer.keys():
                    if key == "gs_image":
                        obs_dict[key] = torch.as_tensor(
                            obs_buffer[key][i]).permute(2, 0, 1).to(
                                self.device) / 255
                    else:
                        obs_dict[key] = torch.as_tensor(obs_buffer[key][i]).to(
                            self.device)

            else:
                for key in obs_buffer.keys():
                    if key == "gs_image":

                        obs_dict[key][-1] = torch.as_tensor(
                            obs_buffer[key][i]).permute(2, 0, 1).to(
                                self.device) / 255
                    else:
                        obs_dict[key][-1] = torch.as_tensor(
                            obs_buffer[key][i]).to(self.device)
            if self.args_cli.normalize_action:
                noramlized_action = policy(obs_dict)
                action = self.collector_interface.unnormalize(
                    noramlized_action,
                    self.collector_interface.action_stats["action"])
            else:
                action = policy(obs_dict)

            action = torch.as_tensor(action).to(self.device)
            # action[..., -1] = torch.sign(action[..., -1])

            next_obs, rewards, terminated, time_outs, extras = self.env.step(
                action.unsqueeze(0))

            last_obs = copy.deepcopy(next_obs)

        self.demo_index += 1

        return rewards > 10

    def process_diffusion(self, last_obs, obs_dict):
        if obs_dict == {}:

            for key in last_obs["policy"].keys():
                per_obs = last_obs["policy"][key][0][None]

                batch_obs = torch.cat([per_obs, per_obs], dim=0)

                obs_dict[key] = torch.as_tensor(batch_obs).to(self.device)
            return obs_dict

        for key in last_obs["policy"].keys():
            if key == "gs_image":

                obs_dict[key][-1] = torch.as_tensor(
                    resize_image(
                        last_obs["policy"]["gs_image"][0, 0].cpu().numpy(),
                        (224, 224))).to(self.device).permute(2, 0, 1) / 255
            else:

                obs_dict[key][-1] = torch.as_tensor(
                    last_obs["policy"][key]).to(self.device)
        return obs_dict

    def close_loop_policy(self, policy, rl_env):

        last_obs = self.reset()

        gs_actions = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["actions"]
        gs_actions = np.array(gs_actions)
        obs_buffer = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["obs"]
        if self.args_cli.model_type == "bc":
            pass
        else:
            obs_dict = self.process_diffusion(last_obs, obs_dict={})

        for i in range(gs_actions.shape[0]):

            if self.args_cli.model_type == "bc":

                obs_dict = last_obs["policy"]
                if self.args_cli.use_time:
                    obs_dict["timestep"] = torch.as_tensor([i]).to(
                        self.device).unsqueeze(0)

            else:
                obs_dict = self.process_diffusion(last_obs, obs_dict)

            if self.args_cli.normalize_action:

                # obs_dict["joint_pos"] = torch.as_tensor(
                #     obs_buffer["joint_pos"][i]).unsqueeze(0).to(self.device)

                # obs_dict["ee_pose"] = torch.tensor([[ 4.0000e-01, -1.2048e-06,  4.5000e-01,  3.3927e-06,  1.0000e+00, -1.4073e-06,  1.4967e-06]], device='cuda:0')
                # obs_dict["yellow_cube_state"] = torch.tensor([[ 4.9190e-01, -1.4698e-01,  2.0295e-02,  9.9965e-01, -9.7938e-06,  8.5455e-06, -2.6288e-02]], device='cuda:0')
                noramlized_action = policy(obs_dict)
                action = self.collector_interface.unnormalize(
                    noramlized_action,
                    self.collector_interface.action_stats["action"])
            else:
                action = policy(obs_dict)

            action = torch.as_tensor(action).to(self.device)
            if not self.args_cli.eval:

                next_obs, rewards, terminated, time_outs, extras = self.env.step(
                    action.unsqueeze(0))
            else:

                next_obs, rewards, terminated, time_outs, extras, _ = rl_env.step(
                    action.unsqueeze(0) * 0.0, action.unsqueeze(0))

            del last_obs

            last_obs = copy.deepcopy(next_obs)

        self.demo_index += 1
        print(rewards)

        return rewards > 10
