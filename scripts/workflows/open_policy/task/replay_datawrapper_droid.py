from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
import numpy as np
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer

import copy

import imageio
from tqdm import tqdm


class ReplayDatawrapperDroid:

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
            normalize_action=False)
        if args_cli.save_path is not None:
            self.collector_interface.init_collector_interface()
        self.args_cli = args_cli
        self.use_relative_pose = use_relative_pose
        self.env_config = env_config

        if self.use_relative_pose:
            self.reset_actions = 0 * torch.rand(env.action_space.shape,
                                                device=self.device)

        self.num_collected_demo = len(
            self.collector_interface.raw_data["data"])

        self.demo_index = 0
        reset_buffer(self)
        self.rigbid_objects_list = self.env_config["params"]["Task"][
            "reset_object_names"]
        self.env_ids = torch.arange(self.env.num_envs, device=self.device)

    def reset(self):
        self.env.reset()
        self.cur_obs = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["obs"]

        for name in self.rigbid_objects_list:
            asset = self.env.scene.rigid_objects[name]

            init_pose = torch.as_tensor(
                self.cur_obs[f"{name}_state"][0]).unsqueeze(0).to(self.device)
            asset.data.reset_root_state[:, :7] = init_pose[:, :7]

            # set into the physics simulation
            asset.write_root_link_pose_to_sim(init_pose[:, :7],
                                              env_ids=self.env_ids)
            # asset.write_root_com_velocity_to_sim(init_pose[:, 7:],
            #                                      env_ids=self.env_ids)

        init_robot_joint_pos = torch.as_tensor(
            self.cur_obs["joint_pos"][0]).unsqueeze(0).to(self.device)

        for i in range(self.env_config["params"]["Task"]["reset_horizon"]):

            need_dof_pos = self.env.scene[
                "robot"].root_physx_view.get_dof_positions()
            need_dof_pos[:, :8] = init_robot_joint_pos
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

    def success_or_not(self, next_obs, i, reset_frame, total_frame, rewards):

        # if next_obs["policy"]["eggplant_state"][:, 1] > 0.05 and reset_frame:
        if rewards > 10 and reset_frame:
            total_frame = np.min([i + 20, total_frame])
            self.pbar.total = total_frame  # Dynamically update tqdm total
            self.pbar.refresh()  # Refresh tqdm to show the updated total
            reset_frame = False
        # return reset_frame, total_frame, next_obs["policy"][
        #     "eggplant_state"][:, 1] > 0.05
        return reset_frame, total_frame, rewards > 10

    def step(self, last_obs, video_name=None):
        image_buffer = []

        actions = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["actions"]
        actions = torch.as_tensor(np.array(actions)).to(self.device)
        rewards_buffer = []

        total_frame = actions.shape[0]

        self.pbar = tqdm(total=total_frame,
                         desc="Processing frames")  # Initialize tqdm
        reset_frame = True
        i = 0
        while i < total_frame:

            next_obs, rewards, terminated, time_outs, extras = self.env.step(
                actions[i].unsqueeze(0).reshape(-1, actions.shape[-1]))

            rewards_buffer.append(rewards)
            reset_frame, total_frame, success = self.success_or_not(
                next_obs, i, reset_frame, total_frame, rewards)

            update_buffer(self, next_obs, last_obs, actions[i], rewards,
                          terminated or success, time_outs)
            if isinstance(last_obs["policy"], dict):
                if "gs_image" in last_obs["policy"].keys():
                    image_buffer.append(
                        last_obs["policy"]["gs_image"][0][0].cpu().numpy())
            last_obs = copy.deepcopy(next_obs)
            i += 1
            self.pbar.update(1)  # Increment the progress bar

        success = rewards > 10
        if video_name is not None:
            self.collector_interface.save_video(f"{video_name}_{success[0]}",
                                                image_buffer)
        if self.args_cli.save_path is not None:
            if success.cpu().numpy()[0]:
                self.collector_interface.add_demonstraions_to_buffer(
                    self.obs_buffer, self.action_buffer, self.rewards_buffer,
                    self.does_buffer, self.next_obs_buffer)
        reset_buffer(self)
        self.demo_index += 1

        return success
