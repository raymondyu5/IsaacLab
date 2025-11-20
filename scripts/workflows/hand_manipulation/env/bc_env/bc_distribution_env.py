from scripts.workflows.hand_manipulation.env.bc_env.bc_env_wrapper import HandBCEnvWrapper
import torch

import isaaclab.utils.math as math_utils
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

import os
from isaaclab.envs import mdp
from collections import defaultdict


class BCDistributionEnv(HandBCEnvWrapper):

    def __init__(self, env, config, args_cli, step=0.02):
        super().__init__(env, config, args_cli)

        self.exploration_type = args_cli.exploration_type
        self.env_config
        self.use_delta_pose = True if "Rel" in self.args_cli.task else False
        self.env_ids = torch.arange(self.env.unwrapped.num_envs,
                                    device=self.device)
        self.counter = 0
        self.success_distribution = defaultdict(list)
        self.step = step

    def draw_trajectory(self, trajectories, result_path, init_object_pose):

        num_traj, horizon, _ = trajectories.shape
        time = np.arange(horizon)

        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

        titles = ['X Trajectories', 'Y Trajectories', 'Z Trajectories']

        for i, ax in enumerate(axes):
            for traj in range(num_traj):
                ax.plot(time, trajectories[traj, :, i], alpha=0.3)
            ax.set_ylabel(['x', 'y', 'z'][i])
            ax.set_title(titles[i])
            ax.grid(True)

        axes[-1].set_xlabel('Timestep')
        plt.tight_layout()

        name = init_object_pose.tolist()
        name = [f"{pos:.2f}" for pos in name]

        plt.savefig(f"{result_path}/trajectories_x_{name[0]}_y_{name[1]}.png")

    def draw_3d_trajectory(self, trajectories):

        num_traj, horizon, _ = trajectories.shape

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for traj in range(num_traj):
            ax.plot(trajectories[traj, :, 0],
                    trajectories[traj, :, 1],
                    trajectories[traj, :, 2],
                    alpha=0.3)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Trajectories')

        plt.tight_layout()

        plt.show()

    def eval_policy_distribution(self, result_path):
        all_trajectries = []

        with torch.no_grad():

            while True:
                last_obs, init_object_pose = self.reset_distribution()

                self.diffusion_env.last_diffusion_obs = self.get_diffusion_obs(
                    last_obs["policy"])

                trajectries = []

                for i in range(160):

                    ee_pose = self.env.scene[
                        f"{self.hand_side}_panda_link7"]._data.root_pos_w[:, :
                                                                          3].clone(
                                                                          )
                    ee_pose -= self.env.scene.env_origins

                    trajectries.append(ee_pose.cpu().numpy())
                    next_obs, rewards, terminated, time_outs, extras, predict_action = self.diffusion_env.step_diffusion(
                    )

                    last_obs = next_obs
                success = self.evaluate_success()
                print(f"Episode success:",
                      success.sum().item() / self.env.num_envs,
                      init_object_pose.tolist())

                trajectries = np.stack(trajectries, axis=1)
                all_trajectries.append(trajectries)

                name = init_object_pose.tolist()
                name = [f"{pos:.2f}" for pos in name]
                self.success_distribution[tuple(name)].append(
                    success.sum().item() / self.env.num_envs)

                self.sortout_result(
                    [all_trajectries[0][success.cpu().numpy()]],
                    init_object_pose, result_path)
                np.save(f"{result_path}/success_distribution.npy",
                        self.success_distribution)

            # self.draw_3d_trajectory(trajectries)

    def draw_success_heatmap(self, result_path):
        """
            Draw heatmap of success rate over the XY grid region
            based on self.success_distribution.
            """
        # get grid points (already built in reset_rigid_object_pose)
        grid_xy = self.grid_xy.cpu().numpy()

        # get dictionary of success values
        success_dict = self.success_distribution  # key: ('x', 'y') string pair or tuple

        # prepare empty heatmap

        x_vals = np.unique(grid_xy[:, 0])
        y_vals = np.unique(grid_xy[:, 1])
        heatmap = np.zeros(
            (len(y_vals), len(x_vals)))  # note: rows = y, cols = x

        # fill heatmap with success values if available, else 0
        for i, y in enumerate(y_vals):
            for j, x in enumerate(x_vals):
                key = (f"{x:.2f}", f"{y:.2f}")
                if tuple(key) in success_dict:
                    val_list = success_dict[tuple(key)]
                    heatmap[i, j] = np.mean(val_list)
                else:
                    heatmap[i, j] = 0.0

        # plot the heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(
            heatmap,
            origin='lower',
            extent=[x_vals.min(),
                    x_vals.max(),
                    y_vals.min(),
                    y_vals.max()],
            cmap='viridis',
            aspect='auto')
        plt.colorbar(label='Success Rate')
        plt.xlabel('X position (m)')
        plt.ylabel('Y position (m)')
        plt.title('Success Rate Heatmap over XY Grid')
        plt.tight_layout()

        plt.savefig(f"{result_path}/success_heatmap.png")
        plt.close()

    def sortout_result(
        self,
        all_trajectries,
        init_object_pose,
        result_path,
    ):

        # if self.exploration_type == "fixed":
        self.draw_trajectory(np.concatenate(all_trajectries, axis=0),
                             result_path, init_object_pose)
        if self.exploration_type == "evenly":

            self.draw_success_heatmap(result_path)

    def reset_distribution(self):
        self.env.reset()
        init_object_pose = self.reset_rigid_object_pose()

        if self.use_delta_pose:
            actions = torch.zeros(self.env.unwrapped.action_space.shape,
                                  dtype=torch.float32,
                                  device=self.device)

        else:

            actions = torch.as_tensor(
                self.env_config["params"].get("init_ee_pose")).to(
                    self.device).unsqueeze(0).repeat_interleave(
                        self.env.unwrapped.num_envs, dim=0)
            actions = torch.concat([
                actions,
                torch.zeros((self.env.unwrapped.num_envs, 16),
                            dtype=torch.float32,
                            device=self.device)
            ],
                                   dim=-1)

        for i in range(10):
            next_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
                actions)

        return next_obs, init_object_pose

    def reset_rigid_object_pose(self):
        rigid_object = self.env.scene[f"{self.hand_side}_hand_object"]

        rot_config = self.env_config["params"]["RigidObject"][
            self.args_cli.target_object_name]["rot"]
        fixed_quat = math_utils.obtain_target_quat_from_multi_angles(
            rot_config["axis"], rot_config["angles"]).unsqueeze(0).to(
                self.device).repeat_interleave(self.env.num_envs, dim=0)
        fixed_xyz = torch.as_tensor([
            self.env_config["params"]["RigidObject"][
                self.args_cli.target_object_name]["pos"]
        ]).to(self.device).repeat_interleave(self.env.num_envs, dim=0)
        if self.exploration_type == "fixed":
            init_pose = fixed_xyz[0, :2].cpu().numpy()

            fixed_xyz[:, :3] += self.env.scene.env_origins

            fixed_pose = torch.cat([fixed_xyz, fixed_quat], dim=-1)
            rigid_object.write_root_link_pose_to_sim(fixed_pose)
            return init_pose
        elif self.exploration_type == "evenly":
            pose_range = self.env_config["params"]["multi_cluster_rigid"][
                f"{self.hand_side}_hand_object"]["pose_range"]

            range_list = [
                pose_range.get(key, (0.0, 0.0))
                for key in ["x", "y", "z", "roll", "pitch", "yaw"]
            ]

            range_list = torch.as_tensor(range_list).to(self.device)

            # create grid for x and y
            x_vals = torch.arange(range_list[0, 0],
                                  range_list[0, 1] + self.step / 2, self.step)
            y_vals = torch.arange(range_list[1, 0],
                                  range_list[1, 1] + self.step / 2, self.step)

            # make meshgrid (XY plane)
            X, Y = torch.meshgrid(x_vals, y_vals, indexing='xy')

            # flatten to N×2 grid points
            self.grid_xy = torch.stack([X.flatten(), Y.flatten()], dim=-1)
            xy_pose = self.grid_xy[self.counter %
                                   self.grid_xy.shape[0]].unsqueeze(
                                       0).repeat_interleave(self.env.num_envs,
                                                            dim=0).to(
                                                                self.device)
            fixed_xyz[:, :2] += xy_pose + self.env.scene.env_origins[..., :2]
            fixed_pose = torch.cat([fixed_xyz, fixed_quat], dim=-1)
            rigid_object.write_root_link_pose_to_sim(fixed_pose)

            self.counter += 1
            init_pose = xy_pose[0, :2].cpu().numpy()
            return init_pose
