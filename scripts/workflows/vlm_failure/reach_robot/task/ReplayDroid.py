import h5py
import numpy as np
import torch


class ReplayDroidWrapper:

    def __init__(self, env, save_config, args_cli, use_ik_pose):
        self.env = env
        self.save_config = save_config
        self.args_cli = args_cli
        self.use_ik_pose = use_ik_pose

        data = h5py.File(self.args_cli.load_path, "r")
        self.action = data["action"]
        self.observation = data["observation"]
        self.device = self.args_cli.device
        self.env_ids = torch.arange(self.args_cli.num_envs).to(self.device)

    def reset_env(self):

        self.joint_positons = torch.as_tensor(
            np.array(self.observation["robot_state"]["joint_positions"])).to(
                self.device)
        for i in range(20):
            need_dof = self.env.scene[
                "robot"].root_physx_view.get_dof_positions()
            need_dof[:, :7] = self.joint_positons[0]
            self.env.scene["robot"].root_physx_view.set_dof_positions(
                need_dof, indices=self.env_ids)
            last_obs, rew, terminated, truncated, extras = self.env.step(
                torch.as_tensor(self.env.action_space.sample()).to(self.device)
                * 0.0)

        return last_obs

    def step_motion(self, last_obs):

        for i in range(self.joint_positons.shape[0]):
            need_dof = self.env.scene[
                "robot"].root_physx_view.get_dof_positions()
            need_dof[:, :7] = self.joint_positons[i]
            self.env.scene["robot"].root_physx_view.set_dof_positions(
                need_dof, indices=self.env_ids)

            last_obs, rew, terminated, truncated, extras = self.env.step(
                torch.as_tensor(self.env.action_space.sample()).to(self.device)
                * 0.0)

        return last_obs
