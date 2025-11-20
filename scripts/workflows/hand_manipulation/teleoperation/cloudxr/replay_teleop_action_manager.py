import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.utils.math import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import numpy as np
from scripts.workflows.hand_manipulation.utils.cloudxr.utils import reset_root_state_uniform

from scripts.workflows.hand_manipulation.teleoperation.cloudxr.replay_aug_wrapper import ReplayAugWrapper


class ReplayTeleopActionManager(ReplayAugWrapper):

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
        begin_index=4,
        skip_steps=1,
    ):
        super().__init__(env, env_cfg, args_cli, begin_index, skip_steps)

        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_cfg,
            save_path=args_cli.save_path,
            load_path=args_cli.load_path,
            save_zarr=True)

    def init_setting(self):

        init_ee_pose = torch.as_tensor(
            self.env_cfg["params"]["init_ee_pose"]).to(
                self.device).unsqueeze(0)
        self.num_hand_joint = self.env_cfg["params"]["num_hand_joints"]

        init_pose = torch.cat([
            init_ee_pose,
            torch.zeros(1, self.num_hand_joint).to(self.device)
        ],
                              dim=1)
        self.init_rigid_object_setting()

        self.init_actions = []
        if self.add_left_hand:
            self.init_actions.append(init_pose)
        if self.add_right_hand:
            self.init_actions.append(init_pose)
        self.init_actions = torch.cat(self.init_actions,
                                      dim=1).repeat_interleave(self.num_envs,
                                                               dim=0).to(
                                                                   self.device)

        self.action_range = torch.as_tensor(
            self.env_cfg["params"]["Task"]["action_range"]).to(
                self.device)[:2] * self.env.step_dt

        for i in range(30):
            if self.use_delta_pose:
                self.env.step(
                    torch.zeros(self.env.action_space.shape).to(self.device))

            else:

                self.env.step(self.init_actions)

        self.raw_data = self.collector_interface.raw_data["data"]
        self.demo_index = 0
        self.num_hand_joints = self.env_cfg["params"]["num_hand_joints"]
        arm_action_bound = torch.as_tensor(
            self.env_cfg["params"]["Task"]["action_range"]).to(self.device)
        arm_action_bound = torch.as_tensor(
            self.env_cfg["params"]["Task"]["action_range"]).to(self.device)

        arm_action_bound = torch.stack([
            torch.tensor(
                [-arm_action_bound[0]] * 3 + [-arm_action_bound[1]] * 3,
                device=self.device),
            torch.tensor([arm_action_bound[0]] * 3 + [arm_action_bound[1]] * 3,
                         device=self.device)
        ],
                                       dim=1)
        hand_action_bound = self.env.scene[f"right_hand"]._data.joint_limits[
            0, -self.num_hand_joint:]
        self.action_bound = torch.cat([arm_action_bound, hand_action_bound],
                                      dim=0)
