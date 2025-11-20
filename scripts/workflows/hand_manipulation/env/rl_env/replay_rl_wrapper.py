import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp

from isaaclab.utils.math import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import numpy as np
from scripts.workflows.hand_manipulation.utils.cloudxr.utils import reset_root_state_uniform
import copy
from tools.visualization_utils import vis_pc, visualize_pcd
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer, filter_out_data
from source.isaaclab.isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from tools.visualization_utils import vis_pc, visualize_pcd
from scripts.workflows.hand_manipulation.env.bc_env.zarr_replay_env_wrapper import ZarrReplayWrapper


class ReplayRLWrapper(ZarrReplayWrapper):

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
    ):

        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.device = env.device
        self.num_envs = env.num_envs
        self.use_delta_pose = False if "Rel" not in self.args_cli.task else True
        self.hand_side = "right" if self.add_right_hand else "left"

        self.target_object_name = f"{self.hand_side}_hand_object"
        self.demo_index = 0

        super().__init__(
            env,
            env_cfg,
            args_cli,
        )

        self.demo_index = self.collector_interface.traj_count

        self.num_arm_actions = 6

        self.init_data_buffer()

        self.init_setting()

    def init_data_buffer(self):

        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

    def replay(self, ):
        self.reset_env()

        self.last_finger_pose = self.env.scene[
            f"{self.hand_side}_hand"]._data.joint_pos[..., -16:].clone()
        for act in self.raw_data["actions"]:
            act = torch.as_tensor(act).to(self.env.device).unsqueeze(0)

            if self.args_cli.use_relative_finger_pose:
                act[:, -16:] += self.last_finger_pose
            self.env.step(act)

            self.last_finger_pose = self.env.scene[
                f"{self.hand_side}_hand"]._data.joint_pos[..., -16:].clone()
