import numpy as np
import torch

from scipy.spatial.transform import Rotation as R
from scripts.workflows.hand_manipulation.utils.cloudxr.leap_dex_retargeting_utils import LeapDexRetargeting


class TestDataWrapper:

    def __init__(self, env, env_cfg, args_cli):
        self.env = env
        self.env_cfg = env_cfg
        self.args_cli = args_cli
        self.load_data()
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.num_hand_joints = self.env_cfg["params"]["num_hand_joints"]
        self.init_retargeting()
        self.demo_index = 0

    def init_retargeting(self):
        self.retarget = LeapDexRetargeting(
            env=self.env,
            add_left_hand=self.add_left_hand,
            add_right_hand=self.add_right_hand,
        )

    def extract_hand_data(self):

        left_hand_data = self.teleop_data[self.demo_index]["left_hand_joint"]
        right_hand_data = self.teleop_data[self.demo_index]["right_hand_joint"]
        actions = []
        if self.add_left_hand:

            left_hand_joint = self.retarget.compute_left(left_hand_data)
            left_hand_joint[:6] = 0
            actions.append(left_hand_joint)
        if self.add_right_hand:
            right_hand_joint = self.retarget.compute_right(right_hand_data)
            right_hand_joint[:6] = 0
            actions.append(right_hand_joint)

        self.demo_index += 1
        return torch.as_tensor(np.concatenate(actions, axis=0)).to(
            self.env.device).unsqueeze(0)

    def step_data(self):
        actions = self.extract_hand_data()

        # actions = torch.rand(self.env.action_space.shape,
        #                      device=self.env.unwrapped.device)
        self.env.step(actions)

    def load_data(self, ):

        self.teleop_data = np.load(self.args_cli.test_data_dir,
                                   allow_pickle=True)["arr_0"]
