from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import init_leap_hand_retarget
import torch
import numpy as np


class HandRetargetWrapper:

    def __init__(self, args_cli, teleop_config):

        self.args_cli = args_cli

        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.teleop_config = teleop_config

        kinematics_path = self.teleop_config.get("kinematics_path", )
        retarget_type = self.teleop_config.get("retarget_type", None)

        init_leap_hand_retarget(
            self,
            kinematics_path=kinematics_path,
            add_left_hand=self.add_left_hand,
            add_right_hand=self.add_right_hand,
            retarget_type=retarget_type,
        )

        real_joints_names = [
            'j1', 'j12', 'j5', 'j9', 'j0', 'j13', 'j4', 'j8', 'j2', 'j14',
            'j6', 'j10', 'j3', 'j15', 'j7', 'j11'
        ]
        self.retarget2rea_index = [
            self.retargeting[0].joint_names.index(real_joint_name)
            for real_joint_name in real_joints_names
        ]

    def retarget_hand(self, joint_positions, hand_side):

        if hand_side == "left":
            retargeted_qpos = self.compute_ref_value(joint_positions,
                                                     self.retargeting[0])
        if hand_side == "right":
            retargeted_qpos = self.compute_ref_value(joint_positions,
                                                     self.retargeting[-1])
        return retargeted_qpos[self.retarget2rea_index]

    def compute_ref_value(
        self,
        joint,
        retargeting,
    ) -> np.ndarray:
        """Computes reference value for retargeting.

        Args:
            joint_position: Joint positions array
            indices: Target link indices
            retargeting_type: Type of retargeting ("POSITION" or other)

        Returns:
            Reference value in cartesian space
        """
        optimizer = retargeting.optimizer
        retargeting_type = optimizer.retargeting_type
        indices = optimizer.target_link_human_indices

        if retargeting_type == "POSITION":
            ref_value = joint[indices, :]
        else:

            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint[task_indices, :] - joint[origin_indices, :]
            ref_value[:, 1] *= 1.5
            ref_value[[3, 7, -1], 1] *= 1.5

        with torch.enable_grad():
            with torch.inference_mode(False):
                retargeted_qpos = retargeting.retarget(ref_value[..., :3])

                return retargeted_qpos
