import pickle
import time

import tyro
from avp_stream import VisionProStreamer
from scripts.workflows.hand_manipulation.utils.dex_retargeting.visionpro_utils import *
from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import get_hand_joint_names, init_leap_hand_retarget, display_hand
#Set this to your IP address of you AVP

import numpy as np


class LeapvHandBunnyVisionProTeleop():
    """
    This class is used to stream the AVP data and run inverse kinematics to output LEAP Hand joint angles.
    """

    def __init__(self,
                 record: bool = True,
                 use_left: bool = False,
                 use_right: bool = True,
                 AVP_IP="10.0.0.160"):
        self.vps = VisionProStreamer(ip=AVP_IP, record=record)
        self.use_left_hand = use_left
        self.use_right_hand = use_right
        self.num_hand_joints = 16
        self.init_retargeting_config()

    def init_retargeting_config(self):

        init_leap_hand_retarget(
            self,
            kinematics_path=
            # "source/config/task/hand_env/teleoperation/bunny/kinematics_config/bimanual_free_leap_dexpilot.yml",
            "source/config/task/hand_env/teleoperation/bunny/kinematics_config/bimanual_leap_vector_offline.yml",
            add_left_hand=self.use_left_hand,
            add_right_hand=self.use_right_hand,
            retarget_type="vectorada")
        self.robot_joint_name = real_joints_names = [
            'j0', 'j1', 'j2', 'j3', 'j4', 'j15', 'j6', 'j7', 'j8', 'j9', 'j10',
            'j11', 'j12', 'j13', 'j14', 'j15'
        ]

        if isinstance(self.retargeting, list):
            self.left_retargeting = self.retargeting[0]
            self.right_retargeting = self.retargeting[1]

            self.retarget2sim = np.array([0, 1, 2, 3, 4, 5] + [
                self.left_retargeting.optimizer.target_joint_names.index(j)
                for j in self.robot_joint_name
            ])

        else:
            if self.use_left_hand:
                self.left_retargeting = self.retargeting
                self.retarget2sim = np.array([0, 1, 2, 3, 4, 5] + [
                    self.left_retargeting.optimizer.target_joint_names.index(j)
                    for j in self.robot_joint_name
                ])

            if self.use_right_hand:
                self.right_retargeting = self.retargeting
                self.retarget2sim = np.array([0, 1, 2, 3, 4, 5] + [
                    self.right_retargeting.optimizer.target_joint_names.index(
                        j) for j in self.robot_joint_name
                ])

    def update_last_retargeted_qpos(self, joint, retargeting):

        optimizer = retargeting.optimizer
        retargeting_type = optimizer.retargeting_type
        indices = optimizer.target_link_human_indices

        if retargeting_type == "POSITION":
            ref_value = joint[indices, :]
        else:
            # TODO: need to debug the retargeting for the vectorada
            origin_indices = indices[0, :]
            task_indices = indices[1, :]
            ref_value = joint[task_indices, :] - joint[origin_indices, :]
            ref_value[[0, 2, 4, 6], 1] *= 2.0
            ref_value[[1, 5], 1] *= 1.0
            ref_value[[3, 7], 1] *= 3.0
            # ref_value[2, 1] *= 1.3
            # ref_value[3, 1] *= 1.3

        retargeted_qpos = retargeting.retarget(ref_value)
        return retargeted_qpos[self.retarget2sim]

    def retarget_hand(self, hand_pose, transform, hand_side='right'):
        joint_pose = two_mat_batch_mul(hand_pose, transform)
        joint_pos = joint_avp2hand(joint_pose)[:, :3, 3]
        retarget = getattr(self, f"{hand_side}_retargeting")
        hand_pose = self.update_last_retargeted_qpos(joint_pos, retarget)
        return hand_pose

    def get_avp_data(self):
        #gets the data converts it and then computes IK and visualizes
        data = self.vps.latest
        if self.use_left_hand:
            hand_pose = self.retarget_hand(
                (np.asarray(data['left_fingers'])).astype(float),
                OPERATOR2AVP_LEFT.T,
                hand_side='left')
            return hand_pose

        if self.use_right_hand:
            hand_pose = self.retarget_hand(
                (np.asarray(data['right_fingers'])).astype(float),
                OPERATOR2AVP_RIGHT.T,
                hand_side='right')
            return hand_pose


if __name__ == "__main__":
    teleoperate_client = LeapvHandBunnyVisionProTeleop()
    all_data = []
    for i in range(1000):
        print(f"Iteration {i}")
        data = teleoperate_client.get_avp_data()
        all_data.append(data)

        time.sleep(0.01)
        np.save("logs/avp_hand_data.npy", np.array(all_data))
