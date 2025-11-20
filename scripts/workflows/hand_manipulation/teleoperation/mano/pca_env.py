import torch
import zmq
import json

from dex_retargeting.retargeting_config import RetargetingConfig

from pathlib import Path
import yaml

import numpy as np

from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import get_hand_joint_names, init_leap_hand_retarget, load_pca_data, display_hand, reconstruct_mano_pose45

from scripts.workflows.hand_manipulation.utils.dex_retargeting.mano_layer import MANOLayer
from manopth.manolayer import ManoLayer


class ManoTOhandPCA:

    def __init__(self,
                 env,
                 arg_cli,
                 env_config,
                 pca_fn="/media/ensu/data/CrossDex/results/pca_9_grab.pkl"):
        self.env = env
        self.arg_cli = arg_cli
        self.env_config = env_config
        self.add_right_hand = arg_cli.add_right_hand
        self.add_left_hand = arg_cli.add_left_hand
        self.device = env.device

        if self.add_right_hand:
            self.hand_side = "right"
        elif self.add_left_hand:
            self.hand_side = "left"

        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]
        self.principal_vectors, self.min_values, self.max_values, self.D_mean, self.D_std = load_pca_data(
            pca_fn)
        self.init_setting()

    def init_setting(self):
        self.left_robot_joint_names, self.right_robot_joint_names = None, None

        self.init_retargeting_config()

        self.mano_layer = ManoLayer(
            mano_root=
            'scripts/workflows/hand_manipulation/utils/manopth/mano_v1_2/models',
            use_pca=False,
            ncomps=45,
            flat_hand_mean=True)

        pose = torch.zeros([1, 48])
        self.init_pose45 = reconstruct_mano_pose45(self.principal_vectors,
                                                   self.D_mean, self.D_std,
                                                   self.max_values)

        pose[0, 3:] = self.init_pose45

        self.hand_verts0, self.hand_joints0 = self.mano_layer(pose)

        display_hand({
            "hand_info": {
                'verts': torch.as_tensor(self.hand_verts0[0]),
                'joints': torch.as_tensor(self.hand_joints0[0]),
                "faces": self.mano_layer.th_faces
            },
        })

    def init_retargeting_config(self):

        init_leap_hand_retarget(self, )
        self.robot_joint_name = get_hand_joint_names(
            self,
            self.hand_side,
        )[-self.num_hand_joints:]

        self.retarget2sim = [
            self.retargeting.optimizer.target_joint_names.index(j)
            for j in self.robot_joint_name
        ]

    def update_last_retargeted_qpos(self, joint):
        retargeting = self.retargeting
        optimizer = retargeting.optimizer
        retargeting_type = optimizer.retargeting_type
        indices = optimizer.target_link_human_indices

        if retargeting_type == "POSITION":
            ref_value = joint[indices, :]
        else:
            origin_indices = indices[0, :]
            task_indices = indices[1, :]

            ref_value = joint[task_indices, :] - joint[origin_indices, :]
        # ref_value = joint[indices, :]

        qpos = retargeting.retarget(ref_value)[self.retarget2sim]

        # retargeted_qpos = retargeting.retarget(ref_value)[self.robot_reorder]
        return qpos

    def run(self):

        hand_pose = torch.as_tensor(
            self.update_last_retargeted_qpos(
                self.hand_joints0[0].cpu().numpy())).unsqueeze(0).to(
                    self.device)
        for i in range(100):
            actions = torch.zeros(self.env.action_space.shape).to(self.device)
            actions[:, -self.num_hand_joints:] = hand_pose
            self.env.step(actions)
