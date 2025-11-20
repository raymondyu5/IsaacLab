# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]
# Modified by Yuzhe Qin to use the sequential information inside the dataset
"""DexYCB dataset."""

from pathlib import Path

import numpy as np
import yaml
from manopth import demo
from manopth.manolayer import ManoLayer

import os
import glob
import torch
import trimesh
from scripts.workflows.hand_manipulation.utils.dex_retargeting.retarget_utils import display_hand
import isaaclab.utils.math as math_utils
from smplx.lbs import batch_rodrigues


class ArcticDatasetLoader:

    def __init__(self, data_dir, add_translation=False):

        grab_s_folders = os.listdir(data_dir)

        data_list = []
        for folder in grab_s_folders:
            all_seqs = glob.glob(data_dir + "/" + folder + '/*.mano.npy')
            data_list.append(all_seqs)

        self.data_list = [item for sublist in data_list for item in sublist]
        self.add_translation = add_translation

    def __len__(self):
        return len(self.data_list)

    def extract_hand_info(
        self,
        info,
        side='right',
    ):

        mano_layer = ManoLayer(
            mano_root=
            'scripts/workflows/hand_manipulation/utils/manopth/mano_v1_2/models',
            use_pca=False,
            flat_hand_mean=True,
            side=side,
        )

        hand_finger_pose = np.concatenate([info["trans"], info["rot"]], axis=1)
        hand_finger_translate = torch.as_tensor(info["pose"])
        all_pose = torch.zeros((hand_finger_translate.shape[0], 48))
        all_pose[:, 3:] = hand_finger_translate

        hand_verts, hand_joints = mano_layer(
            all_pose.to(dtype=torch.float32), )

        return hand_verts / 1000, hand_joints / 1000, mano_layer.th_faces

    def get_mano_parameters(self, info):

        rhand = info["right"]
        lhand = info["left"]
        rhand_verts, rhand_joints, rhand_faces = self.extract_hand_info(
            rhand, side='right')
        lhand_verts, lhand_joints, lhand_faces = self.extract_hand_info(
            lhand, side='left')
        return rhand_verts, rhand_joints, lhand_verts, lhand_joints, rhand_faces, lhand_faces

    def get_object_parameters(self, object_info):
        trasl = torch.as_tensor(object_info["params"]["transl"])
        global_orient = torch.as_tensor(object_info["params"]["global_orient"])
        rot_quat = math_utils.quat_from_matrix(
            batch_rodrigues(global_orient.view(-1, 3)).view(
                [global_orient.shape[0], 3, 3]))

        object_transformation = torch.cat([trasl, rot_quat], dim=1)
        object_mesh_file = object_info["object_mesh"].split("/")[-1]
        mesh = trimesh.load(
            f"/media/ensu/data/datasets/grab/contact_mesh/{object_mesh_file}")

        return torch.as_tensor(mesh.vertices), torch.as_tensor(
            mesh.faces), torch.as_tensor(object_transformation), object_info[
                "object_mesh"].split("/")[-1].split(".")[0]

    def __getitem__(self, item):
        if item > self.__len__():
            raise ValueError(f"Index {item} out of range")
        file_name = self.data_list[item]
        info = np.load(file_name, allow_pickle=True).item()
        seq_name = file_name.split("/")[-2]
        vtemp_mesh = trimesh.load(
            "/media/ensu/data/datasets/arctic/downloads/data/meta/subject_vtemplates/"
            + seq_name + ".obj")
        import pdb
        pdb.set_trace()
        rhand_verts, rhand_joints, lhand_verts, lhand_joints, rhand_faces, lhand_faces = self.get_mano_parameters(
            info)
        # object_verts, object_faces, object_transformation, object_name = self.get_object_parameters(
        #     info["object"].item())

        grab_data = dict(
            rhand_verts=rhand_verts,
            rhand_joints=rhand_joints,
            lhand_verts=lhand_verts,
            lhand_joints=lhand_joints,
            rhand_faces=rhand_faces,
            lhand_faces=lhand_faces,
            # object_verts=object_verts,
            # object_faces=object_faces,
            # object_transformation=object_transformation,
            # object_name=object_name,
            # rhand_pose=rhand_pose,
            # lhand_pose=lhand_pose,
        )

        display_hand({
            "hand_info": {
                'verts': rhand_verts[0],
                'joints': rhand_joints[0],
                "faces": rhand_faces
            },
            # "obj_info": {
            #     "verts": trnasformed_vertices[0],
            #     "faces": object_faces
            # }
        })

        return grab_data


def main():
    from collections import Counter

    dataset = ArcticDatasetLoader(
        "/media/ensu/data/datasets/arctic/downloads/data/raw_seqs")
    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())


if __name__ == "__main__":
    main()
