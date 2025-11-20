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


class GrabDatasetLoader:

    def __init__(
        self,
        data_dir,
        add_translation=False,
    ):

        grab_s_folders = os.listdir(data_dir + "/hand_dataset")
        self.data_dir = data_dir

        data_list = []
        for folder in grab_s_folders:
            all_seqs = glob.glob(data_dir + "/hand_dataset/" + folder +
                                 '/*.npz')
            data_list.append(all_seqs)

        self.data_list = [item for sublist in data_list for item in sublist]
        self.add_translation = add_translation

    def __len__(self):
        return len(self.data_list)

    def extract_hand_info(
        self,
        info,
        side='right',
        n_comps=24,
    ):
        info["params"]["global_orient"] *= 0.0
        info["params"]["transl"][..., :3] *= 0.0
        info["params"]["transl"][..., 2] = 0.10
        hand_vtem_file = "/".join(info["vtemp"].split("/")[-2:])

        hand_vertices = trimesh.load(
            f"{self.data_dir}/people/{hand_vtem_file}")

        mano_layer = ManoLayer(
            mano_root=
            'scripts/workflows/hand_manipulation/utils/manopth/mano_v1_2/models',
            use_pca=True,
            flat_hand_mean=True,
            side=side,
            ncomps=n_comps)

        mano_layer.th_v_template[0] = torch.as_tensor(hand_vertices.vertices)

        hand_finger_pose = np.concatenate(
            [info["params"]["global_orient"], info["params"]["hand_pose"]],
            axis=1)
        hand_finger_translate = torch.as_tensor(info["params"]["transl"])
        hand_beta_file = hand_vtem_file.split(".")[0] + "_betas.npy"
        hand_betas = torch.as_tensor(
            np.load(f"{self.data_dir}/people/{hand_beta_file}"))

        hand_verts, hand_joints = mano_layer(
            torch.as_tensor(hand_finger_pose).to(dtype=torch.float32),
            hand_betas.unsqueeze(0).to(dtype=torch.float32).repeat_interleave(
                len(hand_finger_pose), dim=0), hand_finger_translate)

        hand_global_orient = torch.as_tensor(info["params"]["global_orient"])
        hand_global_orient = math_utils.quat_from_matrix(
            batch_rodrigues(hand_global_orient.view(-1, 3)).view(
                [hand_global_orient.shape[0], 3, 3]))
        hand_pose = torch.cat(
            [torch.as_tensor(info["params"]["transl"]), hand_global_orient],
            dim=1)

        return hand_verts / 1000, hand_joints / 1000, mano_layer.th_faces, hand_pose

    def get_mano_parameters(self, info):

        rhand = info["rhand"].item()
        lhand = info["lhand"].item()
        rhand_verts, rhand_joints, rhand_faces, rhand_pose = self.extract_hand_info(
            rhand, side='right', n_comps=info["n_comps"])
        lhand_verts, lhand_joints, lhand_faces, lhand_pose = self.extract_hand_info(
            lhand, side='left', n_comps=info["n_comps"])
        return rhand_verts, rhand_joints, lhand_verts, lhand_joints, rhand_faces, lhand_faces, rhand_pose, lhand_pose

    def get_object_parameters(self, object_info):
        trasl = torch.as_tensor(object_info["params"]["transl"])
        global_orient = torch.as_tensor(object_info["params"]["global_orient"])
        rot_quat = math_utils.quat_from_matrix(
            batch_rodrigues(global_orient.view(-1, 3)).view(
                [global_orient.shape[0], 3, 3]))

        object_transformation = torch.cat([trasl, rot_quat], dim=1)
        object_mesh_file = object_info["object_mesh"].split("/")[-1]
        mesh = trimesh.load(
            f"{self.data_dir}//contact_mesh/{object_mesh_file}")

        return torch.as_tensor(mesh.vertices), torch.as_tensor(
            mesh.faces), torch.as_tensor(object_transformation), object_info[
                "object_mesh"].split("/")[-1].split(".")[0]

    def __getitem__(self, item):
        if item > self.__len__():
            raise ValueError(f"Index {item} out of range")
        file_name = self.data_list[item]
        info = np.load(file_name, allow_pickle=True)
        rhand_verts, rhand_joints, lhand_verts, lhand_joints, rhand_faces, lhand_faces, rhand_pose, lhand_pose = self.get_mano_parameters(
            info)
        object_verts, object_faces, object_transformation, object_name = self.get_object_parameters(
            info["object"].item())

        grab_data = dict(
            rhand_verts=rhand_verts,
            rhand_joints=rhand_joints,
            lhand_verts=lhand_verts,
            lhand_joints=lhand_joints,
            rhand_faces=rhand_faces,
            lhand_faces=lhand_faces,
            object_verts=object_verts,
            object_faces=object_faces,
            object_transformation=object_transformation,
            object_name=object_name,
            rhand_pose=rhand_pose,
            lhand_pose=lhand_pose,
        )

        trnasformed_vertices = math_utils.transform_points(
            object_verts.unsqueeze(0),
            object_transformation[200, :3].unsqueeze(0),
            object_transformation[200, 3:7].unsqueeze(0))

        # display_hand({
        #     "hand_info": {
        #         'verts': rhand_verts[0],
        #         'joints': rhand_joints[0],
        #         "faces": rhand_faces
        #     },
        #     # "obj_info": {
        #     #     "verts": trnasformed_vertices[0],
        #     #     "faces": object_faces
        #     # }
        # })

        return grab_data


def main():
    from collections import Counter

    dataset = GrabDatasetLoader(f"{self.data_dir}//hand_dataset")
    print(len(dataset))

    sample = dataset[0]
    print(sample.keys())


if __name__ == "__main__":
    main()
