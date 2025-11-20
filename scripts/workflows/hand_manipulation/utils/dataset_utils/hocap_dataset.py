# DexYCB Toolkit
# Copyright (C) 2021 NVIDIA Corporation
# Licensed under the GNU General Public License v3.0 [see LICENSE for details]
# Modified by Yuzhe Qin to use the sequential information inside the dataset
"""DexYCB dataset."""

from pathlib import Path

import numpy as np
import yaml
import torch
from ruamel.yaml import YAML
import os
from scripts.workflows.hand_manipulation.utils.dex_retargeting.mano_layer import MANOLayer
from manopth import demo

from scipy.spatial.transform import Rotation as R

_SUBJECTS = [
    'subject_2', 'subject_4', 'subject_3', 'subject_5', 'subject_8',
    'subject_7', 'subject_6', 'subject_9', 'subject_1'
]

HOCAO_CLASSES = {
    'G09_3', 'G07_3', 'G04_1', 'G16_2', 'G20_4', 'G21_4', 'G09_2', 'G01_2',
    'G15_1', 'G07_1', 'G22_2', 'G06_1', 'G21_1', 'G16_3', 'G22_4', 'G11_3',
    'G05_1', 'G10_4', 'G19_3', 'G18_2', 'G07_4', 'G10_3', 'G18_1', 'G18_4',
    'G16_4', 'G02_2', 'G21_2', 'G01_3', 'G06_4', 'G02_3', 'G11_4', 'G02_1',
    'G19_1', 'G19_2', 'G16_1', 'G04_2', 'G21_3', 'G04_3', 'G15_4', 'G10_1',
    'G04_4', 'G20_3', 'G22_1', 'G09_1', 'G20_1', 'G19_4', 'G11_1', 'G10_2',
    'G01_4', 'G05_3', 'G07_2', 'G15_3', 'G02_4', 'G01_1', 'G15_2', 'G18_3',
    'G11_2', 'G06_3', 'G05_2', 'G05_4', 'G06_2', 'G22_3', 'G09_4', 'G20_2'
}


def read_data_from_yaml(file_path):
    """Read data from a YAML file and return it."""

    yaml = YAML()
    yaml.default_flow_style = False
    yaml.indent(mapping=2, sequence=4, offset=2)
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        with open(str(file_path), "r", encoding="utf-8") as f:
            return yaml.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading YAML file from {file_path}: {e}")


def quat_to_mat(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion and translation vector to a pose matrix.

    This function supports converting a single quaternion or a batch of quaternions.

    Args:
        quat (np.ndarray): Quaternion and translation vector. Shape can be (7,) for a single quaternion
                           or (N, 7) for a batch of quaternions, where N is the batch size.

    Returns:
        np.ndarray: Pose matrix. Shape will be (4, 4) for a single quaternion or (N, 4, 4) for a batch of quaternions.

    Raises:
        ValueError: If the input does not have the expected shape or dimensions.
    """
    # Validate input shape
    if not isinstance(quat, np.ndarray):
        raise TypeError("Input must be a numpy array.")

    if quat.ndim == 1 and quat.shape[0] == 7:
        batch_mode = False
    elif quat.ndim == 2 and quat.shape[1] == 7:
        batch_mode = True
    else:
        raise ValueError(
            "Input must have shape (7,) for a single quaternion or (N, 7) for a batch of quaternions."
        )

    # Extract quaternion (q) and translation (t)
    q = quat[..., :4]  # Quaternion (4 elements)
    t = quat[..., 4:]  # Translation (3 elements)

    # Prepare the pose matrix
    if batch_mode:
        N = quat.shape[0]
        p = np.tile(np.eye(4), (N, 1, 1))  # Create N identity matrices
    else:
        p = np.eye(4)  # Single identity matrix

    # Convert quaternion to rotation matrix and fill in the pose matrix
    r = R.from_quat(q)
    p[..., :3, :3] = r.as_matrix()  # Fill rotation part
    p[..., :3, 3] = t  # Fill translation part

    return p.astype(np.float32)


class HOCAPDatasetLoader:

    def __init__(self,
                 data_dir,
                 hand_type="right",
                 filter_objects=[],
                 device="cuda"):
        self._data_dir = Path(data_dir)
        self._calib_folder = self._data_dir.parent.parent / "calibration"
        self._models_folder = self._data_dir.parent.parent / "models"
        self._device = device
        # Filter
        self.use_filter = len(filter_objects) > 0

        self._crop_lim = [-0.60, +0.60, -0.35, +0.35, -0.01, +0.80]
        self.hand_type = hand_type
        self.list_all_data()
        self.hand_index = 0 if hand_type == "right" else 1

    def list_all_data(self):
        # List all data folders
        self._data_folders = []
        for subject in _SUBJECTS:
            subject_folder = self._data_dir / subject
            if not subject_folder.is_dir():
                continue

            for dataset in subject_folder.iterdir():
                if dataset.is_dir():

                    file_path = Path(dataset) / "meta.yaml"
                    data = read_data_from_yaml(file_path)
                    mano_side = data["mano_sides"]
                    # if self.hand_type in mano_side and len(mano_side) == 1:
                    for i in range(4):

                        self._data_folders.append(dataset)

        self._data_folders.sort()

    def filter_out_of_bound(self, transformed_poes, object_index):
        segment_frame_id = []
        num_frames = transformed_poes.shape[0]

        for index in range(transformed_poes.shape[1]):
            init_object_height = transformed_poes[0, index, 2, 3]
            all_height = transformed_poes[:, index, 2, 3]
            lift_index = np.argmax(all_height - init_object_height)
            lower_bound_index = np.where(
                abs(all_height - init_object_height) < 0.02)[0]

            down_count = lower_bound_index[lower_bound_index > lift_index]

            if len(down_count) > 0:
                seg_id = np.clip(
                    lower_bound_index[lower_bound_index > lift_index][0] + 10,
                    0, num_frames)
            else:
                seg_id = num_frames - 1

            segment_frame_id.append(seg_id)
        segment_frame_id = np.array(segment_frame_id)

        end_frame = segment_frame_id[object_index]
        if end_frame == np.max(segment_frame_id):
            end_frame = num_frames - 1

        possible_start_frame = segment_frame_id[segment_frame_id < end_frame]

        if len(possible_start_frame) == 0:
            start_frame = 0

        elif len(possible_start_frame) == 4:
            start_frame = np.sort(possible_start_frame)[-2] + 20

        else:
            start_frame = np.max(possible_start_frame) + 20

        assert start_frame < end_frame, f"start frame {start_frame} >= end frame {end_frame}"
        # start_frame, end_frame = 0, num_frames - 1
        # if (end_frame - start_frame) < 30:
        #     import pdb
        #     pdb.set_trace()

        return start_frame, end_frame

    def lood_data(self, index):
        # Load data from all folders

        data_path = self._data_folders[index]
        self._load_metadata(data_path)
        hand_betas = self._load_mano_beta(data_path)
        # self._load_extrinsics("extrinsics_20231014.yaml")

        data_buffer = {}

        # TODO: check the transformation
        poses = np.load(os.path.join(str(data_path), "poses_o.npy"), )

        transformed_poes = np.stack([quat_to_mat(p) for p in poses], axis=1)
        start_frame, end_frame = self.filter_out_of_bound(
            transformed_poes, index % 4)

        data_buffer["hand_pose"] = np.load(
            os.path.join(
                str(data_path),
                "poses_m.npy"), )[self.hand_index][start_frame:end_frame]

        data_buffer["object_pose"] = transformed_poes[start_frame:end_frame]
        data_buffer["hand_state"] = np.load(
            os.path.join(
                str(data_path),
                "poses_pv.npy"), )[self.hand_index][start_frame:end_frame]

        data_buffer["hand_shape"] = hand_betas
        data_buffer["mano_sides"] = self._mano_sides
        data_buffer["object_ids"] = self._object_ids
        # data_buffer["extrinsics"] = self._extr2world

        return data_buffer

    def _load_mano_beta(self, data_path) -> torch.Tensor:

        subject_name = str(data_path).split("/")[-2]
        file_path = self._data_dir / "calibration" / "mano" / f"{subject_name}.yaml"
        data = read_data_from_yaml(file_path)
        return np.array(data["betas"]).astype(np.float32)

    def _load_metadata(self, _data_folder):

        data = read_data_from_yaml(Path(_data_folder) / "meta.yaml")

        self._num_frames = data["num_frames"]
        self._object_ids = data["object_ids"]
        self._mano_sides = data["mano_sides"]
        self._task_id = data["task_id"]
        self._subject_id = data["subject_id"]
        # RealSense camera metadata
        self._rs_serials = data["realsense"]["serials"]
        self._rs_width = data["realsense"]["width"]
        self._rs_height = data["realsense"]["height"]
        self._num_cams = len(self._rs_serials)
        # HoloLens metadata
        self._hl_serial = data["hololens"]["serial"]
        self._hl_pv_width = data["hololens"]["pv_width"]
        self._hl_pv_height = data["hololens"]["pv_height"]
        # Object models file paths
        self._object_textured_files = [
            self._models_folder / obj_id / "textured_mesh.obj"
            for obj_id in self._object_ids
        ]
        self._object_cleaned_files = [
            self._models_folder / obj_id / "cleaned_mesh_10000.obj"
            for obj_id in self._object_ids
        ]

    def _load_intrinsics(self):

        def read_K_from_yaml(serial, cam_type="color"):
            yaml_file = self._calib_folder / "intrinsics" / f"{serial}.yaml"
            data = read_data_from_yaml(yaml_file)[cam_type]
            K = np.array(
                [
                    [data["fx"], 0.0, data["ppx"]],
                    [0.0, data["fy"], data["ppy"]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            return K

        rs_Ks = np.stack(
            [read_K_from_yaml(serial) for serial in self._rs_serials], axis=0)
        rs_Ks_inv = np.stack([np.linalg.inv(K) for K in rs_Ks], axis=0)

        hl_K = read_K_from_yaml(self._hl_serial)
        hl_K_inv = np.linalg.inv(hl_K)

        # Convert intrinsics to torch tensors
        self._rs_Ks = torch.from_numpy(rs_Ks).to(self._device)
        self._rs_Ks_inv = torch.from_numpy(rs_Ks_inv).to(self._device)
        self._hl_K = torch.from_numpy(hl_K).to(self._device)
        self._hl_K_inv = torch.from_numpy(hl_K_inv).to(self._device)

    def _load_extrinsics(self, file_name):

        def create_mat(values):
            return np.array(
                [values[0:4], values[4:8], values[8:12], [0, 0, 0, 1]],
                dtype=np.float32)

        data = read_data_from_yaml(self._calib_folder / "extrinsics" /
                                   f"{file_name}")

        # Read rs_master serial
        self._rs_master = data["rs_master"]

        # Create extrinsics matrices
        extrinsics = data["extrinsics"]
        tag_0 = create_mat(extrinsics["tag_0"])
        tag_0_inv = np.linalg.inv(tag_0)
        tag_1 = create_mat(extrinsics["tag_1"])
        tag_1_inv = np.linalg.inv(tag_1)
        extr2master = np.stack(
            [create_mat(extrinsics[s]) for s in self._rs_serials], axis=0)
        extr2master_inv = np.stack([np.linalg.inv(t) for t in extr2master],
                                   axis=0)
        extr2world = np.stack([tag_1_inv @ t for t in extr2master], axis=0)
        extr2world_inv = np.stack([np.linalg.inv(t) for t in extr2world],
                                  axis=0)

        # Convert extrinsics to torch tensors
        self._tag_0 = torch.from_numpy(tag_0).to(self._device)
        self._tag_0_inv = torch.from_numpy(tag_0_inv).to(self._device)
        self._tag_1 = torch.from_numpy(tag_1).to(self._device)
        self._tag_1_inv = torch.from_numpy(tag_1_inv).to(self._device)
        self._extr2master = torch.from_numpy(extr2master).to(self._device)
        self._extr2master_inv = torch.from_numpy(extr2master_inv).to(
            self._device)
        self._rs_RTs = torch.from_numpy(extr2world).to(self._device)
        self._rs_RTs_inv = torch.from_numpy(extr2world_inv).to(self._device)


if __name__ == "__main__":

    dataset = HOCAPDatasetLoader("/media/ensu/data/datasets/HO-Cap/datasets")
    print(len(dataset._data_folders))
    for i in range(len(dataset._data_folders)):

        demo_data = dataset.lood_data(i)
    hand_pose_frame = demo_data["hand_pose"]

    mano_layer = MANOLayer(demo_data["mano_sides"][0], demo_data["hand_shape"])

    p = torch.from_numpy(hand_pose_frame[0, :48].astype(
        np.float32)).unsqueeze(0)
    t = torch.from_numpy(hand_pose_frame[0, 48:51].astype(
        np.float32)).unsqueeze(0)

    hand_verts, hand_poses = mano_layer(p, t)

    demo.display_hand({
        'verts': hand_verts,
        'joints': hand_poses
    },
                      mano_faces=mano_layer._mano_layer.th_faces)
