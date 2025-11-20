import h5py
import cv2
import os

import isaaclab.utils.math as math_utils
import numpy as np
from pathlib import Path
import json
import torch
import math
import argparse
import copy

import sys

sys.path.append(".")
from tools.caliberation_json import *

from tools.metashape_tool import *

import numpy as np

import yaml

import shutil
import zipfile

import json
import xml.etree.ElementTree as ET

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from typing import List, Literal, Optional, OrderedDict, Tuple, Union
import cv2
import imageio
try:
    import rawpy
except ImportError:
    import newrawpy as rawpy  # type: ignore
import collections
import struct
from enum import Enum

import trimesh


class DataConvertion:

    def __init__(self,
                 dir_name,
                 dataset_type,
                 skip_frames,
                 fg_id=None,
                 bg_id=None,
                 target_mesh_path=None):
        self.dir_name = dir_name
        self.dataset_type = dataset_type
        self.skip_frames = skip_frames
        self.target_mesh_path = target_mesh_path

        self.fg_id = fg_id
        self.bg_id = bg_id

    def convert_h5py(self):
        self.save_sperate_data()

    def save_sperate_data(self):

        for gs_name in ["dynamics_gs", "static_gs"]:

            if not os.path.exists(f"{self.dir_name}/{gs_name}"):
                continue

            with h5py.File(
                    f"{self.dir_name}/{gs_name}/{self.dataset_type}.hdf5",
                    'r') as self.h5py_file:

                if gs_name == "static_gs":
                    info = self.save_static_images(gs_name)

    def make_dir(self, dir_name):
        os.makedirs(dir_name, exist_ok=True)

    def make_folder(self, gs_name, dataset_id, camera_id):
        self.make_dir(
            f"{self.dir_name}/{gs_name}/raw_imgs/dataset_{dataset_id}/cam_{camera_id}",
        )
        self.make_dir(
            f"{self.dir_name}/{gs_name}/seg_imgs/dataset_{dataset_id}/cam_{camera_id}"
        )
        self.make_dir(
            f"{self.dir_name}/{gs_name}/bg_imgs/dataset_{dataset_id}/cam_{camera_id}"
        )
        self.make_dir(
            f"{self.dir_name}/{gs_name}/fg_imgs/dataset_{dataset_id}/cam_{camera_id}"
        )

    def load_obs_data(self, per_batch_dataset, camera_id):
        rgb_data = per_batch_dataset["obs"]["rgb"][camera_id][
            self.skip_frames:, ..., :3]
        seg_data = per_batch_dataset["obs"]["seg_rgb"][camera_id][
            self.skip_frames:, ..., :3]
        segmentation = per_batch_dataset["obs"]["segmentation"][camera_id][
            self.skip_frames:]
        extrinsic_params = per_batch_dataset["obs"]["extrinsic_params"][
            camera_id][self.skip_frames:]
        intrinsic_params = per_batch_dataset["obs"]["intrinsic_params"][
            camera_id][self.skip_frames:]

        rgb_dataset = rgb_data.reshape(-1, *rgb_data.shape[2:])
        seg_data_dataset = seg_data.reshape(-1, *rgb_data.shape[2:])
        segmentation = segmentation.reshape(-1, *segmentation.shape[2:])

        extrinsic_dataset = extrinsic_params.reshape(
            -1, *extrinsic_params.shape[2:])
        intrinsic_dataset = intrinsic_params.reshape(
            -1, *intrinsic_params.shape[2:])

        return rgb_dataset, seg_data_dataset, segmentation, extrinsic_dataset, intrinsic_dataset

    def save_static_images(self, gs_name):
        num_demo = len(self.h5py_file["data"].keys())
        num_camera = len(self.h5py_file["data"][f"demo_0"]["obs"]["rgb"])

        for dataset_id in range(num_demo):
            info = {}
            for camera_id in range(num_camera):
                self.make_folder(gs_name, dataset_id, camera_id)

                per_batch_dataset = self.h5py_file["data"][
                    f"demo_{dataset_id}"]
                rgb_dataset, seg_data_dataset, segmentation, extrinsic_dataset, intrinsic_dataset = self.load_obs_data(
                    per_batch_dataset, camera_id)

                info["num_static_frame"] = rgb_dataset.shape[0]
                info["num_static_timestep"] = rgb_dataset.shape[0]

                info = self.get_intrinsic(intrinsic_dataset, rgb_dataset, info)

                self.save_frames_info(rgb_dataset, seg_data_dataset,
                                      segmentation, gs_name, extrinsic_dataset,
                                      info, dataset_id, camera_id)
            self.save_info_for_multiple_cameras(info,
                                                gs_name,
                                                dataset_id,
                                                camera_id,
                                                frame_name="static_frame")

        return info

    def get_intrinsic(self, intrinsic_dataset, rgb_dataset, info):

        intrinsic = intrinsic_dataset[0]

        info["w"] = rgb_dataset[0].shape[1]
        info["h"] = rgb_dataset[0].shape[0]
        info["fl_x"] = float(intrinsic[0][0])
        info["fl_y"] = float(intrinsic[0][0])
        info["cx"] = float(intrinsic[0][2])
        info["cy"] = float(intrinsic[1][2])
        info["k1"] = float(0.0)
        info["k2"] = float(0.0)
        info["p1"] = float(0.0)
        info["p2"] = float(0.0)
        info["camera_model"] = "OPENCV"

        return info

    def save_frames_info(self,
                         rgb_dataset,
                         seg_data_dataset,
                         segmentation,
                         gs_name,
                         extrinsic_dataset,
                         info,
                         dataset_id,
                         camera_id,
                         frame_name="static_frame"):
        transformed_poses = []
        frames = []

        for img_idx in range(rgb_dataset.shape[0]):
            rgb_image = rgb_dataset[img_idx]
            seg_image = seg_data_dataset[img_idx]
            seg_id = segmentation[img_idx]

            self.save_image(rgb_image, seg_image, seg_id, gs_name, frame_name,
                            img_idx, dataset_id, camera_id)
            from scipy.spatial.transform import Rotation as R
            rotate_matrix = np.eye(4)

            w2c = np.eye(4)
            w2c[:3, 3] = extrinsic_dataset[img_idx][:3, 3]

            r = R.from_matrix(extrinsic_dataset[img_idx][:3, :3])

            # # Convert to Euler angles
            # # You can specify the order of rotations, e.g., 'xyz', 'zyx', etc.
            euler_angles0 = r.as_euler('xyz', degrees=True)
            # print(euler_angles0)
            rotate_matrix[:3, :3] = math_utils.matrix_from_quat(
                obtain_target_quat_from_multi_angles([2, 0], [
                    -euler_angles0[2] / 180 * np.pi - np.pi,
                    -euler_angles0[0] / 180 * np.pi
                ])).numpy()

            w2c = rotate_matrix @ w2c
            c2w = np.linalg.inv(w2c)

            name = f"images/{gs_name}/{frame_name}_{img_idx:05d}.jpg"
            name = Path(name)
            transformed_poses.append(c2w)

            frame = {
                "file_path": name.as_posix(),
                "transform_matrix": c2w.tolist(),
                "colmap_im_id": img_idx,
            }

            frames.append(frame)

        poses = np.array(transformed_poses).astype(np.float32)
        poses, transform_matrix = auto_orient_and_center_poses(
            poses,
            method="up",
            center_method="poses",
        )

        # Scale poses
        scale_factor = 1.0

        scale_factor /= float(np.max(np.abs(poses[:, :3, 3])))

        poses[:, :3, 3] *= scale_factor

        if "frames" in info.keys():
            info["frames"] += frames
        else:
            info["frames"] = frames

        return info

    def save_info_for_multiple_cameras(self, info, gs_name, dataset_id,
                                       camera_id, frame_name):
        for cam_obs in ["seg_imgs", "raw_imgs"]:
            with open(Path(self.dir_name +
                           f"/{gs_name}/{cam_obs}/dataset_{dataset_id}/") /
                      "transforms.json",
                      "w",
                      encoding="utf-8") as f:
                for img_idx, frame in enumerate(info["frames"]):
                    info["frames"][img_idx][
                        "file_path"] = f"cam_{camera_id}/cam_{camera_id}_{frame_name}_{img_idx:05d}.jpg"

                json.dump(info, f, indent=4)

    def save_image(self, rgb_image, seg_image, seg_id, gs_name, frame_name,
                   img_idx, dataset_id, camera_id):

        cv2.imwrite(
            f"{self.dir_name}/{gs_name}/seg_imgs/dataset_{dataset_id}/cam_{camera_id}/cam_{camera_id}_{frame_name}_{img_idx:05d}.jpg",
            seg_image[..., ::-1])

        cv2.imwrite(
            f"{self.dir_name}/{gs_name}/raw_imgs/dataset_{dataset_id}/cam_{camera_id}/cam_{camera_id}_{frame_name}_{img_idx:05d}.jpg",
            rgb_image[..., ::-1])

        # unique_id = np.unique(seg_id)

        fg_img = self.seg_iamge(seg_image, seg_id, self.fg_id[dataset_id])
        cv2.imwrite(
            f"{self.dir_name}/{gs_name}/fg_imgs/dataset_{dataset_id}/cam_{camera_id}/cam_{camera_id}_{frame_name}_{img_idx:05d}.jpg",
            fg_img[..., ::-1])

        bg_img = self.seg_iamge(seg_image, seg_id, self.bg_id[dataset_id])
        cv2.imwrite(
            f"{self.dir_name}/{gs_name}/bg_imgs/dataset_{dataset_id}/cam_{camera_id}/cam_{camera_id}_{frame_name}_{img_idx:05d}.jpg",
            bg_img[..., ::-1])

    def seg_iamge(self, seg_image, seg_id, target_id):

        tg = np.where(seg_id == target_id, True, False)[..., None]
        tg = np.concatenate([tg, tg, tg], axis=-1)
        tg_img = np.where(tg, seg_image, 0)
        return tg_img


class MetaShapeCaliberation:

    def __init__(self, dir_name):
        self.dir_name = dir_name
        self.config_metashape()
        self.caliberation = AutomatedProcessing()

    def process(self):
        num_dataset = len(os.listdir(f"{self.dir_name}/static_gs/raw_imgs"))
        for dataset_id in range(num_dataset):
            self.align_bg_and_fg()
            for obs_name in ["raw_imgs", "seg_imgs"]:
                self.caliberation.read_config(
                    f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}.yaml"
                )
                self.caliberation.init_workspace()
                self.caliberation.init_tasks()
                self.upzip_and_move_result(dataset_id, obs_name)
                if obs_name == "seg_imgs":

                    for tg_name in ["bg_imgs", "fg_imgs"]:
                        self.copy_single_file(
                            f"{self.dir_name}/metashape/{tg_name}/dataset_{dataset_id}/transforms.json",
                            f"{self.dir_name}/metashape/seg_imgs/dataset_{dataset_id}/transforms.json"
                        )
                        self.copy_single_file(
                            f"{self.dir_name}/metashape/{tg_name}/dataset_{dataset_id}/sparse_pc.ply",
                            f"{self.dir_name}/metashape/seg_imgs/dataset_{dataset_id}/sparse_pc.ply"
                        )

    def align_bg_and_fg(self):
        self.move_files(f"{self.dir_name}/metashape/bg_imgs",
                        f"{self.dir_name}/static_gs/bg_imgs")
        self.move_files(f"{self.dir_name}/metashape/fg_imgs",
                        f"{self.dir_name}/static_gs/fg_imgs")

    def copy_single_file(self, destination_file, source_file):
        if os.path.exists(destination_file):
            os.remove(destination_file)  # Remove the existing file

        shutil.copyfile(source_file, destination_file)

    def upzip_and_move_result(self, dataset_id, obs_name):

        # # unzip the process result
        folders = [
            name for name in os.listdir(
                f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}/"
            ) if os.path.isdir(
                os.path.join(
                    f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}/",
                    name))
        ]
        folders.sort()

        unzip_and_move(
            f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}/{folders[-1]}/0/0/point_cloud/point_cloud.zip",
            f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}/{folders[-1]}/0/0/point_cloud/",
            move_to=
            f"{self.dir_name}/metashape/{obs_name}/dataset_{dataset_id}",
            move_filename=f"points0.ply")

        unzip_and_move(
            f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}/{folders[-1]}/0/chunk.zip",
            f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}/{folders[-1]}/",
        )

        # convert metashape to json
        self.convert_metashape_data(
            xml=
            f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}/{folders[-1]}/doc.xml",
            dataset_id=dataset_id,
            obs_name=obs_name)

    def convert_metashape_data(self,
                               xml,
                               dataset_id,
                               obs_name,
                               max_dataset_size=300):

        # Copy images to output directory

        path = f"{self.dir_name}/metashape"

        image_path = path + f"/{obs_name}/dataset_{dataset_id}"
        cam_folders = [
            name for name in os.listdir(image_path)
            if os.path.isdir(os.path.join(image_path, name))
        ]
        image_list = []

        for cam_file in cam_folders:

            image_filenames, num_orig_images = get_image_filenames(
                Path(f"{image_path}/{cam_file}"), max_dataset_size)

            image_list += image_filenames

        image_path = [
            Path(*image_file.parts[-2:]) for image_file in image_list
        ]
        image_name = [
            image_file.name.split(".")[0] for image_file in image_list
        ]
        image_filename_map = dict(zip(image_name, image_path))

        metashape_to_json(
            image_filename_map=image_filename_map,
            xml_filename=xml,
            output_dir=Path(path + f"/{obs_name}/dataset_{dataset_id}"),
            ply_filename=Path(path +
                              f"/{obs_name}/dataset_{dataset_id}/points0.ply"),
            verbose=True,
        )

    def move_files(self, destination_folder, source_folder):
        os.makedirs(destination_folder, exist_ok=True)
        # Check if the destination directory exists
        if os.path.exists(destination_folder):
            # Remove the existing destination directory
            shutil.rmtree(destination_folder)
            print(f"Existing destination folder {destination_folder} removed.")

        # Copy the source folder to the destination
        shutil.copytree(source_folder, destination_folder)

    def config_metashape(self):

        num_dataset = len(os.listdir(f"{self.dir_name}/static_gs/raw_imgs"))

        os.makedirs(f"{self.dir_name}/metashape", exist_ok=True)

        self.move_files(f"{self.dir_name}/metashape/raw_imgs",
                        f"{self.dir_name}/static_gs/raw_imgs")
        self.move_files(f"{self.dir_name}/metashape/seg_imgs",
                        f"{self.dir_name}/static_gs/seg_imgs")

        for dataset_id in range(num_dataset):
            for obs_name in ["raw_imgs", "seg_imgs"]:

                os.makedirs(
                    f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}",
                    exist_ok=True)

                config = {
                    "run_name":
                    "static_metashape_caliberation",  # User-defined run name
                    "load_project_path": "",  # Remains empty
                    "project_path":
                    f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}",  # User-defined project path
                    "project_crs": "EPSG::32633",  # Default value
                    "subdivide_task": True,  # Default value
                    "enable_overwrite": False,  # Default value
                    "addPhotos": {
                        "enabled": True,  # Default value
                        "photo_path":
                        f"{self.dir_name}/metashape/{obs_name}/dataset_{dataset_id}",  # User-defined photo path
                        "remove_photo_location_metadata":
                        False,  # Default value
                        "multispectral": False  # Default value
                    },
                    "alignPhotos": {
                        "enabled": True,  # Default value
                        "downscale": 0,  # Default value
                        "adaptive_fitting": True,  # Default value
                        "filter_mask_tiepoints": True,  # Default value
                        "double_alignment": True  # Default value
                    },
                }

                # Write the YAML structure to a file
                with open(
                        f"{self.dir_name}/metashape/config/{obs_name}/dataset_{dataset_id}.yaml",
                        'w') as yaml_file:
                    yaml.dump(config, yaml_file, default_flow_style=False)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Process some images and camera data.")
    parser.add_argument("--dir_name",
                        type=str,
                        help="Directory name where the data is stored")
    parser.add_argument("--type",
                        type=str,
                        default="cat",
                        help="Type of the dataset")
    parser.add_argument("--skip_frames",
                        type=int,
                        default=0,
                        help="skip the frames")
    parser.add_argument("--keep_original_world_coordinate",
                        action="store_true",
                        help="Keep original world coordinate")

    args = parser.parse_args()

    dir_name = args.dir_name
    dataset_type = args.type

    #covert h5py
    # converter = DataConvertion(
    #     dir_name,
    #     dataset_type,
    #     args.skip_frames,
    #     fg_id=[2, 6],
    #     bg_id=[4, 8],
    #     target_mesh_path="source/assets/Plush/usd/rabbit/rabbit.glb")
    # converter.convert_h5py()

    metashape_caliberation = MetaShapeCaliberation(dir_name)
    metashape_caliberation.process()


if __name__ == "__main__":
    main()
