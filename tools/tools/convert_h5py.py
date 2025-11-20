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

from tools.caliberation_json import *

import trimesh


def save_image(rgb_image, seg_image, target_color, dir_name, gs_name,
               frame_name, img_idx):

    frontground_mask = np.all(seg_image == target_color, axis=-1)

    background_mask = ~frontground_mask

    front_ground = np.ones_like(rgb_image) * 0
    back_ground = np.ones_like(rgb_image) * 0

    front_ground[frontground_mask] = rgb_image[frontground_mask]
    back_ground[background_mask] = rgb_image[background_mask]

    cv2.imwrite(
        f"{dir_name}/{gs_name}/seg_imgs/{frame_name}_{img_idx:05d}.jpg",
        front_ground[..., ::-1])
    cv2.imwrite(f"{dir_name}/{gs_name}/bg_imgs/{frame_name}_{img_idx:05d}.jpg",
                back_ground[..., ::-1])
    cv2.imwrite(
        f"{dir_name}/{gs_name}/raw_imgs/{frame_name}_{img_idx:05d}.jpg",
        rgb_image[..., ::-1])
    cv2.imwrite(f"{dir_name}/{gs_name}/mask/{frame_name}_{img_idx:05d}.jpg",
                frontground_mask.astype(np.int32) * 255)


def save_frames_info(rgb_dataset,
                     seg_data_dataset,
                     target_color,
                     dir_name,
                     gs_name,
                     extrinsic_dataset,
                     out,
                     intrinsic_dataset,
                     target_mesh_path,
                     frame_name="static_frame"):
    transformed_poses = []
    frames = []
    out_static = {}
    for img_idx in range(rgb_dataset.shape[0]):
        rgb_image = rgb_dataset[img_idx]
        seg_image = seg_data_dataset[img_idx]

        save_image(rgb_image, seg_image, target_color, dir_name, gs_name,
                   frame_name, img_idx)
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

        # if img_idx == 0:

        #     scene = trimesh.load(target_mesh_path)
        #     geometries = list(scene.geometry.values())
        #     vertices = geometries[0].vertices
        #     ones = np.ones((vertices.shape[0], 1))
        #     ones[:, 0] += 0.9
        #     homogeneous_points = np.hstack([vertices, ones])
        #     transformed_points_homogeneous = homogeneous_points @ c2w.T
        #     bb_max, bb_min = np.max(
        #         transformed_points_homogeneous,
        #         axis=0)[:3], np.min(transformed_points_homogeneous, axis=0)[:3]
        #     out["bbox"] = np.array([bb_max, bb_min]).tolist()
        #     out_static["bbox"] = np.array([bb_max, bb_min]).tolist()

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

    # for index, _ in enumerate(frames):
    #     I = np.eye(4)
    #     I[:3, :3] = poses[index][:3, :3]
    #     frames[index]["transform_matrix"] = I.tolist()

    if "frames" in out.keys():
        out["frames"] += frames
    else:
        out["frames"] = frames

    out_static = get_intrinsic(intrinsic_dataset, rgb_dataset, out_static)

    out_static["frames"] = copy.deepcopy(frames)
    with open(Path(dir_name + f"/{gs_name}") / "transforms.json",
              "w",
              encoding="utf-8") as f:

        for img_idx, frame in enumerate(out_static["frames"]):
            out_static["frames"][img_idx][
                "file_path"] = name = f"raw_imgs/{frame_name}_{img_idx:05d}.jpg"

        json.dump(out_static, f, indent=4)

    return out


def save_static_images(file, dir_name, gs_name, skip_frames, out, target_id,
                       target_mesh_path):

    rgb_data = file["data"]["demo_0"]["obs"]["rgb"][skip_frames:, :, :, :, :3]
    extrinsic_params = file["data"]["demo_0"]["obs"]["extrinsic_params"][
        skip_frames:]
    intrinsic_params = file["data"]["demo_0"]["obs"]["intrinsic_params"][
        skip_frames:]
    seg_data = file["data"]["demo_0"]["obs"]["semantic_segmentation"][
        skip_frames:, :, :, :, :3]

    rgb_dataset = rgb_data.reshape(-1, rgb_data.shape[2], rgb_data.shape[3], 3)
    seg_data_dataset = seg_data.reshape(-1, rgb_data.shape[2],
                                        rgb_data.shape[3], 3)
    extrinsic_dataset = extrinsic_params.reshape(-1, extrinsic_params.shape[2],
                                                 extrinsic_params.shape[3])
    intrinsic_dataset = intrinsic_params.reshape(-1, intrinsic_params.shape[2],
                                                 intrinsic_params.shape[3])

    out["num_static_frame"] = rgb_data.shape[0]
    out["num_static_timestep"] = rgb_data.shape[0]
    id_to_color_map = generate_id_map_from_rgb(seg_data_dataset[0])
    out = get_intrinsic(intrinsic_dataset, rgb_dataset, out)

    target_color = id_to_color_map[target_id]
    save_frames_info(rgb_dataset, seg_data_dataset, target_color, dir_name,
                     gs_name, extrinsic_dataset, out, intrinsic_dataset,
                     target_mesh_path)

    return out


def save_dynamics_images(file, dir_name, gs_name, out, target_id,
                         target_mesh_path):

    rgb_data = file["data"]["demo_0"]["obs"]["rgb"][3:50, :, :, :, :3]
    extrinsic_params = file["data"]["demo_0"]["obs"]["extrinsic_params"][2:50]
    intrinsic_params = file["data"]["demo_0"]["obs"]["intrinsic_params"][2:50]
    seg_data = file["data"]["demo_0"]["obs"]["semantic_segmentation"][
        3:50, :, :, :, :3]

    num_cameras = rgb_data.shape[1]
    frames = []
    out["num_dynamics_frame"] = rgb_data.shape[0] * rgb_data.shape[1]
    out["num_dynamics_timestep"] = rgb_data.shape[0]

    for num_cam in range(num_cameras):

        rgb_dataset = rgb_data[:, num_cam]
        seg_data_dataset = seg_data[:, num_cam]
        extrinsic_dataset = extrinsic_params[:, num_cam]
        intrinsic_dataset = intrinsic_params[:, num_cam]

        out = get_intrinsic(intrinsic_dataset, rgb_dataset, out)
        id_to_color_map = generate_id_map_from_rgb(seg_data_dataset[0])
        target_color = id_to_color_map[target_id]

        out = save_frames_info(rgb_dataset,
                               seg_data_dataset,
                               target_color,
                               dir_name,
                               gs_name,
                               extrinsic_dataset,
                               out,
                               intrinsic_dataset,
                               target_mesh_path,
                               frame_name=f"gs_{num_cam}_frame")

    return out


def save_sperate(dir_name, dataset_type, skip_frames, static_target_id,
                 dynamics_target_id, target_mesh_path):

    info = {}
    os.makedirs(dir_name + "/dataset", exist_ok=True)

    for gs_name in ["dynamics_gs", "static_gs"]:
        if not os.path.exists(f"{dir_name}/{gs_name}"):
            continue

        os.makedirs(f"{dir_name}/{gs_name}/raw_imgs", exist_ok=True)
        os.makedirs(f"{dir_name}/{gs_name}/seg_imgs", exist_ok=True)
        os.makedirs(f"{dir_name}/{gs_name}/bg_imgs", exist_ok=True)
        os.makedirs(f"{dir_name}/{gs_name}/mask", exist_ok=True)
        with h5py.File(f"{dir_name}/{gs_name}/{dataset_type}.hdf5",
                       'r') as file:

            if gs_name == "static_gs":
                info = save_static_images(file, dir_name, gs_name, skip_frames,
                                          info, static_target_id,
                                          target_mesh_path)
            elif gs_name == "dynamics_gs":
                info = save_dynamics_images(file, dir_name, gs_name, info,
                                            dynamics_target_id,
                                            target_mesh_path)

    with open(Path(dir_name + "/dataset") / "transforms.json",
              "w",
              encoding="utf-8") as f:
        json.dump(info, f, indent=4)


def copy_image_to_dataset(dir_name):
    for gs_name in ["static_gs", "dynamics_gs"]:
        image_list = []

        path = str(dir_name) + f"/{gs_name}"
        if not os.path.exists(path):
            continue
        image_path = path + "/seg_imgs/"

        image_filenames, num_orig_images = get_image_filenames(
            Path(image_path), max_num_images=1000)
        image_list += image_filenames

        copied_image_paths = copy_images_list(
            image_list,
            image_dir=Path(dir_name + "/dataset/images/" + gs_name),
            verbose=True,
            num_downscales=1)


def convert_h5py(dir_name, dataset_type, skip_frames, static_target_id,
                 dynamics_target_id, target_mesh_path):
    save_sperate(dir_name, dataset_type, skip_frames, static_target_id,
                 dynamics_target_id, target_mesh_path)

    # Copy images to output directory
    copy_image_to_dataset(dir_name)


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
                        default=10,
                        help="skip the frames")
    parser.add_argument("--keep_original_world_coordinate",
                        action="store_true",
                        help="Keep original world coordinate")

    args = parser.parse_args()

    dir_name = args.dir_name
    dataset_type = args.type

    #covert h5py
    convert_h5py(dir_name,
                 dataset_type,
                 args.skip_frames,
                 static_target_id=2,
                 dynamics_target_id=2,
                 target_mesh_path="source/assets/Plush/usd/cat/cat.glb")

    #metashape setting
    # for gs_name in ["static_gs"]:
    #     # write_yaml("isaac_sim_alignment",
    #     #            f"{dir_name}/metashape/running_{gs_name}",
    #     #            f"{dir_name}/{gs_name}",
    #     #            output_file=f"{dir_name}/metashape/config_{gs_name}.yaml")

    #     # process = AutomatedProcessing()
    #     # process.read_config(f"{dir_name}/metashape/config_{gs_name}.yaml")
    #     # process.init_workspace()
    #     # process.init_tasks()

    #     # # unzip the process result
    #     folders = [
    #         name
    #         for name in os.listdir(f"{dir_name}/metashape/running_{gs_name}")
    #         if os.path.isdir(
    #             os.path.join(f"{dir_name}/metashape/running_{gs_name}", name))
    #     ]
    #     folders.sort()
    #     unzip_and_move(
    #         f"{dir_name}/metashape/running_{gs_name}/{folders[-1]}/0/chunk.zip",
    #         f"{dir_name}/metashape/running_{gs_name}/{folders[-1]}/",
    #     )

    #     # convert metashape to json
    #     convert_metashape_data(
    #         data=f"{dir_name}/",
    #         xml=f"{dir_name}/metashape/running_{gs_name}/{folders[-1]}/doc.xml",
    #         output_dir=f"{dir_name}/metashape/",
    #         max_dataset_size=1000)

    # # unzip and move the point0.ply
    # unzip_and_move(
    #     f"{dir_name}/metashape/running/{folders[-1]}/0/0/point_cloud/point_cloud.zip",
    #     f"{dir_name}/metashape/running/{folders[-1]}/0/0/point_cloud/",
    #     move_to=f"{dir_name}/metashape/",
    #     move_filename=f"points0.ply")


if __name__ == "__main__":
    main()
