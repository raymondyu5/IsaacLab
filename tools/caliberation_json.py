# import open3d as o3d
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

import imageio
try:
    import rawpy
except ImportError:
    import newrawpy as rawpy  # type: ignore
import collections
import struct
from enum import Enum

import os
import torch
import isaaclab.utils.math as math_utils


def generate_id_map_from_rgb(segmentation_rgb):
    # Get unique colors from the RGB segmentation result
    unique_colors = np.unique(segmentation_rgb.reshape(-1, 3), axis=0)

    # Create a mapping from RGB color to ID
    id_to_color_map = {idx: color for idx, color in enumerate(unique_colors)}

    return id_to_color_map


def obtain_target_quat_from_multi_angles(axis, angles):
    quat_list = []
    for index, cam_axis in enumerate(axis):
        euler_xyz = torch.zeros(3)
        euler_xyz[cam_axis] = angles[index]
        quat_list.append(
            math_utils.quat_from_euler_xyz(euler_xyz[0], euler_xyz[1],
                                           euler_xyz[2]))
    if len(quat_list) == 1:
        return quat_list[0]
    else:
        target_quat = quat_list[0]
        for index in range(len(quat_list) - 1):
            target_quat = math_utils.quat_mul(quat_list[index + 1],
                                              target_quat)
        return target_quat


def get_intrinsic(intrinsic_dataset, rgb_dataset, out):

    intrinsic = intrinsic_dataset[0]

    out["w"] = rgb_dataset[0].shape[1]
    out["h"] = rgb_dataset[0].shape[0]
    out["fl_x"] = float(intrinsic[0][0] / 2)
    out["fl_y"] = float(intrinsic[0][0] / 2)
    out["cx"] = float(intrinsic[0][2])
    out["cy"] = float(intrinsic[1][2])
    out["k1"] = float(0.0)
    out["k2"] = float(0.0)
    out["p1"] = float(0.0)
    out["p2"] = float(0.0)
    out["camera_model"] = "OPENCV"

    return out


def unzip_and_move(zip_file_path,
                   extract_to,
                   move_to=None,
                   move_filename=None):
    # Ensure the extraction and move directories exist
    os.makedirs(extract_to, exist_ok=True)

    # Unzip the file
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    if move_to is not None:

        if os.path.isfile(f"{move_to}/{move_filename}"):
            os.remove(f"{move_to}/{move_filename}")
        shutil.copy(f"{extract_to}/{move_filename}", move_to)


def _find_param(calib_xml: ET.Element, param_name: str):
    param = calib_xml.find(param_name)
    if param is not None:
        return float(param.text)  # type: ignore
    return 0.0


POLYCAM_UPSCALING_TIMES = 2
"""Lowercase suffixes to treat as raw image."""
ALLOWED_RAW_EXTS = [".cr2"]
"""Suffix to use for converted images from raw."""
RAW_CONVERTED_SUFFIX = ".jpg"


class CameraModel(Enum):
    """Enum for camera types."""

    OPENCV = "OPENCV"
    OPENCV_FISHEYE = "OPENCV_FISHEYE"
    EQUIRECTANGULAR = "EQUIRECTANGULAR"
    PINHOLE = "PINHOLE"
    SIMPLE_PINHOLE = "SIMPLE_PINHOLE"


CAMERA_MODELS = {
    "perspective": CameraModel.OPENCV,
    "fisheye": CameraModel.OPENCV_FISHEYE,
    "equirectangular": CameraModel.EQUIRECTANGULAR,
    "pinhole": CameraModel.PINHOLE,
    "simple_pinhole": CameraModel.SIMPLE_PINHOLE,
}


def copy_images_list(
    image_paths: List[Path],
    image_dir: Path,
    num_downscales: int,
    image_prefix: str = "frame_",
    crop_border_pixels: Optional[int] = None,
    crop_factor: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    verbose: bool = False,
    keep_image_dir: bool = False,
    upscale_factor: Optional[int] = None,
    nearest_neighbor: bool = False,
    same_dimensions: bool = True,
) -> List[Path]:
    """Copy all images in a list of Paths. Useful for filtering from a directory.
    Args:
        image_paths: List of Paths of images to copy to a new directory.
        image_dir: Path to the output directory.
        num_downscales: Number of times to downscale the images. Downscales by 2 each time.
        image_prefix: Prefix for the image filenames.
        crop_border_pixels: If not None, crops each edge by the specified number of pixels.
        crop_factor: Portion of the image to crop. Should be in [0,1] (top, bottom, left, right)
        verbose: If True, print extra logging.
        keep_image_dir: If True, don't delete the output directory if it already exists.
    Returns:
        A list of the copied image Paths.
    """

    # Remove original directory and its downscaled versions
    # only if we provide a proper image folder path and keep_image_dir is False

    image_dir.mkdir(exist_ok=True, parents=True)

    copied_image_paths = []

    # Images should be 1-indexed for the rest of the pipeline.
    for idx, image_path in enumerate(image_paths):

        copied_image_path = image_dir / f"{ image_paths[idx].stem}{image_path.suffix}"
        image_paths[idx].stem

        try:
            # if CR2 raw, we want to read raw and write RAW_CONVERTED_SUFFIX, and change the file suffix for downstream processing
            if image_path.suffix.lower() in ALLOWED_RAW_EXTS:
                copied_image_path = image_dir / f"{image_prefix}{idx + 1:05d}{RAW_CONVERTED_SUFFIX}"
                with rawpy.imread(str(image_path)) as raw:
                    rgb = raw.postprocess()
                imageio.imsave(copied_image_path, rgb)
                image_paths[idx] = copied_image_path
            elif same_dimensions:
                # Fast path; just copy the file
                shutil.copy(image_path, copied_image_path)
            else:
                # Slow path; let ffmpeg perform autorotation (and clear metadata)
                ffmpeg_cmd = f"ffmpeg -y -i {image_path} -metadata:s:v:0 rotate=0 {copied_image_path}"

        except shutil.SameFileError:
            pass
        copied_image_paths.append(copied_image_path)

    nn_flag = "" if not nearest_neighbor else ":flags=neighbor"
    downscale_chains = [
        f"[t{i}]scale=iw/{2**i}:ih/{2**i}{nn_flag}[out{i}]"
        for i in range(num_downscales + 1)
    ]
    downscale_dirs = [
        Path(str(image_dir) + (f"_{2**i}" if i > 0 else ""))
        for i in range(num_downscales + 1)
    ]

    for dir in downscale_dirs:
        dir.mkdir(parents=True, exist_ok=True)

    downscale_chain = (f"split={num_downscales + 1}" +
                       "".join([f"[t{i}]"
                                for i in range(num_downscales + 1)]) + ";" +
                       ";".join(downscale_chains))

    num_frames = len(image_paths)
    image_paths.sort()
    # ffmpeg batch commands assume all images are the same dimensions.
    # When this is not the case (e.g. mixed portrait and landscape images), we need to do individually.
    # (Unfortunately, that is much slower.)
    for framenum in range(1, (1 if same_dimensions else num_frames) + 1):
        framename = f"{image_paths[framenum].stem}" if same_dimensions else f"{image_prefix}{framenum:05d}"
        ffmpeg_cmd = f'ffmpeg -y -noautorotate -i "{image_dir / f"{framename}{copied_image_paths[0].suffix}"}" '

        crop_cmd = ""
        if crop_border_pixels is not None:
            crop_cmd = f"crop=iw-{crop_border_pixels*2}:ih-{crop_border_pixels*2}[cropped];[cropped]"
        elif crop_factor != (0.0, 0.0, 0.0, 0.0):
            height = 1 - crop_factor[0] - crop_factor[1]
            width = 1 - crop_factor[2] - crop_factor[3]
            start_x = crop_factor[2]
            start_y = crop_factor[0]
            crop_cmd = f"crop=w=iw*{width}:h=ih*{height}:x=iw*{start_x}:y=ih*{start_y}[cropped];[cropped]"

        select_cmd = "[0:v]"
        if upscale_factor is not None:
            select_cmd = f"[0:v]scale=iw*{upscale_factor}:ih*{upscale_factor}:flags=neighbor[upscaled];[upscaled]"

        downscale_cmd = f' -filter_complex "{select_cmd}{crop_cmd}{downscale_chain}"' + "".join(
            [
                f' -map "[out{i}]" -q:v 2 "{downscale_dirs[i] / f"{framename}{copied_image_paths[0].suffix}"}"'
                for i in range(num_downscales + 1)
            ])

        ffmpeg_cmd += downscale_cmd

    return copied_image_paths


def list_images(data: Path, recursive: bool = True) -> List[Path]:
    """Lists all supported images in a directory

    Args:
        data: Path to the directory of images.
        recursive: Whether to search check nested folders in `data`.
    Returns:
        Paths to images contained in the directory
    """
    allowed_exts = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]
    glob_str = "**/[!.]*" if recursive else "[!.]*"

    image_paths = sorted(
        [p for p in data.glob(glob_str) if p.suffix.lower() in allowed_exts])
    return image_paths


def get_image_filenames(directory: Path, max_num_images: int = -1):
    """Returns a list of image filenames in a directory.

    Args:
        dir: Path to the directory.
        max_num_images: The maximum number of images to return. -1 means no limit.
    Returns:
        A tuple of A list of image filenames, number of original image paths.
    """
    image_paths = list_images(directory)
    num_orig_images = len(image_paths)

    if max_num_images != -1 and num_orig_images > max_num_images:
        idx = np.round(np.linspace(0, num_orig_images - 1,
                                   max_num_images)).astype(int)
    else:
        idx = np.arange(num_orig_images)

    image_filenames = list(np.array(image_paths)[idx])

    return image_filenames, num_orig_images


def save_static_cameras(cameras, image_filename_map, sensor_dict,
                        component_dict):
    frames = []
    num_skipped = 0
    for camera in cameras.iter("camera"):
        frame = {}
        camera_label = camera.get("label").split("/")[1].split(".")[0]
        assert isinstance(camera_label, str)

        if camera_label not in image_filename_map:
            # Labels sometimes have a file extension. Try without the extension.
            # (maybe it's just a '.' in the image name)
            camera_label = camera_label.split(".")[0]  # type: ignore

            if camera_label not in image_filename_map:
                continue

        frame["file_path"] = image_filename_map[camera_label].as_posix()

        sensor_id = camera.get("sensor_id")
        if sensor_id not in sensor_dict:
            # this should only happen when we have a sensor that doesn't have calibration

            num_skipped += 1
            continue
        # Add all sensor parameters to this frame.
        frame.update(sensor_dict[sensor_id])

        if camera.find("transform") is None:

            num_skipped += 1
            continue
        transform = np.array(
            [float(x) for x in camera.find("transform").text.split()]).reshape(
                (4, 4))  # type: ignore

        component_id = camera.get("component_id")
        if component_id in component_dict:
            transform = component_dict[component_id] @ transform

        # Metashape camera is looking towards -Z, +X is to the right and +Y is to the top/up of the first cam
        # Rotate the scene according to nerfstudio convention
        transform = transform[[2, 0, 1, 3], :]
        # Convert from Metashape's camera coordinate system (OpenCV) to ours (OpenGL)
        transform[:, 1:3] *= -1
        frame["transform_matrix"] = transform.tolist()
        frames.append(frame)
    return frames

    # oriented_poses, translations = auto_orient_and_center_poses(
    #     np.array(transformed_matrices), center_method="poses", method="pca")

    # for index, frame in enumerate(frames):
    #     frames[index]["transform_matrix"] = oriented_poses[index].tolist()


def save_dynamics_cameras(cameras, image_filename_map, sensor_dict,
                          component_dict):
    frames = []
    num_skipped = 0

    gs_camera_transform = {}

    for camera in cameras.iter("camera"):

        if camera_label not in image_filename_map:
            # Labels sometimes have a file extension. Try without the extension.
            # (maybe it's just a '.' in the image name)
            camera_label = camera_label.split(".")[0]  # type: ignore

            if camera_label not in image_filename_map:
                continue

        if camera.find("transform") is None:
            num_skipped += 1
            continue
        transform = np.array(
            [float(x) for x in camera.find("transform").text.split()]).reshape(
                (4, 4))  # type: ignore
        camera_label = camera.get("label").split("/")[1].split(".")[0]
        gs_camera_transform["gs_" + camera_label.split("_")[1]] = transform

    for camera in cameras.iter("camera"):
        frame = {}
        camera_label = camera.get("label").split("/")[1].split(".")[0]
        assert isinstance(camera_label, str)

        if camera_label not in image_filename_map:
            # Labels sometimes have a file extension. Try without the extension.
            # (maybe it's just a '.' in the image name)
            camera_label = camera_label.split(".")[0]  # type: ignore

            if camera_label not in image_filename_map:
                continue

        frame["file_path"] = image_filename_map[camera_label].as_posix()

        sensor_id = camera.get("sensor_id")
        if sensor_id not in sensor_dict:
            # this should only happen when we have a sensor that doesn't have calibration

            num_skipped += 1
            continue
        # Add all sensor parameters to this frame.
        frame.update(sensor_dict[sensor_id])

        if camera.find("transform") is None:

            num_skipped += 1
            continue
        transform = np.array(
            [float(x) for x in camera.find("transform").text.split()]).reshape(
                (4, 4))  # type: ignore

        component_id = camera.get("component_id")
        if component_id in component_dict:
            transform = component_dict[component_id] @ transform

        # Metashape camera is looking towards -Z, +X is to the right and +Y is to the top/up of the first cam
        # Rotate the scene according to nerfstudio convention
        transform = transform[[2, 0, 1, 3], :]
        # Convert from Metashape's camera coordinate system (OpenCV) to ours (OpenGL)
        transform[:, 1:3] *= -1
        frame["transform_matrix"] = transform.tolist()
        frames.append(frame)
    return frames


def convert_metashape_data(data, xml, output_dir, max_dataset_size=300):
    data = Path(data)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Copy images to output directory
    image_list = []
    for gs_name in ["static_gs", "dynamics_gs"]:

        path = str(data) + f"/{gs_name}"
        if not os.path.exists(path):
            continue
        image_path = path + "/seg_imgs"

        if gs_name == "static_gs":
            image_filenames, num_orig_images = get_image_filenames(
                Path(image_path), max_dataset_size)
            image_list += image_filenames

        if gs_name == "dynamics_gs":

            for dd in os.listdir(image_path):
                image_filenames, num_orig_images = get_image_filenames(
                    Path(image_path + f"/{dd}"), max_dataset_size)
                image_list += image_filenames

    copied_image_paths = copy_images_list(
        image_list,
        image_dir=image_dir,
        verbose=True,
        num_downscales=1,
    )

    copied_image_paths = [
        Path("images/" + copied_image_path.name)
        for copied_image_path in copied_image_paths
    ]

    original_names = [image_path.stem for image_path in copied_image_paths]
    image_filename_map = dict(zip(original_names, copied_image_paths))

    metashape_to_json(
        image_filename_map=image_filename_map,
        xml_filename=xml,
        output_dir=output_dir,
        ply_filename=None,
        verbose=True,
    )


def metashape_to_json(
    image_filename_map: Dict[str, Path],
    xml_filename: Path,
    output_dir: Path,
    ply_filename: Optional[Path] = None,  # type: ignore
    verbose: bool = False,
) -> List[str]:
    """Convert Metashape data into a nerfstudio dataset.

    Args:
        image_filename_map: Mapping of original image filenames to their saved locations.
        xml_filename: Path to the metashape cameras xml file.
        output_dir: Path to the output directory.
        ply_filename: Path to the exported ply file.
        verbose: Whether to print verbose output.

    Returns:
        Summary of the conversion.
    """

    xml_tree = ET.parse(xml_filename)
    root = xml_tree.getroot()
    chunk = root
    sensors = chunk.find("sensors")

    if sensors is None:
        raise ValueError("No sensors found")

    calibrated_sensors = [
        sensor for sensor in sensors.iter("sensor")
        if sensor.get("type") == "spherical" or sensor.find("calibration")
    ]
    if not calibrated_sensors:
        raise ValueError("No calibrated sensor found in Metashape XML")
    sensor_type = [s.get("type") for s in calibrated_sensors]
    if sensor_type.count(sensor_type[0]) != len(sensor_type):
        raise ValueError(
            "All Metashape sensors do not have the same sensor type. "
            "nerfstudio does not support per-frame camera_model types."
            "Only one camera type can be used: frame, fisheye or spherical (perspective, fisheye or equirectangular)"
        )

    data = {}
    if sensor_type[0] == "frame":
        data["camera_model"] = CAMERA_MODELS["perspective"].value
    elif sensor_type[0] == "fisheye":
        data["camera_model"] = CAMERA_MODELS["fisheye"].value
    elif sensor_type[0] == "spherical":
        data["camera_model"] = CAMERA_MODELS["equirectangular"].value
    else:
        # Cylindrical and RPC sensor types are not supported
        raise ValueError(
            f"Unsupported Metashape sensor type '{sensor_type[0]}'")

    sensor_dict = {}
    for sensor in calibrated_sensors:
        s = {}
        resolution = sensor.find("resolution")
        assert resolution is not None, "Resolution not found in Metashape xml"
        s["w"] = int(resolution.get("width"))  # type: ignore
        s["h"] = int(resolution.get("height"))  # type: ignore

        calib = sensor.find("calibration")
        if calib is None:
            assert sensor_type[
                0] == "spherical", "Only spherical sensors should have no intrinsics"
            s["fl_x"] = s["w"] / 2.0
            s["fl_y"] = s["h"]
            s["cx"] = s["w"] / 2.0
            s["cy"] = s["h"] / 2.0
        else:
            f = calib.find("f")
            assert f is not None, "Focal length not found in Metashape xml"
            s["fl_x"] = s["fl_y"] = float(f.text)  # type: ignore
            s["cx"] = _find_param(calib, "cx") + s["w"] / 2.0  # type: ignore
            s["cy"] = _find_param(calib, "cy") + s["h"] / 2.0  # type: ignore

            s["k1"] = _find_param(calib, "k1")
            s["k2"] = _find_param(calib, "k2")
            s["k3"] = _find_param(calib, "k3")
            s["k4"] = _find_param(calib, "k4")
            s["p1"] = _find_param(calib, "p1")
            s["p2"] = _find_param(calib, "p2")

        sensor_dict[sensor.get("id")] = s

    components = chunk.find("components")
    component_dict = {}
    if components is not None:
        for component in components.iter("component"):
            transform = component.find("transform")
            if transform is not None:
                rotation = transform.find("rotation")
                if rotation is None:
                    r = np.eye(3)
                else:
                    assert isinstance(rotation.text, str)
                    r = np.array([float(x)
                                  for x in rotation.text.split()]).reshape(
                                      (3, 3))
                translation = transform.find("translation")
                if translation is None:
                    t = np.zeros(3)
                else:
                    assert isinstance(translation.text, str)
                    t = np.array([float(x) for x in translation.text.split()])
                scale = transform.find("scale")
                if scale is None:
                    s = 1.0
                else:
                    assert isinstance(scale.text, str)
                    s = float(scale.text)

                m = np.eye(4)
                m[:3, :3] = r
                m[:3, 3] = t / s
                component_dict[component.get("id")] = m

    frames = []
    cameras = chunk.find("cameras")
    assert cameras is not None, "Cameras not found in Metashape xml"
    num_skipped = 0

    frames = save_static_cameras(cameras, image_filename_map, sensor_dict,
                                 component_dict)
    filtered_data = {
        key: value
        for key, value in image_filename_map.items() if 'gs' in key
    }

    if filtered_data:
        frames = save_dynamics_cameras(cameras, filtered_data, sensor_dict,
                                       component_dict)

    data["frames"] = frames
    applied_transform = np.eye(4)[:3, :]

    applied_transform = applied_transform[np.array([2, 0, 1]), :]

    data["applied_transform"] = applied_transform.tolist()

    summary = []

    if ply_filename is not None:
        assert ply_filename.exists()
        pc = o3d.io.read_point_cloud(str(ply_filename))
        points3D = np.asarray(pc.points)
        points3D = np.einsum("ij,bj->bi", applied_transform[:3, :3],
                             points3D) + applied_transform[:3, 3]
        pc.points = o3d.utility.Vector3dVector(points3D)
        o3d.io.write_point_cloud(str(output_dir / "sparse_pc.ply"), pc)
        data["ply_file_path"] = "sparse_pc.ply"
        summary.append(f"Imported {ply_filename} as starting points")

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    if num_skipped == 1:
        summary.append(
            f"{num_skipped} image skipped because it was missing its camera pose."
        )
    if num_skipped > 1:
        summary.append(
            f"{num_skipped} images were skipped because they were missing camera poses."
        )

    summary.append(f"Final dataset is {len(data['frames'])} frames.")

    return summary


def focus_of_attention(poses: np.ndarray,
                       initial_focus: np.ndarray) -> np.ndarray:
    """
    Compute the focus of attention of a set of cameras. Only cameras
    that have the focus of attention in front of them are considered.

    Args:
        poses: The poses to orient, shape (*num_poses, 4, 4).
        initial_focus: The 3D point views to decide which cameras are initially activated, shape (3,).

    Returns:
        The 3D position of the focus of attention, shape (3,).
    """
    active_directions = -poses[:, :3, 2:3]
    active_origins = poses[:, :3, 3:4]
    focus_pt = initial_focus

    active = np.sum(np.squeeze(active_directions, axis=-1) *
                    (focus_pt - np.squeeze(active_origins, axis=-1)),
                    axis=-1) > 0
    done = False

    while np.sum(active.astype(int)) > 1 and not done:
        active_directions = active_directions[active]
        active_origins = active_origins[active]

        m = np.eye(3) - active_directions * np.transpose(
            active_directions, (0, 2, 1))
        mt_m = np.transpose(m, (0, 2, 1)) @ m
        focus_pt = np.linalg.inv(
            mt_m.mean(0)) @ (mt_m @ active_origins).mean(0)[:, 0]

        active = np.sum(np.squeeze(active_directions, axis=-1) *
                        (focus_pt - np.squeeze(active_origins, axis=-1)),
                        axis=-1) > 0
        if np.all(active):
            done = True

    return focus_pt


def rotation_matrix_between(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v = np.cross(v1, v2)
    c = np.dot(v1, v2)
    s = np.linalg.norm(v)

    if s == 0:
        return np.eye(3) if c > 0 else np.eye(3) * -1

    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s**2))


def auto_orient_and_center_poses(poses, center_method="poses", method="pca"):
    """Orients and centers the poses.

    We provide three methods for orientation:

    - pca: Orient the poses so that the principal directions of the camera centers are aligned
        with the axes, Z corresponding to the smallest principal component.
        This method works well when all of the cameras are in the same plane, for example when
        images are taken using a mobile robot.
    - up: Orient the poses so that the average up vector is aligned with the z axis.
        This method works well when images are not at arbitrary angles.
    - vertical: Orient the poses so that the Z 3D direction projects close to the
        y axis in images. This method works better if cameras are not all
        looking in the same 3D direction, which may happen in camera arrays or in LLFF.

    There are two centering methods:

    - poses: The poses are centered around the origin.
    - focus: The origin is set to the focus of attention of all cameras (the
        closest point to cameras optical axes). Recommended for inward-looking
        camera configurations.

    Args:
        poses: The poses to orient.
        method: The method to use for orientation.
        center_method: The method to use to center the poses.

    Returns:
        Tuple of the oriented poses and the transform matrix.
    """

    origins = poses[..., :3, 3]

    mean_origin = np.mean(origins, axis=0)
    translation_diff = origins - mean_origin

    if center_method == "poses":
        translation = mean_origin
    elif center_method == "focus":
        translation = focus_of_attention(poses, mean_origin)
    elif center_method == "none":
        translation = np.zeros_like(mean_origin)
    else:
        raise ValueError(f"Unknown value for center_method: {center_method}")

    if method == "pca":
        _, eigvec = np.linalg.eigh(translation_diff.T @ translation_diff)
        eigvec = np.flip(eigvec, axis=-1)

        if np.linalg.det(eigvec) < 0:
            eigvec[:, 2] = -eigvec[:, 2]

        transform = np.hstack([eigvec, eigvec @ -translation[..., None]])
        oriented_poses = transform @ poses

        if np.mean(oriented_poses, axis=0)[2, 1] < 0:
            oriented_poses[:, 1:3] = -1 * oriented_poses[:, 1:3]
    elif method in ("up", "vertical"):
        up = np.mean(poses[:, :3, 1], axis=0)
        up = up / np.linalg.norm(up)
        if method == "vertical":
            x_axis_matrix = poses[:, :3, 0]
            _, S, Vh = np.linalg.svd(x_axis_matrix, full_matrices=False)

            if S[1] > 0.17 * np.sqrt(poses.shape[0]):
                up_vertical = Vh[2, :]
                up = up_vertical if np.dot(up_vertical,
                                           up) > 0 else -up_vertical
            else:
                up = up - Vh[0, :] * np.dot(up, Vh[0, :])
                up = up / np.linalg.norm(up)

        rotation = rotation_matrix_between(up, np.array([0, 0, 1]))
        transform = np.hstack([rotation, rotation @ -translation[..., None]])
        oriented_poses = transform @ poses
    elif method == "none":
        transform = np.eye(4)
        transform[:3, 3] = -translation
        transform = transform[:3, :]
        oriented_poses = transform @ poses
    else:
        raise ValueError(f"Unknown value for method: {method}")

    return oriented_poses, transform
