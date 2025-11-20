import torch

from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.sim as sim_utils

import math

import isaaclab.utils.math as math_utils


def reset_camera_pose(env,
                      env_ids: torch.Tensor,
                      random_pose_range: tuple,
                      camera_name: str,
                      theta_range_rad=(-50, 50),
                      phi_range_rad=(30, 60)):
    random_pose_range = torch.as_tensor(random_pose_range).to(env.device)

    bbox = random_pose_range[..., :6].reshape(2, 3)
    bbox_min = bbox[0]
    bbox_max = bbox[1]
    radius = torch.rand((env.num_envs), device=bbox.device) * (
        random_pose_range[-1] - random_pose_range[-2]) + random_pose_range[-2]

    look_at_location = torch.rand(
        (env.num_envs, 3),
        device=bbox.device) * (bbox_max - bbox_min) + bbox_min

    eye_location = sample_spherical_point(origin=look_at_location,
                                          theta_range_rad=theta_range_rad,
                                          phi_range_rad=phi_range_rad,
                                          radius=radius,
                                          num_envs=env.num_envs,
                                          device=env.device)

    eye_location += env.scene.env_origins
    look_at_location += env.scene.env_origins

    env.scene[camera_name].set_world_poses_from_view(eye_location,
                                                     look_at_location,
                                                     env_ids=env_ids)


def reset_camera_intrisic(env, env_ids: torch.Tensor, camera_name: str,
                          focal_length_range: list):
    focal_length = torch.rand((env.num_envs), device=env_ids.device) * (
        focal_length_range[0] - focal_length_range[1]) + focal_length_range[0]
    for i in env_ids:

        # Get corresponding sensor prim
        sensor_prim = env.scene[camera_name]._sensor_prims[i]
        # get camera parameters
        sensor_prim.GetFocalLengthAttr().Set(focal_length[i].item())


def sample_spherical_point(
        origin,
        radius=1.8,
        theta_range_rad=(-0.872, 0.872),  # ≈ (-50°, 50°)
        phi_range_rad=(0.524, 1.047),  # ≈ (30°, 60°)
        num_envs=1,
        device='cuda:0'):
    # Sample theta (azimuth) and phi (elevation)
    theta = torch.empty(num_envs, device=device).uniform_(*theta_range_rad)
    phi = torch.empty(num_envs, device=device).uniform_(*phi_range_rad)

    # Convert spherical to Cartesian coordinates
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)

    offset = torch.stack([x, y, z], dim=1)

    # Expand origin if needed
    if origin.ndim == 1:
        origin = origin.unsqueeze(0).expand(num_envs, -1)
    elif origin.shape[0] != num_envs:
        raise ValueError(
            f"origin must be shape (3,) or (num_envs, 3), got {origin.shape}")

    return origin + offset


def reset_real_camera_pose(env, env_ids, camera_name, cam_trans, cam_quat,
                           random_pose_range, focal_length):

    cam_translation = cam_trans.to(env.device).unsqueeze(0).expand(
        env.num_envs, -1)  # (N, 3)
    cam_orientation = cam_quat.to(env.device).unsqueeze(0).expand(
        env.num_envs, -1)  # (N, 4)

    delta_translation = torch.rand(
        (env.num_envs, 3),
        device=env.device) * random_pose_range[0] * 2 - random_pose_range[0]
    delta_euler = torch.rand(
        (env.num_envs, 3),
        device=env.device) * random_pose_range[1] * 2 - random_pose_range[1]
    delta_quat = math_utils.quat_from_euler_xyz(delta_euler[:, 0],
                                                delta_euler[:, 1],
                                                delta_euler[:, 2])
    env.scene[camera_name].set_world_poses(
        cam_translation + delta_translation + env.scene.env_origins,
        math_utils.quat_mul(delta_quat, cam_orientation),
        env_ids=env_ids,
        convention="ros")
    # reset_focal_length = (torch.rand(
    #     (env.num_envs), device=env.device) * 2 - 1) * 3 + focal_length
    # for i in env_ids:

    #     # Get corresponding sensor prim
    #     sensor_prim = env.scene[camera_name]._sensor_prims[i]
    #     # get camera parameters
    #     sensor_prim.GetFocalLengthAttr().Set(reset_focal_length[i].item())


def configure_camera(object, env_cfg):

    if not env_cfg["params"]["Camera"]["random_pose"]:
        return
    random_pose_range = torch.as_tensor(
        env_cfg["params"]["Camera"]["random_pose_range"])
    camera_name_list = list(env_cfg["params"]["Camera"]["cameras"].keys())

    if env_cfg["params"]["Camera"].get("camera_json", None) is None:
        for camera_name in camera_name_list:
            camera_pose_reset = EventTerm(
                func=reset_camera_pose,
                mode="reset",
                params={
                    "camera_name":
                    camera_name,
                    "random_pose_range":
                    random_pose_range,
                    "theta_range_rad":
                    env_cfg["params"]["Camera"]["cameras"][camera_name]
                    ["theta_range_rad"],
                    "phi_range_rad":
                    env_cfg["params"]["Camera"]["phi_range_rad"],
                })
            setattr(object.events, f"{camera_name}_pose_reset",
                    camera_pose_reset)

            # reset_camera_intrisic_func = EventTerm(
            #     func=reset_camera_intrisic,
            #     mode="reset",
            #     params={
            #         "camera_name":
            #         camera_name,
            #         "focal_length_range":
            #         env_cfg["params"]["Camera"]["focal_length_range"],
            #     })
            # setattr(object.events, f"{camera_name}_intrisic_reset",
            #         reset_camera_intrisic_func)
    else:
        for camera_name in camera_name_list:
            camera_pose_reset = EventTerm(
                func=reset_real_camera_pose,
                mode="reset",
                params={
                    "camera_name":
                    camera_name,
                    "cam_trans":
                    torch.as_tensor(env_cfg["params"]["Camera"]["cameras"]
                                    [camera_name]["cam_trans"]),
                    "cam_quat":
                    torch.as_tensor(env_cfg["params"]["Camera"]["cameras"]
                                    [camera_name]["cam_quat"]),
                    "random_pose_range":
                    env_cfg["params"]["Camera"]["camera_json_random_pose"],
                    "focal_length":
                    env_cfg["params"]["Camera"]["cameras"][camera_name]
                    ["focal_length"]
                })
            setattr(object.events, f"{camera_name}_pose_reset",
                    camera_pose_reset)
