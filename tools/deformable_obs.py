import torch

from isaaclab.sensors.camera.batch_utils import create_pointcloud_from_rgbd_batch
from typing import TYPE_CHECKING

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

import isaaclab.utils.math as math_utils


def reset_deformable_root_state_uniform(
        env,
        env_ids: torch.Tensor,
        velocity_range: dict[str, tuple[float, float]],
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation | DeformableObject = env.scene[
        asset_cfg.name]
    # get default root state

    root_states = asset.data.default_root_state[env_ids].clone()
    pose_range = asset.cfg.deform_cfg["pose_range"]
    if pose_range is None:
        pose_range = {}

    # poses
    range_list = [
        pose_range.get(key, (0.0, 0.0))
        for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]

    ranges = torch.tensor(range_list, device='cuda:0')
    rand_samples = math_utils.sample_uniform(ranges[:, 0],
                                             ranges[:, 1], (len(env_ids), 6),
                                             device=asset.device)

    # positions = root_states[:, 0:3] + env.scene.env_origins[
    #     env_ids] + rand_samples[:, 0:3]

    positions = rand_samples[:, 0:3]

    orientations_delta = math_utils.quat_from_euler_xyz(
        rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
    # orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
    orientations = orientations_delta

    # velocities = root_states[:, 7:13] + rand_samples
    root_pos = torch.cat([positions, orientations], dim=-1)

    raw_nodal_w = asset.data.default_nodal_state_w[:, :asset.
                                                   max_simulation_mesh_vertices_per_body, :
                                                   3].clone(
                                                   ) - env.scene.env_origins[:, None, :].repeat_interleave(
                                                       asset.
                                                       max_simulation_mesh_vertices_per_body,
                                                       1)

    inital_delta = asset._data.default_root_state[:, :3][:, None].repeat_interleave(
        asset.max_simulation_mesh_vertices_per_body, 1)

    raw_nodal_w -= inital_delta

    root_pos = math_utils.transform_points(
        raw_nodal_w, root_pos[:, :3],
        root_pos[:, 3:]) + env.scene.env_origins[:, None, :].repeat_interleave(
            asset.max_simulation_mesh_vertices_per_body, 1)

    # set into the physics simulation\
    nodal_w = torch.cat([root_pos, root_pos * 0], dim=1)
    nodal_w[:, :asset.
            max_simulation_mesh_vertices_per_body, :3] += inital_delta
    asset._data.reset_nodal_state_w = nodal_w

    asset.write_root_state_to_sim(nodal_w, env_ids=env_ids)


def object_physical_params(env: ManagerBasedRLEnv, ) -> torch.Tensor:
    physical_params = []
    for name in env.scene.keys():
        if "deform" in name:
            deformable_object = env.scene[name]
            physical_params.append(deformable_object.data.physical_params)

    physical_params = torch.stack(physical_params)
    num_object = physical_params.shape[0]
    num_env = physical_params.shape[1]
    physical_params = physical_params.view(-1, physical_params.shape[2])

    pc_idx_per_env = (
        torch.arange(0, num_env * num_object, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_object)).to(
            physical_params.device)[0]

    return physical_params[pc_idx_per_env].view(num_env, num_object,
                                                physical_params.shape[1])


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, ) -> torch.Tensor:
    """The position of the object in the robot's root frame."""

    for name in env.scene.keys():
        if "deform" in name:

            deformable_object = env.scene[name]

            tube_front_positions = deformable_object.data.nodal_state_w[:, 200] - torch.mean(
                deformable_object.data.
                default_nodal_state_w[:, :deformable_object.
                                      max_simulation_mesh_vertices_per_body],
                dim=1)
            tube_back_positions = deformable_object.data.nodal_state_w[:,
                                                                       int(
                                                                           deformable_object
                                                                           .max_simulation_mesh_vertices_per_body
                                                                           / 2
                                                                       )] - torch.mean(
                                                                           deformable_object
                                                                           .
                                                                           data
                                                                           .
                                                                           default_nodal_state_w[:, :
                                                                                                 deformable_object
                                                                                                 .
                                                                                                 max_simulation_mesh_vertices_per_body],
                                                                           dim=1
                                                                       )
            robot_assest = env.scene["robot"]
            body_ids, body_names = robot_assest.find_bodies("panda_hand")
            gripper_site_pos = robot_assest.data.body_state_w[:,
                                                              body_ids[0]][:, :
                                                                           3]
            front_to_gripper = tube_front_positions - gripper_site_pos

    return torch.cat(
        [tube_front_positions, tube_back_positions, front_to_gripper], dim=-1)


def object_node_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, ) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    nodals_position = []
    # default_root_states = []
    contact_ornot = []
    for name in env.scene.keys():
        if "deform" in name:

            deformable_object = env.scene[name]

            nodals_position.append(
                deformable_object.data.
                nodal_state_w[:, :deformable_object.
                              max_simulation_mesh_vertices_per_body])
            # default_root_states.append(
            #     deformable_object.data.default_root_state)
            deformable_object.data.root_pos_w

            # collision_element_stresses = deformable_object.data.collision_element_deformation_gradients
            # collision_element_stresses = collision_element_stresses.view(
            #     collision_element_stresses.shape[0],
            #     collision_element_stresses.shape[1] * 9)
            # conact_env_index = torch.where(collision_element_stresses > 1)[0]
            # contact_array = torch.zeros(len(collision_element_stresses), 1).to(
            #     collision_element_stresses.device)

            # contact_array[conact_env_index] = 1

            # contact_ornot.append(contact_array)

    nodals_position = torch.stack(nodals_position)
    num_object = nodals_position.shape[0]
    num_env = nodals_position.shape[1]
    nodals_position = nodals_position.view(-1, nodals_position.shape[2],
                                           nodals_position.shape[3])
    # default_root_states = torch.stack(default_root_states)
    # default_root_states = default_root_states.view(
    #     -1, default_root_states.shape[2])
    nodals_position[:, :, :
                    2] -= env.scene.env_origins[:, :
                                                2][:,
                                                   None, :].repeat_interleave(
                                                       nodals_position.
                                                       shape[1], 1)

    pc_idx_per_env = (
        torch.arange(0, num_env * num_object, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_object)).to(
            nodals_position.device)[0]

    return nodals_position[pc_idx_per_env].view(num_env, num_object,
                                                nodals_position.shape[1], 3)


def deformable_pose(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("deform_object"),
) -> torch.Tensor:
    deform_objects = [*env.scene._deformable_objects.keys()]
    deformable_object_state = []
    for deformable_object_name in deform_objects:
        deformable_object = env.scene[deformable_object_name]
        deform_pos_w = deformable_object.root_physx_view.get_sim_nodal_positions(
        )
        deform_pos_w_mean = deform_pos_w.mean(dim=1)
        deform_pos_w_mean -= env.scene.env_origins
        deform_pos_w_mean_max = deform_pos_w.max(dim=1).values
        deform_pos_w_mean_max -= env.scene.env_origins
        deformable_object_state.append(
            torch.cat([deform_pos_w_mean, deform_pos_w_mean_max], dim=-1))

    return torch.cat(deformable_object_state, dim=0)


def object_physical_params(env: ManagerBasedRLEnv, ) -> torch.Tensor:
    physical_params = []
    deform_objects = [*env.scene._deformable_objects.keys()]
    if env.scene[deform_objects[0]].parames_generator is None:
        return {}

    for name in deform_objects:
        deformable_object = env.scene[name]
        physical_params.append(deformable_object.data.physical_params)

    physical_params = torch.stack(physical_params)
    num_object = physical_params.shape[0]
    num_env = physical_params.shape[1]

    physical_params = physical_params.view(-1, physical_params.shape[2])

    pc_idx_per_env = (
        torch.arange(0, num_env * num_object, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_object)).to(
            physical_params.device)[0]

    return physical_params[pc_idx_per_env].view(num_env, num_object,
                                                physical_params.shape[1])[:, 0]


def object_node_position_in_robot_root_frame(
    env: ManagerBasedRLEnv, ) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    nodals_position = []
    default_root_states = []
    contact_ornot = []
    for name in env.scene.keys():
        if "deform" in name:

            deformable_object = env.scene[name]

            nodals_position.append(
                deformable_object.data.
                nodal_state_w[:, :deformable_object.
                              max_simulation_mesh_vertices_per_body]
                [:, ::deformable_object.cfg.
                 deform_cfg["sample_node_interval"]])
            default_root_states.append(
                deformable_object.data.default_root_state)
            deformable_object.data.root_pos_w

            # collision_element_stresses = deformable_object.data.collision_element_deformation_gradients
            # collision_element_stresses = collision_element_stresses.view(
            #     collision_element_stresses.shape[0],
            #     collision_element_stresses.shape[1] * 9)
            # conact_env_index = torch.where(collision_element_stresses > 1)[0]
            # contact_array = torch.zeros(len(collision_element_stresses), 1).to(
            #     collision_element_stresses.device)

            # contact_array[conact_env_index] = 1

            # contact_ornot.append(contact_array)

    nodals_position = torch.stack(nodals_position)
    num_object = nodals_position.shape[0]
    num_env = nodals_position.shape[1]
    nodals_position = nodals_position.view(-1, nodals_position.shape[2],
                                           nodals_position.shape[3])
    default_root_states = torch.stack(default_root_states)
    default_root_states = default_root_states.view(
        -1, default_root_states.shape[2])
    nodals_position[:, :, :
                    2] -= default_root_states[:, :
                                              2][:, None, :].repeat_interleave(
                                                  nodals_position.shape[1], 1)
    # import pdb

    # pdb.set_trace()
    # contact_ornot = torch.stack(contact_ornot)[:, None, :].repeat_interleave(
    #     nodals_position.shape[1], 1)
    pc_idx_per_env = (
        torch.arange(0, num_env * num_object, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_object)).to(
            nodals_position.device)[0]

    return nodals_position[pc_idx_per_env].view(num_env, num_object,
                                                nodals_position.shape[1], 3)


def object_3d_seg_rgb(env: ManagerBasedRLEnv):
    from tools.caliberation_json import generate_id_map_from_rgb

    # rgb
    seg_name = env.scene["deform_object"].cfg.deform_cfg["camera_obs"][
        "segmentation_name"]
    # seg = object_3d_observation(env, seg_name)[..., :3]
    # rgb = object_3d_observation(env, "rgb")[..., :3]

    camera_list = []
    for name in env.scene.keys():
        if "camera" in name:

            camera_list.append(env.scene[name].data)

    rgb_data = []
    seg_data = []
    num_camera = 0
    for data in camera_list:
        num_camera += 1
        num_env = len(data.intrinsic_params)

        rgb_data.append(data.output["rgb"])
        seg_data.append(data.output[seg_name])

    rgb = torch.stack(rgb_data).view(-1, *rgb_data[0].shape[1:])

    seg = torch.stack(seg_data).view(-1, *seg_data[0].shape[1:])

    pc_idx_per_env = (
        torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_camera)).to(
            rgb.device)[0]

    rgb = rgb[pc_idx_per_env].view(num_env, num_camera,
                                   *rgb.shape[1:])[..., :3]
    seg = seg[pc_idx_per_env].view(num_env, num_camera, *seg.shape[1:])

    # generate_id_map_from_rgb(seg[0].cpu().numpy())
    background_mask = torch.all(seg.unsqueeze(-1) == 0, axis=-1)

    frontground_mask = ~background_mask
    front_ground = torch.zeros_like(rgb) + 255
    front_ground[frontground_mask] = rgb[frontground_mask]

    front_ground = torch.cat([front_ground, seg.unsqueeze(-1)], dim=-1)
    del rgb
    del seg
    torch.cuda.empty_cache()
    return front_ground


def object_3d_observation(
        env: ManagerBasedRLEnv,
        image_name="rgb",  #"instance_segmentation_fast"
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""

    camera_list = []
    for name in env.scene.keys():
        if "camera" in name:

            camera_list.append(env.scene[name].data)

    rgb_data = []
    num_camera = 0
    for data in camera_list:
        num_camera += 1
        num_env = len(data.intrinsic_params)

        rgb_data.append(data.output[image_name])

    rgb = torch.stack(rgb_data).view(-1, *rgb_data[0].shape[1:])

    pc_idx_per_env = (
        torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_camera)).to(
            rgb.device)[0]

    return rgb[pc_idx_per_env].view(num_env, num_camera,
                                    *rgb_data[0].shape[1:])


def obtain_camera_intrinsic(
        env: ManagerBasedRLEnv,
        image_name="rgb",  #"instance_segmentation_fast"
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""

    camera_list = []
    for name in env.scene.keys():
        if "camera" in name:
            camera_list.append(env.scene[name].data)

    intrinsic_params = []
    num_camera = 0
    for data in camera_list:
        num_camera += 1
        num_env = len(data.intrinsic_params)

        intrinsic_params.append(data.intrinsic_params)

    params = torch.stack(intrinsic_params).view(-1,
                                                intrinsic_params[0].shape[1],
                                                intrinsic_params[0].shape[2])

    pc_idx_per_env = (
        torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_camera)).to(
            params.device)[0]

    return params[pc_idx_per_env].view(num_env, num_camera, params.shape[1],
                                       params.shape[2])


def obtain_camera_extrinsic(
        env: ManagerBasedRLEnv,
        image_name="rgb",  #"instance_segmentation_fast"
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""

    camera_list = []
    for name in env.scene.keys():
        if "camera" in name:
            camera_list.append(env.scene[name].data)

    extrinsic_params = []
    num_camera = 0
    for data in camera_list:
        num_camera += 1
        num_env = len(data.intrinsic_params)

        transformation = torch.eye(4).to(
            data.intrinsic_params.device)[None].repeat_interleave(
                len(data.intrinsic_params), 0)
        transformation[:, :3, :3] = math_utils.matrix_from_quat(
            data.quat_local_opengl)
        transformation[:, :3, 3] = data.pos_local

        extrinsic_params.append(transformation)

    params = torch.stack(extrinsic_params).view(-1,
                                                extrinsic_params[0].shape[1],
                                                extrinsic_params[0].shape[2])

    pc_idx_per_env = (
        torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_camera)).to(
            params.device)[0]

    return params[pc_idx_per_env].view(num_env, num_camera, params.shape[1],
                                       params.shape[2])


def get_color_pc(env: ManagerBasedRLEnv, ) -> torch.Tensor:
    # camera: Camera = env.scene[camera_cfg.name]
    intrinsic_params = []
    depth = []
    rgb = []
    position = []
    orientation = []
    num_camera = 0
    camera_list = []

    camera_obs_cobfig = env.scene["deform_object"].cfg.deform_cfg["camera_obs"]

    if camera_obs_cobfig is not None:
        segmentation_name = camera_obs_cobfig["segmentation_name"]
        bbox = camera_obs_cobfig["bbox"]
        max_length = camera_obs_cobfig["max_length"]
    else:
        segmentation_name = None
        bbox = None
        max_length = 0

    if segmentation_name is not None:
        seg_mask = []
    for name in env.scene.keys():

        if "camera" in name:
            num_camera += 1
            camera_list.append(env.scene[name].data)

    for data in camera_list:
        position.append(data.pos_local)
        orientation.append(data.quat_local_ros)

        num_env = len(data.intrinsic_params)

        intrinsic_params.append(data.intrinsic_params)

        depth.append(data.output["distance_to_image_plane"])
        rgb.append(data.output["rgb"])
        if segmentation_name is not None:
            seg_mask.append(data.output[segmentation_name])

    if segmentation_name is None:
        seg_mask_tensor = None
    else:
        # Ensure seg_mask is a list or similar iterable
        seg_mask_tensor = torch.stack(seg_mask).view(-1, rgb[0].shape[1],
                                                     rgb[0].shape[2],
                                                     rgb[0].shape[3])
    points_xyz, points_rgb = create_pointcloud_from_rgbd_batch(
        intrinsic_matrix=torch.stack(intrinsic_params).view(-1, 3, 3),
        depth=torch.stack(depth).view(-1, depth[0].shape[1],
                                      depth[0].shape[2]),
        rgb=torch.stack(rgb).view(-1, rgb[0].shape[1], rgb[0].shape[2],
                                  rgb[0].shape[3]),
        seg_mask=seg_mask_tensor,
        position=torch.stack(position).reshape(-1, 3),
        orientation=torch.stack(orientation).reshape(-1, 4),
        bbox=bbox,
        max_length=max_length)

    pc_idx_per_env = (
        torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
        torch.arange(0, num_env).repeat_interleave(num_camera)).to(
            points_xyz.device)[0]

    # rearrange color,pc

    points_xyz_rgb = torch.cat([points_xyz, points_rgb], dim=-1)
    points_xyz_rgb = points_xyz_rgb[pc_idx_per_env]

    combined_points_xyz_rgb = points_xyz_rgb.view(num_env, -1, 6)
    del points_xyz
    del points_rgb
    del points_xyz_rgb
    del pc_idx_per_env
    torch.cuda.empty_cache()

    return combined_points_xyz_rgb


def randomize_camera_pose(env: ManagerBasedRLEnv, env_ids, time_step):
    for name in env.scene.keys():

        if "camera" in name:

            if env.scene[name].cfg.customize_camera_setting:
                env.scene[name].randonmize_camera_pose(env_ids, time_step,
                                                       env.scene.env_origins)


def shot_pc(env: ManagerBasedRLEnv) -> dict:

    result = {}

    camera_obs_config = env.scene["deform_object"].cfg.deform_cfg["camera_obs"]
    whole_rgb = camera_obs_config["whole_rgb"]
    render_pc = camera_obs_config["whole_pc"]
    if whole_rgb or render_pc:

        camera_list = [
            env.scene[name].data for name in env.scene.keys()
            if "camera" in name
        ]
        pc_data = []
        rgb_data = []
        for data in camera_list:
            pc_data.append(data.output["pointcloud"])
            if whole_rgb:
                rgb_data.append(data.output["rgb"])

        num_env = pc_data[0].shape[0]
        num_camera = len(pc_data)
        pc_idx_per_env = (
            torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
            torch.arange(0, num_env).repeat_interleave(num_camera)).to(
                env.device)[0]

        pc_data = torch.stack(pc_data).view(-1, *pc_data[0].shape[1:])

        pc_data[:, :, :3] -= env.scene.env_origins[:, None, :3]
        result["seg_pc"] = pc_data

        if whole_rgb:
            rgb_data = torch.stack(rgb_data).view(
                -1, *rgb_data[0].shape[1:])[..., :3]
            rgb_data = rgb_data[pc_idx_per_env].view(num_env, num_camera,
                                                     *rgb_data.shape[1:])
            result["rgb"] = rgb_data

        del pc_data, rgb_data

    return result


def process_camera_data(env: ManagerBasedRLEnv) -> dict:

    deform_objects = [*env.scene._deformable_objects.keys()]
    camera_obs_config = env.scene[
        deform_objects[0]].cfg.deform_cfg["camera_obs"]

    segmentation_name = camera_obs_config["segmentation_name"]
    bbox = camera_obs_config["bbox"]
    max_length = camera_obs_config["max_length"]
    whole_pc = camera_obs_config["whole_pc"]
    seg_pc = camera_obs_config["seg_pc"]
    whole_rgb = camera_obs_config["whole_rgb"]
    seg_rgb = camera_obs_config["seg_rgb"]
    if not whole_pc and not seg_pc and not whole_rgb and not seg_rgb:
        return {}

    camera_list = [
        env.scene[name].data for name in env.scene.keys() if "camera" in name
    ]

    result = {}

    rgb_data = []
    seg_data = []
    intrinsic_params = []
    depth = []
    position = []
    orientation = []

    extrinsic_params = []
    extrinsic_orientation = []

    # loop through all the cameras
    for data in camera_list:
        if whole_rgb or seg_rgb or seg_pc:
            rgb_data.append(data.output["rgb"])

        if seg_rgb or seg_pc:
            seg_data.append(data.output[segmentation_name])

        position.append(data.pos_local)
        orientation.append(data.quat_local_ros)
        extrinsic_orientation.append(data.quat_local_opengl)
        intrinsic_params.append(data.intrinsic_matrices)

        if whole_pc or seg_pc:

            if "distance_to_camera" in env.scene["camera"].cfg.data_types:

                depth.append(data.output["distance_to_camera"])
            else:
                depth.append(data.output["distance_to_image_plane"])

    ########################################

    # rgb and seg image
    if seg_rgb or seg_pc or whole_pc or whole_rgb:
        rgb = torch.stack(rgb_data).view(-1, *rgb_data[0].shape[1:])
        if seg_rgb or seg_pc:
            seg = torch.stack(seg_data).view(-1, *seg_data[0].shape[1:])

        num_env = len(camera_list[0].intrinsic_matrices)
        num_camera = len(camera_list)

        pc_idx_per_env = (
            torch.arange(0, num_env * num_camera, num_env).repeat(1, num_env) +
            torch.arange(0, num_env).repeat_interleave(num_camera)).to(
                rgb.device)[0]

        rgb = rgb[pc_idx_per_env].view(num_env, num_camera,
                                       *rgb.shape[1:])[..., :3]

        if torch.max(rgb) < 1.2:
            rgb = (rgb * 255).byte()
        if seg_rgb or seg_pc:
            seg = seg[pc_idx_per_env].view(num_env, num_camera, *seg.shape[1:])

        if seg_rgb:
            background_mask = torch.all(seg.unsqueeze(-1) == 1,
                                        axis=-1)  # bg is 1
            frontground_mask = ~background_mask
            front_ground = torch.zeros_like(rgb)

            front_ground[frontground_mask.squeeze(-1)] = rgb[
                frontground_mask.squeeze(-1)]

            result['seg_rgb'] = torch.cat([front_ground, seg], dim=-1)

            result['segmentation'] = seg

            del front_ground, background_mask, frontground_mask
        elif whole_rgb:
            result['rgb'] = rgb

    ########################################

    if whole_pc or seg_pc:
        # point cloud
        seg_mask_tensor = (seg if seg_pc else None)

        if not whole_rgb and not seg_rgb:
            rgb = torch.stack(rgb_data).view(-1, *rgb_data[0].shape[1:])
        else:
            rgb = rgb.view(-1, *rgb[0].shape[1:])

        depth_data = torch.stack(depth).view(-1, *depth[0].shape[1:])
        if len(depth_data.size()) == 4:  # for distance_to_camera

            depth_data = math_utils.convert_perspective_depth_image_to_orthogonal_depth_image(
                depth_data,
                torch.stack(intrinsic_params).view(-1, 3, 3))
        points_xyz_rgb = create_pointcloud_from_rgbd_batch(
            intrinsic_matrix=torch.stack(intrinsic_params).view(-1, 3, 3),
            depth=depth_data,
            rgb=rgb,
            seg_mask=seg_mask_tensor,
            position=torch.stack(position).reshape(-1, 3),
            orientation=torch.stack(orientation).reshape(-1, 4),
            bbox=bbox,
            max_length=max_length,
        )
        # del depth_data, rgb, seg_mask_tensor, depth, seg, rgb_data, seg_data

        num_env = len(camera_list[0].intrinsic_matrices)
        num_camera = len(camera_list)

        points_xyz_rgb = points_xyz_rgb[pc_idx_per_env]

        result['seg_pc'] = points_xyz_rgb.view(num_env, -1,
                                               points_xyz_rgb.shape[-1])
        ########################################

        del points_xyz_rgb
    if seg_rgb or seg_pc or whole_pc:
        # extrinsic params and intrinsic params
        extrinsic_position = torch.stack(position).view(
            -1, *position[0].shape[1:])
        extrinsic_orientation = torch.stack(extrinsic_orientation).view(
            -1, *orientation[0].shape[1:])
        extrinsic_orientation = math_utils.matrix_from_quat(
            extrinsic_orientation)

        extrisic_transformation = torch.eye(4).unsqueeze(0).repeat_interleave(
            len(extrinsic_position), 0).to(extrinsic_orientation.device)
        extrisic_transformation[:, :3, :3] = extrinsic_orientation
        extrisic_transformation[:, :3, 3] = extrinsic_position

        result['extrinsic_params'] = extrisic_transformation[
            pc_idx_per_env].view(num_env, num_camera,
                                 *extrisic_transformation.shape[1:3])

        intrinsic_params = torch.stack(intrinsic_params).view(
            -1, *intrinsic_params[0].shape[1:3])
        result['intrinsic_params'] = intrinsic_params[pc_idx_per_env].view(
            num_env, num_camera, *intrinsic_params.shape[1:3])

        del orientation, position, extrinsic_position, extrinsic_orientation, extrisic_transformation, intrinsic_params
    torch.cuda.empty_cache()

    return result
