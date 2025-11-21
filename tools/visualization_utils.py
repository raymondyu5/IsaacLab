# # import open3d as o3d
import cv2
import numpy as np
import sys
import os
import trimesh

sys.path.append("source/extensions")
import isaaclab.utils.math as math_utils
import torch
import matplotlib.pyplot as plt
import open3d as o3d
# from torch_cluster import fps


def apply_noise_action(raw_actions, action_noise):

    raw_actions = torch.as_tensor(raw_actions)

    num_top_noise_percent = int(1.0 * len(raw_actions))
    noise = torch.zeros_like(raw_actions[:num_top_noise_percent])
    noise[:, :3] = (torch.randn_like(noise[::3, :3]) -
                    0.5) * action_noise[0]  # Position noise
    noise[:, 3:6] = (torch.randn_like(noise[::3, 3:6]) -
                     0.5) * action_noise[1]  # Orientation noise
    euler_angles = math_utils.euler_xyz_from_quat(
        raw_actions[:num_top_noise_percent, 3:7])

    noise_euler_angle = torch.cat([
        euler_angles[0].unsqueeze(-1), euler_angles[1].unsqueeze(-1),
        euler_angles[2].unsqueeze(-1)
    ],
                                  dim=1) + noise[:, 3:6]
    noise_quat = math_utils.quat_from_euler_xyz(noise_euler_angle[:, 0],
                                                noise_euler_angle[:, 1],
                                                noise_euler_angle[:, 2])

    noise_position = raw_actions[:num_top_noise_percent, :3] + noise[:, :3]

    noise_pose = torch.cat([noise_position, noise_quat], dim=1)

    return noise_pose


robot_body_names = [
    'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4',
    'panda_link5', 'panda_link6', 'panda_link7', 'panda_hand',
    'panda_leftfinger', 'panda_rightfinger'
]


def load_robot_mesh(log_dir):
    mesh_dict = {}
    for body_id, body_name in enumerate(robot_body_names):
        mesh = trimesh.load(f"logs/mesh/{body_name}.obj")

        vertices = torch.as_tensor(mesh.vertices, dtype=torch.float32)
        mesh_dict[body_name] = vertices
    return mesh_dict


def aug_robot_mesh(
    robot_link_pose,
    mesh_dir,
    sample_robot_poins=400,
):
    robot_links = []

    for body_id, body_name in enumerate(robot_body_names):

        vertices = mesh_dir[body_name].unsqueeze(0).repeat_interleave(
            robot_link_pose.shape[0], 0)
        link_transform = robot_link_pose[:, body_id, :7]
        transformed_link = math_utils.transform_points(vertices,
                                                       link_transform[:, :3],
                                                       link_transform[:, 3:7])
        robot_links.append(transformed_link)

    robot_links = torch.cat(robot_links, dim=1)
    imagin_robot = sample_fps(robot_links, sample_robot_poins)

    return imagin_robot


def sample_fps(data, num_samples=1024):
    point_clouds = torch.as_tensor(data, dtype=torch.float32).to("cuda:0")
    batch_size, num_points, num_dims = point_clouds.shape
    flattened_points = point_clouds.view(-1, num_dims)  # Shape: (71*10000, 3)

    # Create a batch tensor to indicate each point cloud's batch index
    batch = torch.repeat_interleave(torch.arange(batch_size),
                                    num_points).to(flattened_points.device)
    ratio = num_samples / num_points

    # Apply farthest point sampling
    sampled_idx = fps(point_clouds[:, :, :3].reshape(-1, 3),
                      batch,
                      ratio=ratio,
                      batch_size=batch_size)
    sampled_points = flattened_points[sampled_idx]
    sampled_points_per_cloud = sampled_points.size(0) // batch_size
    output = sampled_points.view(batch_size, sampled_points_per_cloud,
                                 num_dims)
    return output.cpu()  #.numpy()


def segmentation_to_rgb(segmentation, num_classes):
    """
    Convert segmentation map to RGB image using a colormap.

    Parameters:
    segmentation (np.ndarray): The segmentation map (H x W), with values 0, 1, 2, ... representing classes.
    num_classes (int): Number of unique classes in the segmentation map.

    Returns:
    np.ndarray: RGB image (H x W x 3).
    """
    # Create a colormap with `num_classes` colors
    colormap = plt.get_cmap('viridis', num_classes)

    # Normalize the segmentation map to match colormap scale
    normalized_seg = segmentation / (num_classes - 1)

    # Apply the colormap to the normalized segmentation map
    rgb_image = colormap(normalized_seg)[:, :, :3]  # Remove the alpha channel

    # Convert to 8-bit values (optional if you need an image format with pixel values between 0-255)
    rgb_image = (rgb_image * 255).astype(np.uint8)

    return rgb_image


def save_classifier_video(traj_rgb, params_smpls_classifer, save_dir):
    for i, (params, video) in enumerate(zip(params_smpls_classifer, traj_rgb)):
        # Extract the classifier label and other parameters
        classifier_label = int(params[0])
        other_params = params[int((len(params) - 1) / 2):]

        # Create the folder based on the classifier label
        label_dir = os.path.join(save_dir, f"class_{classifier_label}")
        os.makedirs(label_dir, exist_ok=True)

        # Create a name based on the other parameters
        video_name = "_".join(map(str, other_params))
        video_path = os.path.join(label_dir, f"video_{video_name}.mp4")

        batch_images_to_video(
            video,
            video_path,
            20,
            image_save_folder=f"logs/image/class_{label_dir}/{video_name}")


def batch_images_to_video(batch_images,
                          output_video_path,
                          frame_rate,
                          image_save_folder=None):
    """
    Converts a batch of images to a video. If images are batched [b, n, h, w, 3], they are concatenated before saving.
    
    :param batch_images: Input images of shape [b, n, h, w, 3]
    :param output_video_path: Path where the output video will be saved
    :param frame_rate: Frame rate for the video
    """
    # Check if input is a batch of images

    if len(batch_images.shape) == 5:
        b, n, h, w, c = batch_images.shape
        images = []

        # Concatenate images in each batch along the width (axis=2)
        for i in range(b):

            concatenated_images = np.concatenate(batch_images[i], axis=1)
            images.extend(concatenated_images
                          )  # Add all images in this batch to the list

    else:
        images = batch_images  # If not a batch, assume images are already in the correct format [n, h, w, 3]

    # Define the codec and create VideoWriter object
    height, width, layers = images[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID' for .avi
    video = cv2.VideoWriter(output_video_path, fourcc, frame_rate,
                            (width, height))

    if image_save_folder is not None:
        os.makedirs(image_save_folder, exist_ok=True)
        for idx, img in enumerate(images):
            image_path = os.path.join(image_save_folder,
                                      f"frame_{idx:04d}.png")
            cv2.imwrite(image_path, img)
    # Iterate through the images and write them to the video file
    for img in images:

        video.write(cv2.resize(img.astype(np.uint8)[..., ::-1], (320, 320)))

    # Release the VideoWriter object
    video.release()


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


def transform_vertices(vertices, quat, translation):
    rotation_matrix = math_utils.matrix_from_quat(quat)

    transformed_vertices = (rotation_matrix @ vertices.T).T + translation

    return transformed_vertices


def o3d_fem_mesh(mesh_path, translate):
    mesh = trimesh.load(mesh_path)
    triangles = mesh.faces
    np_points = mesh.vertices - translate
    np_triangles = np.array(triangles)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np_points)
    mesh.triangles = o3d.utility.Vector3iVector(np_triangles)
    return mesh


def write_fem_to_obj(path, filename, points, indices, fem=False):
    """
    Write FEM data to an OBJ file.

    :param filename: The output filename
    :param points: A (k, 3) array of vertex coordinates
    :param indices: A (N, 4) array of element indices
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    with open(path + "/" + filename, 'w') as f:
        # Write vertices
        for point in points:
            f.write(f"v {point[0]} {point[1]} {point[2]}\n")

        # Write elements (assuming tetrahedra)
        for index in indices:
            # OBJ format uses 1-based indexing
            if not fem:
                f.write(f"f {index[0]+1} {index[1]+1} {index[2]+1}\n")
            else:
                f.write(
                    f"f {index[0]+1} {index[1]+1} {index[2]+1} {index[3]+1}\n")


def visualize_pcd(pc_list,
                  coordinate_original=[0, 0.0, -0.0],
                  video_writer=None,
                  render=False,
                  visible=True,
                  translation=(0, 0, 0),
                  rotation_axis=[0],
                  rotation_angles=[0],
                  widnow_size=(480, 480)):
    import open3d as o3d
    import numpy as np
    import cv2

    if video_writer is not None or render:
        visible = False

    # Create coordinate frame (will be transformed together with point clouds)
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1, origin=coordinate_original)

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='Press q or Esc to quit',
                      visible=visible,
                      width=widnow_size[0],
                      height=widnow_size[1])

    # Compute rotation matrix from quaternion
    R = math_utils.matrix_from_quat(
        obtain_target_quat_from_multi_angles(rotation_axis,
                                             rotation_angles)).numpy()

    # --- Apply same transform to coordinate frame ---
    coordinate_frame.rotate(R, center=(0, 0, 0))
    coordinate_frame.translate(translation)
    vis.add_geometry(coordinate_frame)

    # --- Apply same transform to all point clouds ---
    for pc in pc_list:
        pc_copy = pc
        pc_copy.rotate(R, center=(0, 0, 0))
        pc_copy.translate(translation)
        vis.add_geometry(pc_copy)

    # Load render settings and viewpoint (optional, for better visualization)
    render_option = vis.get_render_option()
    if render_option is not None:
        render_option_path = 'tools/render_option.json'
        if os.path.exists(render_option_path):
            render_option.load_from_json(render_option_path)
    
    view_point_path = 'tools/view_point.json'
    if os.path.exists(view_point_path):
        param = o3d.io.read_pinhole_camera_parameters(view_point_path)
        view_control = vis.get_view_control()
        if view_control is not None:
            view_control.convert_from_pinhole_camera_parameters(
                param, allow_arbitrary=True)

    # Render or display
    if video_writer is not None or render:
        image_float_buffer = vis.capture_screen_float_buffer(do_render=True)
        image = (np.asarray(image_float_buffer) * 255).astype(
            np.uint8)[:, :, [2, 1, 0]]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        vis.destroy_window()
        if video_writer is not None:
            video_writer.append_data(image)
        else:
            return image
    else:
        vis.run()
        vis.destroy_window()


def vis_pc(pc,
           color=None,
           crop_min=[-5.0, -5.0, 0.01],
           crop_max=[5.0, 5.0, 1.0]):

    # creae camera buffer
    camera_o3d = o3d.geometry.PointCloud()

    camera_o3d.points = o3d.utility.Vector3dVector(pc)

    if color is not None:

        # pc, color = crop_points(pc, color, crop_min, crop_max)
        camera_o3d.colors = o3d.utility.Vector3dVector(color / 255)
        camera_o3d.points = o3d.utility.Vector3dVector(pc)

    return camera_o3d


def crop_points(points, color, crop_min, crop_max):
    valid = points[:, 0] < crop_max[0]
    valid = np.logical_and(points[:, 1] < crop_max[1], valid)
    valid = np.logical_and(points[:, 2] < crop_max[2], valid)
    valid = np.logical_and(points[:, 0] > crop_min[0], valid)
    valid = np.logical_and(points[:, 1] > crop_min[1], valid)
    valid = np.logical_and(points[:, 2] > crop_min[2], valid)

    new_points = points[valid]
    new_color = color[valid]
    new_points[:, 1] = -new_points[:, 1]
    new_points[:, 2] = -new_points[:, 2]

    return new_points, new_color


def vis_rgb_depth(rgb, depth, show_depth=False):
    rgb_image = (rgb).astype(np.uint8)[..., [2, 1, 0]]
    if show_depth:
        depth = np.clip(depth, 0, 100)
        depth_image_normalized = cv2.normalize(depth, None, 0, 255,
                                               cv2.NORM_MINMAX)

        # Convert to 8-bit image
        depth_image_8bit = np.uint8(depth_image_normalized)

        # Apply a color map to enhance visualization (optional)
        depth_image_colormap = cv2.applyColorMap(depth_image_8bit,
                                                 cv2.COLORMAP_JET)

        return np.concatenate([rgb_image, depth_image_colormap], axis=0)
    else:
        return rgb_image
