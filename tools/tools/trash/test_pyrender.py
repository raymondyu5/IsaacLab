import numpy as np
import h5py
import torch
import os
import cv2
import imageio
data = h5py.File('/media/aurmr/data1/isaac_data/logs/reach/test_data.hdf5',
                 'r')["data"]

extrinsic = np.load('extrinsic.npy')
intrinsic = np.load('intrinsic.npy')
num_camera = len(extrinsic)

JOINT_COLOR_MAP = {
    1: "red",
    3: "green",
    5: "purple",
}

from tools.trash.vlm_failure.render_data import JointMarker
import isaaclab.utils.math as math_utils

save_dir = "/media/aurmr/data1/isaac_data/logs/reach/pyrender"
cleanup_dataset = "/media/aurmr/data1/isaac_data/logs/reach/raw_clean_data"
os.makedirs(save_dir, exist_ok=True)
os.makedirs(f"{save_dir}/videos", exist_ok=True)

panda_link_names = [
    # "panda_link0", 
    "panda_link1", 
    # "panda_link2", "panda_link3",
    "panda_link4",
    # "panda_link5",
    "panda_link6",
    # "panda_link7",
    "panda_hand"
]

camera_scale = 0.07
num_panda_link = len(panda_link_names)
action_marker = JointMarker(
    image_width=225,
    image_height=225,
    camera_scales=[camera_scale] * num_panda_link,
    sphere_radius=1.0,
    znear=0.00001,
    zfar=100.0,
)
for demo_key in data.keys():
    links_state_name = {}
    obs = data[demo_key]["obs"]
    num_steps = len(obs["panda_hand_state"])
    os.makedirs(f"{save_dir}/{demo_key}", exist_ok=True)
    demo_index = demo_key.split("_")[-1]
    # npz_file = f"{cleanup_dataset}/episode_{demo_index}.npz"
    # npz_images = np.load(npz_file, allow_pickle=True)
    video_writer = imageio.get_writer(f"{save_dir}/videos/{demo_key}.mp4", fps=30)

    for link_name in panda_link_names:
        panda_state = torch.as_tensor(np.array(obs[f"{link_name}_state"]))
        panda_quat = panda_state[:, 3:7]

        panda_matricies = np.tile(np.identity(4), (panda_state.shape[0], 1, 1))
        panda_matricies[:, :3, :3] = math_utils.matrix_from_quat(
            panda_quat).numpy()
        panda_matricies[:, :3, 3] = panda_state[:, :3].numpy()
        links_state_name[link_name] = panda_matricies
    
    for index_step in range(num_steps):
        rgb_images = []
        
        for index_cam in range(num_camera):
            os.makedirs(f"{save_dir}/{demo_key}/cam_{index_cam}", exist_ok=True)
            joint_matrices = []
            sphere_colors = []
            
            for link_name in panda_link_names:
                joint_matrices.append(links_state_name[link_name][index_step])
                sphere_colors.append(
                    np.random.choice(["red", "green", "purple"]))

            rendered_img = action_marker.render_action(
                cam_intrinsic=intrinsic[index_cam],
                cam_extrinsic=extrinsic[index_cam],
                joint_matrices=joint_matrices,
                camera_scale=camera_scale,
                joint_opens=[0] * num_panda_link,
                sphere_colors=sphere_colors)
            
            rgb_image = obs["rgb"][index_step][index_cam]
            
            

            mask = np.all(rendered_img == [255, 255, 255], axis=-1)

            # Replace white pixels in rendered_img with corresponding pixels from rgb_image
            combined_img = np.where(mask[..., None], rgb_image, rendered_img)
            rgb_images.append(combined_img)

            cv2.imwrite(
                f"{save_dir}/{demo_key}/cam_{index_cam}/{index_step}.png",
                combined_img)
     
        stitched_image = np.concatenate(rgb_images, axis=1) 
        video_writer.append_data(stitched_image[:,:,::-1])
