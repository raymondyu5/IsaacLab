import zarr
import imageio
import os
import cv2
import numpy as np

zarr_path = "logs/data_0705/retarget_visionpro_data/rl_data/data/reactive_vae/ycb/image/tomato_soup_can/episode_0.zarr"

cache_dir = "logs/trash/test/"
os.makedirs(cache_dir, exist_ok=True)
data = zarr.open(zarr_path, mode='r')

images = data["data/rgb_0"]

from tools.draw.mesh_visualizer.hand_mesh_synthesize import SynthesizeRealRobotPC

video_writer = imageio.get_writer(os.path.join(cache_dir, "video.mp4"), fps=20)
mesh_dir = "source/assets/robot/leap_hand_v2/glb_mesh/"
target_link_name = [
    # "panda_link0", "panda_link1", "panda_link2", "panda_link3", "panda_link4",
    # "panda_link5", "panda_link6", "panda_link7",
    "palm_lower",
    "mcp_joint",
    "pip",
    "dip",
    "fingertip",
    "mcp_joint_2",
    "dip_2",
    "fingertip_2",
    "mcp_joint_3",
    "pip_3",
    "dip_3",
    "fingertip_3",
    "thumb_temp_base",
    "thumb_pip",
    "thumb_dip",
    "thumb_fingertip",
    "pip_2",
    "thumb_right_temp_base"
]
synthesize_pc = SynthesizeRealRobotPC(mesh_dir, target_link_name)
link_pose = []
for i in range(len(images)):
    video_writer.append_data(images[i])
    cv2.imwrite(os.path.join(cache_dir, f"{i:05d}.png"),
                images[i][..., :3][..., ::-1])
video_writer.close()

import pdb

pdb.set_trace()

link_pose = data["data/body_state"]
link_names = [
    'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4',
    'panda_link5', 'panda_link6', 'panda_link7', 'palm_upper', 'palm_lower',
    'mcp_joint', 'thumb_right_temp_base', 'mcp_joint_2', 'mcp_joint_3', 'pip',
    'thumb_pip', 'pip_2', 'pip_3', 'dip', 'thumb_dip', 'dip_2', 'dip_3',
    'fingertip', 'thumb_fingertip', 'fingertip_2', 'fingertip_3',
    'fingertip_sensor', 'thumb_sensor', 'fingertip_2_sensor',
    'fingertip_3_sensor'
]

link_sorted_pose = []
target_link_name = []
for _, link_name in enumerate(list(synthesize_pc.mesh_dict.keys())):

    if link_name not in link_names:
        continue
    target_link_name.append(link_name)

    index = link_names.index(link_name)

    trajectories_pose = link_pose[:, index]
    link_sorted_pose.append(trajectories_pose)

link_sorted_pose = np.array(link_sorted_pose).transpose(1, 0, 2)

hand_dir = cache_dir.replace("image", "hand_pose")
os.makedirs(hand_dir, exist_ok=True)
rigid_object_state = np.array(data["data/right_manipulated_object_pose"])
for i in range(40, len(link_pose)):

    color = synthesize_pc.render_pose(link_sorted_pose[i],
                                      target_link_name,
                                      rigid_object_state[i],
                                      render_object=True)
    cv2.imwrite(os.path.join(hand_dir, f"{i:05d}.png"),
                color[..., :3][..., ::-1])
