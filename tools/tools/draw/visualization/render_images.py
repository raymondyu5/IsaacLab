import zarr
import imageio
import os
import cv2
import numpy as np

data_path = "/media/ensu/data/datasets/teleop_data/render_image"
object_list = os.listdir(data_path)

from tools.draw.mesh_visualizer.hand_mesh_synthesize import SynthesizeRealRobotPC

mesh_dir = "source/assets/robot/leap_hand_v2/glb_mesh/"
target_link_name = [
    'palm_lower',
    'mcp_joint',
    'pip',
    'dip',
    'fingertip',
    'mcp_joint_2',
    'dip_2',
    'fingertip_2',
    'mcp_joint_3',
    'pip_3',
    'dip_3',
    'fingertip_3',
    'thumb_pip',
    'thumb_dip',
    'thumb_fingertip',
    'pip_2',
    'thumb_right_temp_base',
]
synthesize_pc = SynthesizeRealRobotPC(mesh_dir, target_link_name)

link_names = [
    'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4',
    'panda_link5', 'panda_link6', 'panda_link7', 'palm_upper', 'palm_lower',
    'mcp_joint', 'thumb_right_temp_base', 'mcp_joint_2', 'mcp_joint_3', 'pip',
    'thumb_pip', 'pip_2', 'pip_3', 'dip', 'thumb_dip', 'dip_2', 'dip_3',
    'fingertip', 'thumb_fingertip', 'fingertip_2', 'fingertip_3',
    'fingertip_sensor', 'thumb_sensor', 'fingertip_2_sensor',
    'fingertip_3_sensor'
]

for object_name in object_list:
    object_dir = os.path.join(data_path, object_name)

    episode_list = os.listdir(object_dir)
    for episode_name in episode_list:
        episode_dir = os.path.join(object_dir, episode_name)
        link_sorted_pose = np.load(
            os.path.join(episode_dir, "link_sorted_pose.npy"))

        for i in range(0, len(link_sorted_pose), 10):

            color = synthesize_pc.render_pose(link_sorted_pose[i],
                                              target_link_name,
                                              render_object=False)
            cv2.imwrite(os.path.join(episode_dir, f"{i:05d}.png"),
                        color[..., :3][..., ::-1])
