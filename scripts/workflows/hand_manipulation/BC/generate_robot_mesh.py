from scripts.workflows.hand_manipulation.BC.robot_wrapper import RobotWrapper
import numpy as np
import h5py

from scripts.workflows.hand_manipulation.BC.synthesis_real_pc import SynthesizeRealRobotPC
from tools.visualization_utils import *
import time
import open3d as o3d


def load_data(file_path):
    """
    Load data from an HDF5 file.
    """

    demo_data = []

    with h5py.File(file_path, 'r') as data:

        demo_keys = list(data["data"].keys())

        for key in demo_keys:

            demo = data["data"][key]

            actions = np.array(demo["obs"]["right_hand_joint_pos"])

            demo_data.append(actions)

    return demo_data


if __name__ == "__main__":
    robot_wrapper = RobotWrapper(
        "source/assets/robot/franka/urdf/franka_description/robots/panda_arm_hand.urdf"
    )
    print(list(robot_wrapper.model.names))
    robot_wrapper.sim2real_joint_mapping()
    robot_wrapper.get_target_link_index()

    demo_data = load_data(
        "logs/data_0705/retarget_visionpro_data/rl_data/raw_data/banana.hdf5")
    # pcd_synthesizer = SynthesizeRealRobotPC(
    #     mesh_dir=
    #     "source/assets/robot/franka/urdf/franka_description/meshes/visual",
    # target_link_name=robot_wrapper.target_link_names)

    for data in demo_data:
        for action in data[-10:]:
            start = time.time()
            # all_action = np.zeros(robot_wrapper.dof)
            all_action = action

            robot_wrapper.compute_forward_kinematics(
                all_action[robot_wrapper.real2sim_index])
            robot_wrapper.extract_meshcat_visualizer()

            # all_link_pose = robot_wrapper.get_target_link_pose()
            # sythesis_pc = pcd_synthesizer.synthesize_pc(all_link_pose, )
            sythesis_pc = robot_wrapper.extract_meshcat_visualizer()
            # import pdb
            # pdb.set_trace()
            o3d_pcd = vis_pc(sythesis_pc, )
            o3d.visualization.draw_geometries([o3d_pcd])

            # import pdb
            # pdb.set_trace()
            print(f"Time taken: {time.time() - start:.4f} seconds")
