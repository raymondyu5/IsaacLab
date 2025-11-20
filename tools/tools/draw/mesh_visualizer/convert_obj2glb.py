import trimesh
import os
import numpy as np
# target_link_name = [
#     "palm_lower", "mcp_joint", "pip", "dip", "fingertip", "mcp_joint_2",
#     "dip_2", "fingertip_2", "mcp_joint_3", "pip_3", "dip_3", "fingertip_3",
#     "thumb_temp_base", "thumb_pip", "thumb_dip", "thumb_fingertip", "pip_2"
# ]
# link_name_mapping = {
#     "panda_link0": "panda_link0.obj",
#     "panda_link1": "panda_link1.obj",
#     "panda_link2": "panda_link2.obj",
#     "panda_link3": "panda_link3.obj",
#     "panda_link4": "panda_link4.obj",
#     "panda_link5": "panda_link5.obj",
#     "panda_link6": "panda_link6.obj",
#     "panda_link7": "panda_link7.obj",
#     "palm_lower": "palm_lower",
#     "mcp_joint": "mcp_joint",
#     "pip": "pip",
#     "dip": "dip",
#     "fingertip": "fingertip",
#     "mcp_joint_2": "mcp_joint",
#     "dip_2": "dip",
#     "fingertip_2": "fingertip",
#     "mcp_joint_3": "mcp_joint",
#     "pip_3": "pip",
#     "dip_3": "dip",
#     "fingertip_3": "fingertip",
#     "thumb_temp_base": "pip",
#     "thumb_pip": "thumb_pip",
#     "thumb_dip": "thumb_dip",
#     "thumb_fingertip": "thumb_fingertip",
#     "pip_2": "pip",
#     "thumb_temp_base": "thumb_temp_base"
# }
# mesh_dir = "/home/ensu/Documents/weird/IsaacLab_assets/assets/robot/leap_hand_v2/raw_mesh"
# for link_name in target_link_name:
#     mesh = trimesh.load(mesh_dir + f"/{link_name_mapping[link_name]}.obj", )

#     mesh.export(mesh_dir + f"/{link_name_mapping[link_name]}.glb")

mesh_dir = "source/assets/ycb/dexgrasp/recenter_objects"
obj_files = os.listdir(mesh_dir)

obj_files = ["bleach_cleanser"]
for file in obj_files:
    obj_mesh = trimesh.load(mesh_dir + f"/{file}/textured_recentered.obj", )

    R = trimesh.transformations.rotation_matrix(
        np.radians(-90),  # 90 degrees
        [1, 0, 0]  # y-axis
    )
    obj_mesh.apply_transform(R)

    obj_mesh.export(mesh_dir + f"/{file}/textured_recentered.glb")
