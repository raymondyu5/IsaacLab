import viser
import numpy as np
from scipy.spatial.transform import Rotation as R
import time

server = viser.ViserServer(host='0.0.0.0', port=8080)

# Dictionary to store mesh objects
meshes = {}
import trimesh

data_list = [
    "left_palm_lower",
    "left_mcp_joint",
    "left_mcp_joint_2",
    "left_mcp_joint_3",
    "left_pip",
    "left_thumb_pip",
    "left_pip_2",
    "left_pip_3",
    "left_dip",
    "left_thumb_dip",
    "left_dip_2",
    "left_dip_3",
    "left_fingertip",
    "left_thumb_fingertip",
    "left_fingertip_2",
    "left_fingertip_3",
    "left_thumb_left_temp_base",
]
# Load meshes into the scene
for link_name in data_list:

    if "palm" in link_name:
        mesh_name = link_name
    elif "thumb" in link_name:

        mesh_name = "_".join(link_name.split("_")[1:])

    else:
        parts = link_name.split("_")[1:]
        if parts[-1].isdigit():
            mesh_name = "_".join(parts[:-1])  # Drop the last numeric part
        else:
            mesh_name = "_".join(parts)
    mesh = trimesh.load(
        f"//home/ensu/Documents/weird/IsaacLab/source/assets/robot/leap_hand_v2/raw_mesh/{mesh_name}.obj"
    )
    mesh = server.scene.add_mesh_trimesh(
        name=link_name,
        mesh=mesh,
        position=(0.0, 0.0, 0.0),
        wxyz=(1.0, 0.0, 0.0, 0.0),
        scale=1,
    )
    meshes[link_name] = mesh

while True:
    for i, link_name in enumerate(meshes.keys()):
        # Just for demo: make them orbit in a circle
        angle = time.time() + i
        x = np.cos(angle) * 0.2 * (i + 1)
        y = np.sin(angle) * 0.2 * (i + 1)
        z = 0.2

        quat = R.from_euler('z', angle).as_quat()  # (x, y, z, w)
        quat_wxyz = (quat[3], quat[0], quat[1], quat[2])

        meshes[link_name].position = (x, y, z)
        meshes[link_name].wxyz = quat_wxyz
