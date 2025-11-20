import os
import yaml
import trimesh
import isaaclab.utils.math as math_utils
import torch
import numpy as np
import random
# Folder containing your object files
object_folder = "/home/weirdlab/Documents/IsaacLab_assets/assets/ycb/dexgrasp/recenter_objects"  # <-- change this
output_yaml_file = "source/config/task/hand_env/object/recenter_orient_ycb.yaml"

# Get all file names (or folders) inside object_folder
object_class = os.listdir(object_folder)
yaml_data = {}
# Loop through each object
exclueded_objects = ["wood_block", "cracker_box", "pitcher_base"]
for _, obj_class in enumerate(object_class):
    if obj_class in exclueded_objects:
        continue

    mesh = trimesh.load(
        f"{object_folder}/{obj_class}/textured_recentered.obj", )

    mesh = trimesh.util.concatenate(mesh)

    vertices = torch.as_tensor(
        mesh.vertices).unsqueeze(0).to(dtype=torch.float32)
    vertices = math_utils.transform_points(
        vertices, torch.zeros((1, 3)),
        torch.as_tensor([[0.707, 0.707, 0.0, 0.0]]))

    min_z, max_z = vertices[:, 2].min(), vertices[:, 2].max()

    entry = {
        'path':
        f"source/assets/ycb/dexgrasp/recenter_objects/{obj_class}/recenter_rigid_object.usd",
        'kinematic_enabled': False,
        'scale': [1.0, 1.0, 1.0],
        'pos': [0.50, 0.0, -min_z.tolist()],
        'rot': {
            'axis': [0],
            'angles': [1.57]
        },
        'pose_range': {
            'x': [-0.20, 0.20],
            'y': [-0.20, 0.20],
            'z': [0.0, 0.0],
            'roll': [0.0, 0.0],
            'yaw': [-3.14, 3.14]
        }
    }

    # Add to YAML structure
    yaml_data[obj_class] = entry

full_yaml_data = {"params": {"RigidObject": yaml_data}}

# Save everything into a YAML file
with open(output_yaml_file, 'w') as f:
    yaml.dump(full_yaml_data, f, sort_keys=False, default_flow_style=False)

print(f"Saved {len(yaml_data)} objects to {output_yaml_file}")
