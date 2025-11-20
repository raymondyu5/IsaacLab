import os
import yaml
import trimesh
import isaaclab.utils.math as math_utils
import torch
import numpy as np
import random
# Folder containing your object files
object_folder = "/home/ensu/Documents/weird/IsaacLab_assets/assets/objaverse"  # <-- change this
output_yaml_file = "source/config/task/hand_env/object/objaverse_sample.yaml"

# Get all file names (or folders) inside object_folder
object_class = os.listdir(object_folder)
# object_class.remove("bottle_cup")

# Storage for all YAML entries
yaml_data = {}

total_count = 0
object_class = {
    "apple": 20,
    "banana": 20,
    "bottle": 20,
    "bowl": 20,
    # "pen": 20,
    "cup": 20,
    "tomato": 20,
    "water_bottle": 20,
    "toy": 20,
    "teddy_bear": 20,
    "basket": 20,
    # "book": 20,
    "strawberry": 20,
    # "scissors": 20,
    "rabbit": 20,
    "soupspoon": 20,
    # "spoon": 20,
    "strawberry": 20
}

object_class = list(object_class.keys(
))  # ["bottle", "bowl", "cup", "water_bottle", "teddy_bear"]
# Loop through each object
sizes = []
for _, obj_class in enumerate(object_class):
    # Skip hidden files or irrelevant ones
    object_names = os.listdir(f"{object_folder}/{obj_class}")
    object_names = random.sample(object_names, min([20, len(object_names)
                                                    ]))  # Limit to 100 objects

    for index, name in enumerate(object_names):
        # name = "object_" + str(index)

        try:
            mesh = trimesh.load(
                f"{object_folder}/{obj_class}/{name}/{name}.glb", )
        except:
            continue
        mesh = trimesh.util.concatenate(mesh)
        vertices = torch.as_tensor(
            mesh.vertices).unsqueeze(0).to(dtype=torch.float32)
        transformed_vertices = math_utils.transform_points(
            points=vertices,
            pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
            quat=torch.tensor([[0.707, 0.707, 0.0, 0.0]], dtype=torch.float32),
        )[0]

        min_z, max_z = transformed_vertices[:, 2].min(
        ), transformed_vertices[:, 2].max()

        # Create the pattern
        size_mb = os.path.getsize(
            f"source/assets/objaverse/{obj_class}/{name}/rigid_object.usd") / (
                1024 * 1024)
        if size_mb > 5:
            continue
        sizes.append(size_mb)
        scale = 1.0

        entry = {
            'path':
            f"source/assets/objaverse/{obj_class}/{name}/rigid_object.usd",
            'kinematic_enabled': False,
            'scale': [scale, scale, scale],
            'pos': [0.50, 0.0, -min_z.tolist() * scale],
            'rot': {
                'axis': [0],
                'angles': [1.57]
            },
            'pose_range': {
                'x': [-0.10, 0.10],
                'y': [-0.20, 0.20],
                'z': [0.0, 0.0],
                'roll': [0.0, 0.0],
                'yaw': [-3.14, 3.14]
            }
        }

        # Add to YAML structure
        yaml_data[f"{obj_class}_{total_count}"] = entry
        total_count += 1
    #     if total_count>=1200:
    #         break
    # if total_count>=1200:
    #         break

full_yaml_data = {"params": {"RigidObject": yaml_data}}

# Save everything into a YAML file
with open(output_yaml_file, 'w') as f:
    yaml.dump(full_yaml_data, f, sort_keys=False, default_flow_style=False)

print(f"Saved {len(yaml_data)} objects to {output_yaml_file}")
