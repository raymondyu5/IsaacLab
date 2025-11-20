import yaml
import os
import trimesh
import numpy as np

size_dict = {}


def calculate_bbox_4_corners(vertices):
    """
    Calculate the 4 corners of a bounding box on the specified plane (bottom or top).
    
    :param vertices: The vertices of the object's mesh.
    :param plane: Either 'bottom' or 'top' to specify which plane's corners to calculate.
    :return: 4 corners of the bounding box for the specified plane.
    """
    # Calculate the min and max extents in x, y, z dimensions
    x_max, y_max, z_max = vertices.max(axis=0)
    x_min, y_min, z_min = vertices.min(axis=0)

    # 4 corners on the bottom or top plane
    corners = [
        [x_min, y_min, z_min],  # Bottom-left (or top-left)
        [x_max, y_min, z_min],  # Bottom-right (or top-right)
        [x_max, y_max, z_max],  # Top-right (or bottom-right)
        [x_min, y_max, z_max],  # Top-left (or bottom-left)
    ]

    return corners


def save_yaml(object_list, path, deformable=True):

    for object_name in object_list:
        size_dict[object_name] = {}
        object_path = os.path.join(path, object_name)
        mesh = trimesh.load(object_path + f"/{object_name}.obj")
        vertices = mesh.vertices  #[:, [0, 2, 1]]
        if deformable:
            vertices = mesh.vertices[:, [0, 2, 1]]
        # vertices[:, 2] -= np.min(vertices[:, 2])

        corners = calculate_bbox_4_corners(vertices=vertices)

        if (np.array(vertices.max(axis=0) - vertices.min(axis=0)) > 1).any():
            vertices /= 100
        x_max, y_max, z_max = vertices.max(axis=0)
        x_min, y_min, z_min = vertices.min(axis=0)

        # Convert NumPy arrays to lists to avoid !!python/tuple issues

        size_dict[object_name]["corners"] = [[
            float(corner[0]),
            float(corner[1]),
            float(corner[2])
        ] for corner in corners]

    return size_dict


deformable_objects_dir = "source/assets/Plush/mesh"
deformable_object_list = os.listdir(deformable_objects_dir)
size_dict = save_yaml(deformable_object_list, deformable_objects_dir)

rigid_objects_dir = "source/assets/ycb/mesh"
rigid_object_list = os.listdir(rigid_objects_dir)
size_dict = save_yaml(rigid_object_list, rigid_objects_dir, deformable=False)
# Save the dictionary to a YAML file without python-specific tags
with open('source/assets/object_size.yaml', 'w') as yaml_file:
    yaml.dump(size_dict, yaml_file, default_flow_style=False)
