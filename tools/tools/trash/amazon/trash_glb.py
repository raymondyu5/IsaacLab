import os
import trimesh
import numpy as np

dir_path = "source/assets/bulb/"

object_names = ["bulb_base01"]
target_size = 0.25  # meter
for name in object_names:
    obj = trimesh.load(f"{dir_path}/{name}/{name}.stl")

    R = trimesh.transformations.rotation_matrix(
        np.radians(-90),  # 90 degrees
        [1, 0, 0]  # y-axis
    )

    obj.apply_transform(R)

    if isinstance(obj, trimesh.Scene):
        obj = trimesh.util.concatenate(obj)

    x_min, y_min, z_min = obj.bounds[0]
    x_max, y_max, z_max = obj.bounds[1]
    width = np.abs(x_max - x_min)
    height = np.abs(y_max - y_min)
    depth = np.abs(z_max - z_min)
    max_length = max(width, height, depth)

    # 1. Scale first
    obj.apply_scale(np.array([0.003, 0.003, 0.003]))

    # R = trimesh.transformations.rotation_matrix(
    #     np.radians(-180),  # 90 degrees
    #     [0, 1, 0]  # y-axis
    # )
    # obj.apply_transform(R)

    # R = trimesh.transformations.rotation_matrix(
    #     np.radians(180),  # 90 degrees
    #     [0, 1, 0]  # y-axis
    # )
    # obj.apply_transform(R)

    # 3. Recompute bounding box after scaling & rotation
    mesh = trimesh.util.concatenate(obj)
    x_min, y_min, z_min = mesh.bounds[0]
    x_max, y_max, z_max = mesh.bounds[1]

    mid_point = np.array([-(x_min + x_max) / 2, -y_min, -(z_min + z_max) / 2])

    # 4. Translate so centered at origin
    obj.apply_translation(mid_point)

    # 5. Save
    obj.export(f"{dir_path}/{name}/recenter_{name}.glb")
