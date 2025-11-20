# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Utility to convert a OBJ/STL/FBX into USD format.

The OBJ file format is a simple data-format that represents 3D geometry alone — namely, the position
of each vertex, the UV position of each texture coordinate vertex, vertex normals, and the faces that
make each polygon defined as a list of vertices, and texture vertices.

An STL file describes a raw, unstructured triangulated surface by the unit normal and vertices (ordered
by the right-hand rule) of the triangles using a three-dimensional Cartesian coordinate system.

FBX files are a type of 3D model file created using the Autodesk FBX software. They can be designed and
modified in various modeling applications, such as Maya, 3ds Max, and Blender. Moreover, FBX files typically
contain mesh, material, texture, and skeletal animation data.
Link: https://www.autodesk.com/products/fbx/overview


This script uses the asset converter extension from Isaac Sim (``omni.kit.asset_converter``) to convert a
OBJ/STL/FBX asset into USD format. It is designed as a convenience script for command-line use.


positional arguments:
  input               The path to the input mesh (.OBJ/.STL/.FBX) file.
  output              The path to store the USD file.

optional arguments:
  -h, --help                    Show this help message and exit
  --make-instanceable,          Make the asset instanceable for efficient cloning. (default: False)
  --collision-approximation     The method used for approximating collision mesh. Defaults to convexDecomposition.
                                Set to \"none\" to not add a collision mesh to the converted mesh. (default: convexDecomposition)
  --mass                        The mass (in kg) to assign to the converted asset. (default: None)

"""
"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Utility to convert a mesh file into USD format.")

parser.add_argument(
    "--make-instanceable",
    action="store_true",
    default=True,
    help="Make the asset instanceable for efficient cloning.",
)
parser.add_argument(
    "--collision-approximation",
    type=str,
    default="convexDecomposition",
    choices=["convexDecomposition", "convexHull", "none"],
    help=('The method used for approximating collision mesh. Set to "none" '
          "to not add a collision mesh to the converted mesh."),
)
parser.add_argument(
    "--mass",
    type=float,
    default=None,
    help=
    "The mass (in kg) to assign to the converted asset. If not provided, then no mass is added.",
)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""

import contextlib
import os

import carb
import isaacsim.core.utils.stage as stage_utils
import omni.kit.app

from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.assets import check_file_path
from isaaclab.utils.dict import print_dict
import shutil
import trimesh
import glob
import torch
import isaaclab.utils.math as math_utils
import numpy as np


def main():
    object_dir = f"/media/ensu/data/datasets/objaverse"
    dest_directory = "/home/ensu/Documents/weird/IsaacLab_assets/assets/objaverse/"
    os.makedirs(object_dir, exist_ok=True)

    object_class = {}
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
        "book": 20,
        "strawberry": 20,
        "scissors": 20,
        "rabbit": 20,
        "soupspoon": 20,
        "spoon": 20,
        "strawberry": 20
    }

    for object_name in object_class.keys():
        object_list = []
        object_list = glob.glob(object_dir + f"/{object_name}/*/*/*.glb")
        num_choices = object_class[object_name]
        object_list = np.random.choice(object_list, num_choices)

        index = 0

        for i, name in enumerate(object_list):

            dest_dir = os.path.join(dest_directory, name.split("/")[-4])

            mesh_path = name

            dest_path = os.path.join(
                dest_dir, f"{object_name}_{index}/rigid_object.usd")
            os.makedirs(dest_dir + f"/{object_name}_{index}", exist_ok=True)
            size_mb = os.path.getsize(f"{mesh_path}") / (1024 * 1024)
            if size_mb > 5:
                continue
            mesh = trimesh.load(mesh_path)
            try:
                raw_mesh = trimesh.util.concatenate(mesh)
                vertices = torch.as_tensor(
                    raw_mesh.vertices).unsqueeze(0).to(dtype=torch.float32)
                # transformed_vertices = vertices.clone()
                transformed_vertices = math_utils.transform_points(
                    points=vertices,
                    pos=torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
                    quat=torch.tensor([[0.707, 0.707, 0.0, 0.0]],
                                      dtype=torch.float32),
                )[0]

                min_z, max_z = transformed_vertices[:, 2].min(
                ), transformed_vertices[:, 2].max()
                min_x, max_x = transformed_vertices[:, 0].min(
                ), transformed_vertices[:, 0].max()
                min_y, max_y = transformed_vertices[:, 1].min(
                ), transformed_vertices[:, 1].max()
                if (max_z - min_z) > 100:
                    continue

                scale_factor = (np.random.rand(1)[0] * 0.15 + 0.10) / (max_z -
                                                                       min_z)
                x_scale_constraint = ((np.random.rand(1)[0] * 0.08 + 0.08) /
                                      (max_x - min_x))
                y_scale_constraint = ((np.random.rand(1)[0] * 0.08 + 0.08) /
                                      (max_y - min_y))

                scale_factor = np.min(
                    [x_scale_constraint, scale_factor, y_scale_constraint])
                mesh.apply_scale(scale_factor.tolist())

                scale_mesh = trimesh.util.concatenate(mesh)

                vertices = scale_mesh.vertices

                new_min_x, new_max_x = vertices[:, 0].min(), vertices[:,
                                                                      0].max()
                new_min_y, new_max_y = vertices[:, 1].min(), vertices[:,
                                                                      1].max()
                new_min_z, new_max_z = vertices[:, 2].min(), vertices[:,
                                                                      2].max()
                center_x = (new_min_x + new_max_x) / 2
                center_y = (new_min_y + new_max_y) / 2
                center_z = (new_min_z + new_max_z) / 2

                x_length = max_x - min_x
                y_length = max_y - min_y

                if "pan" in mesh_path:

                    if x_length / y_length > 4.0 or y_length / x_length > 4.0:
                        continue

                    if x_length > y_length:
                        mesh.apply_translation(
                            [-center_x - 0.05, -center_y, -center_z])
                    else:

                        mesh.apply_translation(
                            [-center_x, -center_y, -center_z + 0.08])
                elif "basket" in mesh_path or "bucket" in mesh_path:
                    mesh.apply_translation([-center_x, -new_max_y, -center_z])
                else:
                    mesh.apply_translation([-center_x, -center_y, -center_z])

                mesh.export(
                    dest_dir +
                    f"/{object_name}_{index}/{object_name}_{index}.glb")
            except:
                continue

            mesh_path = dest_dir + f"/{object_name}_{index}/{object_name}_{index}.glb"

            # Mass properties

            mass_props = schemas_cfg.MassPropertiesCfg(mass=1)
            rigid_props = schemas_cfg.RigidBodyPropertiesCfg()

            # Collision properties
            collision_props = schemas_cfg.CollisionPropertiesCfg(
                collision_enabled=args_cli.collision_approximation != "none")

            # Create Mesh converter config
            mesh_converter_cfg = MeshConverterCfg(
                mass_props=mass_props,
                rigid_props=rigid_props,
                collision_props=collision_props,
                asset_path=mesh_path,
                force_usd_conversion=True,
                usd_dir=os.path.dirname(dest_path),
                usd_file_name=os.path.basename(dest_path),
                make_instanceable=False,
                collision_approximation=args_cli.collision_approximation,
                rotation=(0.707, 0.707, 0.0, 0.0),
                # scale=(0.01, 0.01, 0.01),
            )

            try:
                mesh_converter = MeshConverter(mesh_converter_cfg)

                print("Mesh importer output:")
                print(f"Generated USD file: {mesh_converter.usd_path}")
                print("-" * 80)
                print("-" * 80)
            except:
                print(
                    f"error in mesh converter:{mesh_converter_cfg.asset_path}")
                pass
            index += 1


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
