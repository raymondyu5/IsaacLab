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


def main():
    obhect_class = "grab"

    # check valid file path
    if obhect_class == "shapenet":
        object_dir = f"/home/ensu/Documents/weird/IsaacLab_assets/assets/shapenet/bottle"
        dest_dir = "/home/ensu/Documents/weird/IsaacLab_assets/assets/shapenet/bottle"
        os.makedirs(dest_dir, exist_ok=True)
        object_list = os.listdir(object_dir)
    elif obhect_class == "grab":

        object_dir = "/media/ensu/data/datasets/grab/contact_mesh"
        dest_dir = "/home/ensu/Documents/weird/IsaacLab_assets/assets/grab/"
        object_list = os.listdir(object_dir)

    for index, name in enumerate(object_list):

        mesh_path = os.path.join(object_dir, name)

        object_name = name.split(".")[0]
        dest_path = os.path.join(dest_dir, f"{object_name}/rigid_object.usd")
        os.makedirs(dest_dir + f"/{object_name}", exist_ok=True)

        src_file = object_dir + "/" + name

        dst_file = f"{dest_dir}/{object_name}"
        if os.path.isfile(src_file):  # make sure it's a file
            shutil.copy2(src_file, dst_file)  # copy2 preserves metadata

        # Mass properties
        if args_cli.mass is not None:
            mass_props = schemas_cfg.MassPropertiesCfg(mass=args_cli.mass)
            rigid_props = schemas_cfg.RigidBodyPropertiesCfg()
        else:
            mass_props = None
            rigid_props = None

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
        )

        try:
            mesh_converter = MeshConverter(mesh_converter_cfg)

            print("Mesh importer output:")
            print(f"Generated USD file: {mesh_converter.usd_path}")
            print("-" * 80)
            print("-" * 80)
        except:
            print(f"error in mesh converter:{mesh_converter_cfg.asset_path}")
            pass


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
