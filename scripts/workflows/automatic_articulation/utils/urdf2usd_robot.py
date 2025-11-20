import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from PIL import Image
import os
from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import parser

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
from pxr import Gf, Kind, Sdf, Usd, UsdGeom, UsdShade, Vt, UsdPhysics, UsdLux
import transformations

from scipy.spatial.transform import Rotation
from pxr import UsdPhysics, UsdUtils, Sdf, PhysxSchema
from load_obj_utils import load_obj
import isaaclab.utils.math as math_utils
import torch

global_sdf = False
from scripts.workflows.automatic_articulation.utils.usd_assembly_helper import *


def create_rigid_collision(prim, set_sdf=False, sdf_resolution=512):
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.CollisionAPI.Apply(prim)
    meshcollisionAPI = UsdPhysics.MeshCollisionAPI.Apply(prim)
    if set_sdf and global_sdf:
        meshcollisionAPI.CreateApproximationAttr().Set("sdf")
        meshCollision = PhysxSchema.PhysxSDFMeshCollisionAPI.Apply(prim)
        meshCollision.CreateSdfResolutionAttr().Set(sdf_resolution)
    else:
        meshcollisionAPI.CreateApproximationAttr().Set("convexHull")


class AssembleArticulation:

    def __init__(self,
                 ckpt_dir,
                 root='/world',
                 usda_name='kitchen.usda',
                 urdf_dict=None):
        # Path to config YAML file.
        # ckpt_dir = "./scannetpp/2e67a32314/"
        self.ckpt_dir = ckpt_dir  #"/media/aurmr/data1/weird/IsaacLab/tools/auto_articulation/asset"
        self.usda_name = usda_name
        self.urdf_dict = urdf_dict
        self.root = root
        self.scale = (1, 1, 1)

        self.set_env()

    def set_env(self):

        self.stage = Usd.Stage.CreateNew(os.path.join("logs", self.usda_name))
        UsdGeom.SetStageUpAxis(self.stage, UsdGeom.Tokens.z)

        modelRoot = UsdGeom.Xform.Define(self.stage, self.root)
        Usd.ModelAPI(modelRoot).SetKind(Kind.Tokens.component)

        light = UsdLux.DomeLight.Define(self.stage, f"{self.root}/light")

        # Set DomeLight-specific attributes
        light.CreateExposureAttr(3.0)
        light.CreateIntensityAttr(100.0)
        light.CreateColorAttr((1.0, 1.0, 1.0))

        stage = create_xform(
            self.stage, (1, 1, 1), (0.0, 0.0, 0.0),
            transformations.quaternion_from_euler(0.0, 0.0, 0.0, axes='sxyz'),
            f"{self.root}")
        stage = create_xform(
            stage, (1, 1, 1), (0.0, 0.0, 0.0),
            transformations.quaternion_from_euler(0.0, 0.0, 0.0, axes='sxyz'),
            f"{self.root}/joints")

        prim = stage.GetPrimAtPath(f"{self.root}")
        stage.SetDefaultPrim(prim)

    def convert(self):

        self.assembly_link()
        self.assembly_joint()

        fixed_joint = UsdPhysics.FixedJoint.Define(self.stage,
                                                   self.root + "/root_joint")

        child_prim = self.stage.GetPrimAtPath(self.root + "/panda_link0")

        fixed_joint.GetBody1Rel().SetTargets([child_prim.GetPath()])

        prim = self.stage.GetPrimAtPath(self.root)

        articulation_root = UsdPhysics.ArticulationRootAPI.Apply(prim)

        physx_articulation = PhysxSchema.PhysxArticulationAPI.Apply(
            articulation_root.GetPrim())

        physx_articulation.CreateEnabledSelfCollisionsAttr().Set(False)

        self.stage.Save()

    def assembly_joint(self):
        joints_dict = self.urdf_dict["joints"]

        for joint_info in joints_dict.keys():
            joint_dict = joints_dict[joint_info]

            parent_path = self.root + '/' + joint_dict["parent"]
            child_path = self.root + '/' + joint_dict["child"]

            rel0, relrot0 = self.get_xyz_rpy(joint_dict["origin"]["xyz"],
                                             joint_dict["origin"]["rpy"])
            rel1, relrot1 = self.get_xyz_rpy(joint_dict["origin"]["xyz01"],
                                             joint_dict["origin"]["rpy01"])

            if joint_dict["type"] == "fixed":

                add_fixed(self.stage, self.root + f'/joints/' + joint_info,
                          parent_path, child_path, rel0, relrot0, rel1,
                          relrot1)
            elif joint_dict["type"] == "prismatic":

                axis = joint_dict["limit"]["axis"]

                add_prismatic(self.stage, self.root + f'/joints/' + joint_info,
                              parent_path, child_path, rel0, relrot0, [
                                  joint_dict["limit"]["lower"],
                                  joint_dict["limit"]["upper"]
                              ], axis, rel1, relrot1)
            elif joint_dict["type"] == "revolute":
                axis = joint_dict["limit"]["axis"]

                add_revolute(self.stage, self.root + f'/joints/' + joint_info,
                             parent_path, child_path, rel0, relrot0, axis, [
                                 joint_dict["limit"]["lower"],
                                 joint_dict["limit"]["upper"]
                             ], rel1, relrot1)

    def get_xyz_rpy(self, xyz_string, rpy_string, turn_off_quat=False):

        xyz_array = np.array([float(value) for value in xyz_string.split()])

        if rpy_string is None:
            rpy_array = np.array([0.0, 0.0, 0.0])
        else:
            rpy_array = np.array(
                [float(value) for value in rpy_string.split()])
        if turn_off_quat:
            return xyz_array, rpy_array
        rotation = transformations.quaternion_from_euler(rpy_array[0],
                                                         rpy_array[1],
                                                         rpy_array[2],
                                                         axes='sxyz')
        return xyz_array, rotation

    def assembly_meshes(self, link_info, target_meshes, mesh_type="visuals"):

        for index, target_mesh in enumerate(target_meshes):
            if target_mesh["name"] is not None:
                prim_mesh = UsdGeom.Mesh.Define(
                    self.stage,
                    self.root + '/' + link_info + f"/{mesh_type}/" +
                    target_mesh["name"].replace("-", "_") + "_" + str(index))
            else:
                prim_mesh = UsdGeom.Mesh.Define(
                    self.stage, self.root + '/' + link_info +
                    f"/{mesh_type}/mesh_" + str(index))

            mesh_path = os.path.join(self.ckpt_dir, target_mesh["filename"])

            if mesh_path.endswith(".dae"):
                # Load the mesh from DAE file

                raw_mesh = trimesh.load(
                    mesh_path.replace(".dae",
                                      ".stl").replace("visual", "collision"))

            else:
                raw_mesh = trimesh.load(mesh_path)

            if isinstance(raw_mesh, trimesh.Scene):
                # Extract all meshes in the scene
                meshes = [geom for geom in raw_mesh.geometry.values()]

                # Merge all meshes into a single mesh
                combined_mesh = trimesh.util.concatenate(meshes)

                # Access points and faces
                points = combined_mesh.vertices
                faces = combined_mesh.faces
            else:
                points = raw_mesh.vertices
                faces = raw_mesh.faces

            vertex_counts = np.ones_like(faces[:, 0]).reshape(-1).astype(
                np.int32) * 3

            prim_mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
            prim_mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces))
            prim_mesh.CreateFaceVertexCountsAttr(
                Vt.IntArray.FromNumpy(vertex_counts))

            xyz_array, rotation = self.get_xyz_rpy(
                target_mesh["origin"]["xyz"], target_mesh["origin"]["rpy"])

            # AddRotate(prim_mesh, rotation)
            # AddTranslate(prim_mesh, (xyz_array[0], xyz_array[1], xyz_array[2]))

            if mesh_type == "collisions":

                parent_path = self.root + '/' + link_info
                parent_prim = self.stage.GetPrimAtPath(parent_path)

                # mesh_prim = self.stage.GetPrimAtPath(parent_prim)

                create_rigid_collision(parent_prim)
            # elif mesh_type == "visuals":

            #     mtl_path = os.path.join(
            #         self.ckpt_dir,
            #         target_mesh["filename"].replace(".obj", ".mtl"))
            #     if os.path.exists(mtl_path):
            #         materials = parse_mtl(mtl_path)
            #         texture_dir = os.path.dirname(mtl_path)

            #         material_path = str(prim_mesh.GetPath()) + f"/materials"

            #         # material_path = self.root + '/' + link_info + f"/{mesh_type}/materials"
            #         for material_name, material_data in materials.items():
            #             usd_material = create_usd_material(
            #                 self.stage, material_path, material_name,
            #                 material_data, texture_dir)

            #         prim_mesh.GetPrim().ApplyAPI(UsdShade.MaterialBindingAPI)
            #         UsdShade.MaterialBindingAPI(prim_mesh).Bind(usd_material)

    def assembly_link(self):

        links_dict = self.urdf_dict["links"]
        for link_info in links_dict.keys():
            link_dict = links_dict[link_info]
            if len(link_dict["visual_meshes"]) > 0:

                xyz_array, rotation = self.get_xyz_rpy(
                    link_dict["visual_meshes"][0]["origin"]["xyz"],
                    link_dict["visual_meshes"][0]["origin"]["rpy"])

                create_xform(
                    self.stage, self.scale,
                    (xyz_array[0], xyz_array[1], xyz_array[2]),
                    (rotation[0], rotation[1], rotation[2], rotation[3]),
                    self.root + '/' + link_info)
                visual_meshes = link_dict["visual_meshes"]

                collision_meshes = link_dict["collision_meshes"]

                if len(visual_meshes) == 0:
                    continue

                self.assembly_meshes(link_info,
                                     visual_meshes,
                                     mesh_type="visuals")
                if len(collision_meshes) > 0:

                    self.assembly_meshes(link_info,
                                         collision_meshes,
                                         mesh_type="collisions")

            else:
                create_xform(self.stage, None, (0.0, 0.0, 0.0),
                             (1., 0., 0., 0.), self.root + '/' + link_info)


# Example usage
urdf_file_path = "/home/ensu/Documents/weird/franka_panda_description/robots/robot.urdf"

# Parse URDF file into a dictionary
urdf_dict = parse_urdf_with_origin_and_collision(urdf_file_path)

# Load the USD file
stage = Usd.Stage.Open(
    "/home/ensu/Documents/weird/franka_panda_description/robots/panda_arm/robot.usd"
)

# Print all the prims (objects) in the USD file
for prim in stage.Traverse():
    path_name = prim.GetPath()
    if "joint" in path_name.pathString:
        for child_joint in prim.GetChildren():

            joint = UsdPhysics.Joint(child_joint)
            local0 = joint.GetLocalPos0Attr().Get()
            rot0 = joint.GetLocalRot0Attr().Get()
            local1 = joint.GetLocalPos1Attr().Get()
            rot1 = joint.GetLocalRot1Attr().Get()

            joint_name = child_joint.GetPath().pathString.split("/")[-1]
            if joint_name not in urdf_dict["joints"].keys():
                continue
            urdf_dict["joints"][joint_name]["origin"]["xyz"] = " ".join(
                map(str, local0))
            urdf_dict["joints"][joint_name]["origin"]["xyz01"] = " ".join(
                map(str, local1))

            quat_values0 = [rot0.GetReal()] + list(rot0.GetImaginary())
            quat0 = (torch.cat(
                math_utils.euler_xyz_from_quat(torch.as_tensor([quat_values0
                                                                ])))).tolist()
            urdf_dict["joints"][joint_name]["origin"]["rpy"] = " ".join(
                map(str, quat0))

            quat_values1 = [rot1.GetReal()] + list(rot1.GetImaginary())
            quat1 = (torch.cat(
                math_utils.euler_xyz_from_quat(torch.as_tensor([quat_values1
                                                                ])))).tolist()
            urdf_dict["joints"][joint_name]["origin"]["rpy01"] = " ".join(
                map(str, quat1))

    elif "panda_" in path_name.pathString and "collisions" not in path_name.pathString and "visuals" not in path_name.pathString:
        link_name = prim.GetPath().pathString.split("/")[-1]
        if link_name not in urdf_dict["links"].keys():
            continue
        w = prim.GetAttribute("xformOp:orient").Get().GetReal()
        x, y, z = prim.GetAttribute("xformOp:orient").Get().GetImaginary()

        pos = prim.GetAttribute("xformOp:translate").Get()

        orientation = [w, x, y, z]
        quat = (torch.cat(
            math_utils.euler_xyz_from_quat(torch.as_tensor([orientation
                                                            ])))).tolist()

        # if link_name == "panda_link8":
        #     continue
        for index, mesh in enumerate(
                urdf_dict["links"][link_name]["visual_meshes"]):
            urdf_dict["links"][link_name]["visual_meshes"][index]["origin"][
                "xyz"] = " ".join(map(str, pos))

            urdf_dict["links"][link_name]["visual_meshes"][index]["origin"][
                "rpy"] = " ".join(map(str, quat))
        for index, mesh in enumerate(
                urdf_dict["links"][link_name]["collision_meshes"]):
            urdf_dict["links"][link_name]["collision_meshes"][index]["origin"][
                "xyz"] = " ".join(map(str, pos))
            urdf_dict["links"][link_name]["collision_meshes"][index]["origin"][
                "rpy"] = " ".join(map(str, quat))

assembly = AssembleArticulation(
    "/home/ensu/Documents/weird/franka_panda_description/",
    usda_name='modified_mobility.usda',
    urdf_dict=urdf_dict)
assembly.convert()
