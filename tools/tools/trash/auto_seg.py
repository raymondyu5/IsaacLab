import argparse
import open3d as o3d
from isaaclab.app import AppLauncher


def scalar_last(quat):
    return np.array([quat[1], quat[2], quat[3], quat[0]])


# add argparse arguments
parser = argparse.ArgumentParser(
    description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot",
                    type=str,
                    default="franka",
                    help="Name of the robot to load")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import torch
import numpy as np

from pxr import Usd, UsdGeom, Sdf, Gf
from scipy.spatial.transform import Rotation as R
from pathlib import Path

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
import omni.usd
import isaaclab.utils.math as math

from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
import isaaclab.utils.math as math_utils
from isaacsim.core.prims import RigidPrim, Articulation
from scripts.workflows.utils.gs_env.gs_utils.gaussian_model import GaussianModel
from tools.curobo_ik_planner import IKPlanner

sim_cfg = sim_utils.SimulationCfg(
    dt=0.01,
    device="cuda",
)
sim = sim_utils.SimulationContext(sim_cfg)

cfg = sim_utils.UsdFileCfg(
    usd_path=
    f"/home/ensu/Documents/weird/IsaacLab_assets/assets/robot/droid_uw_robot/robot.usdz"
)
cfg.func("/World/test", cfg)


def get_robot_mesh_paths(stage):
    visited = set()
    meshes = set()

    def traverse(prim):
        if prim.IsValid():
            if UsdGeom.Mesh(prim) and "collision" not in prim.GetName():
                meshes.add(str(prim.GetPath()))

            for child in prim.GetChildren():
                traverse(child)

    traverse(stage.GetPseudoRoot())
    return meshes


# Example usage
stage = omni.usd.get_context().get_stage()
path_list = get_robot_mesh_paths(stage)

sim.reset()
sim.step()

planner = IKPlanner()
robot = Articulation("/World/test/panda")
past_joint_pos = robot.get_joint_positions()

ee_position = planner.ik_solver.fk(past_joint_pos[:, :7])
cur_ee_position = ee_position.ee_position
cur_ee_quat = ee_position.ee_quaternion
cur_ee_position[:, 0] += 0.010
cur_ee_position[:, 1] += 0.015
cur_ee_position[:, 2] -= 0.01
cur_joint_pos = planner.plan_motion(cur_ee_position, cur_ee_quat)

past_joint_pos[:, :7] = cur_joint_pos[0, :, :7]

robot.set_joint_positions(past_joint_pos)
sim.step()

base_path = "/World/test/"
transforms = {}
robot_ee_transform = {}
for path in path_list:

    if "Robotiq_2F_85" not in path:
        filtered_paths = "/".join(path.split("/")[:5])

        view = RigidPrim(filtered_paths)

        pos, rot = view.get_world_poses()
        robot_ee_transform[path] = (pos, rot)
    else:

        prim = stage.GetPrimAtPath(path)
        prim_path = "/".join(path.split("/")[:8])
        prim = stage.GetPrimAtPath(prim_path)
        mesh_pose = prim.GetAttribute("xformOp:translate").Get()
        mesh_orientation = prim.GetAttribute("xformOp:orient").Get()
        mesh_pose_tensor = torch.tensor(
            [[mesh_pose[0], mesh_pose[1], mesh_pose[2]]], dtype=torch.float32)

        # Convert mesh_orientation (Quatd) to tensor
        mesh_orientation_tensor = torch.tensor(
            [[mesh_orientation.GetReal(), *mesh_orientation.GetImaginary()]],
            dtype=torch.float32)

        filtered_paths = "/".join(path.split("/")[:7])
        view = RigidPrim(filtered_paths)

        base_pos, base_rot = view.get_world_poses()
        pos, rot = math_utils.combine_frame_transforms(
            base_pos.cpu().to(torch.float32),
            base_rot.cpu().to(torch.float32), mesh_pose_tensor,
            mesh_orientation_tensor)
        robot_ee_transform[path] = (base_pos, base_rot)

    transforms[path] = (pos, rot)


def get_meshes(stage, transforms):
    meshes = {}
    meshes_long = []
    for path in transforms:
        prim = stage.GetPrimAtPath(path)
        if prim:
            mesh = UsdGeom.Mesh(prim)
            if mesh:
                points = mesh.GetPointsAttr().Get()
                face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()
                faces = np.array(face_vertex_indices).reshape(-1, 3)
                raw_vertices = np.array(points).reshape(-1, 3)

                pos, rot = transforms[path]
                pos, rot = pos.detach().cpu().numpy(), rot.detach().cpu(
                ).numpy()
                pos, rot = pos.squeeze(), rot.squeeze()

                robot_ee_transform[path]
                rot = R.from_quat(np.roll(rot, -1)).as_matrix()

                vertices = (rot @ raw_vertices.T).T + pos

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(vertices)
                mesh.triangles = o3d.utility.Vector3iVector(faces)
                meshes[path] = (mesh, pos, rot)
                meshes_long.append(mesh)

    # merged_mesh = o3d.geometry.TriangleMesh()

    # # Keep track of vertex offset for updating triangle indices
    # vertex_offset = 0

    # for raw_mesh in meshes_long:
    #     # Convert to numpy arrays
    #     vertices = np.asarray(raw_mesh.vertices)
    #     triangles = np.asarray(raw_mesh.triangles)

    #     # Update triangle indices based on the offset
    #     triangles += vertex_offset

    #     # Append to merged mesh
    #     merged_mesh.vertices.extend(o3d.utility.Vector3dVector(vertices))
    #     merged_mesh.triangles.extend(o3d.utility.Vector3iVector(triangles))

    #     # Update offset
    #     vertex_offset += len(vertices)

    #     # Export the merged mesh
    #     o3d.io.write_triangle_mesh("merged_mesh.ply", merged_mesh)
    # print("Merged mesh exported as 'merged_mesh.ply'")
    # import pdb
    # pdb.set_trace()

    return meshes


# Convert HSV to RGB (0-1 range) using numpy and a standard HSV to RGB conversion
def hsv_to_rgb(hsv):
    h, s, v = hsv
    i = int(np.floor(h * 6))  # Determine the color sector (0-5)
    f = h * 6 - i  # Fractional part of hue
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    i = i % 6
    if i == 0:
        return [v, t, p]
    elif i == 1:
        return [q, v, p]
    elif i == 2:
        return [p, v, t]
    elif i == 3:
        return [p, q, v]
    elif i == 4:
        return [t, p, v]
    elif i == 5:
        return [v, p, q]


def generate_distinct_colors(n):
    # Generate 'n' evenly spaced hues
    hues = np.linspace(0, 1, n,
                       endpoint=False)  # evenly space hues from 0 to 1
    colors = np.array([[(hue, 1, 1)]
                       for hue in hues])  # Set saturation and value to 1
    colors_rgb = np.array([hsv_to_rgb(hsv[0]) for hsv in colors])
    return colors_rgb


def segment_pcd(o3d_meshes, pcd):
    scenes = []
    for mesh in o3d_meshes:
        scene = o3d.t.geometry.RaycastingScene()
        mesh_ = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene.add_triangles(mesh_)
        scenes.append(scene)

    assignment = -1 * np.ones(len(pcd))
    distances = np.zeros((len(pcd), len(scenes)))
    for i, scene in enumerate(scenes):
        is_inside = scene.compute_occupancy(
            o3d.core.Tensor(pcd[:, :3], dtype=o3d.core.Dtype.Float32)).numpy()
        dist = scene.compute_distance(
            o3d.core.Tensor(pcd[:, :3], dtype=o3d.core.Dtype.Float32)).numpy()
        distances[:, i] = dist
        assignment[is_inside == 1] = i

    assignment[assignment == -1] = np.argmin(distances[assignment == -1],
                                             axis=1)
    assignment = assignment.astype(int)

    n = np.max(assignment) + 1

    colors_options = generate_distinct_colors(n)

    #construct colors
    colors = []
    for i, a in enumerate(assignment):
        colors.append(colors_options[a])
    colors = np.array(colors)

    segmented_pcd = o3d.geometry.PointCloud()
    segmented_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
    segmented_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([segmented_pcd] + o3d_meshes)

    return assignment


def combine_gaussian_models(new_model, old_mdel):

    new_model._xyz = torch.cat([old_mdel._xyz, new_model._xyz], dim=0)
    new_model._features_rest = torch.cat(
        [old_mdel._features_rest, new_model._features_rest], dim=0)
    new_model._scaling = torch.cat([old_mdel._scaling, new_model._scaling],
                                   dim=0)
    new_model._rotation = torch.cat([old_mdel._rotation, new_model._rotation],
                                    dim=0)
    new_model._features_dc = torch.cat(
        [old_mdel._features_dc, new_model._features_dc], dim=0)
    new_model._opacity = torch.cat([old_mdel._opacity, new_model._opacity],
                                   dim=0)
    return new_model


def split_model(model, assignments, meshes):
    path2model = {}

    for i, (mesh_path, (mesh, pos, rot)) in enumerate(meshes.items()):
        indices = np.where(assignments == i)[0]

        # mesh_parent = stage.GetPrimAtPath(mesh_path[:-1])
        # translate = mesh_parent.GetAttribute("xformOp:translate").Get()
        # translate = -pos
        # rotate = np.linalg.inv(rot)
        # if "Robotiq_2F_85" not in mesh_path:
        pose, quat = robot_ee_transform[mesh_path]
        fake_rot = math_utils.matrix_from_quat(quat)
        translate = -pose[0].cpu().numpy()

        rotate = np.linalg.inv(fake_rot[0].cpu().numpy())

        # rotate = R.from_matrix(rot).inv().as_matrix()
        # rotate = np.roll(rotate, 1)

        translate = torch.tensor(translate).cuda().float()
        rotate = torch.tensor(rotate).cuda().float()

        new_model = GaussianModel(3)
        # new_model._xyz = (rotate @ model._xyz[indices].T).T + translate
        new_model._xyz = (rotate @ (model._xyz[indices] + translate).T).T
        # new_model._xyz = model._xyz[indices] - translate
        new_model._features_rest = model._features_rest[indices]
        new_model._scaling = model._scaling[indices]

        # temp_rot = model._rotation[indices].detach().cpu().numpy()
        rotation_quat = math.quat_from_matrix(rotate)

        new_model._rotation = math.quat_mul(
            rotation_quat.repeat(len(model._rotation[indices]), 1),
            model._rotation[indices])

        new_model._features_dc = model._features_dc[indices]
        new_model._opacity = model._opacity[indices]
        if "Robotiq_2F_85" in mesh_path:

            mesh_name_path = "/".join(mesh_path.split("/")[:-2])
            if mesh_name_path in path2model.keys():
                combined_model = combine_gaussian_models(
                    new_model, path2model[mesh_name_path])

                path2model[mesh_name_path] = combined_model
            else:
                path2model[mesh_name_path] = new_model
        else:

            path2model[mesh_path] = new_model

    return path2model


meshes = get_meshes(stage, transforms)

model = GaussianModel(3)
# model.load_ply("./data/pi_robot/splat_real.ply")
model.load_ply(f"//home/ensu/Downloads/droid/splat_og.ply", eval=True)
pcd = model.get_xyz.detach().cpu().numpy()

assignments = segment_pcd([m[0] for m in meshes.values()], pcd)

path2model = split_model(model, assignments, meshes)
save_dir = Path(f"/home/ensu/Downloads/franka/SEGMENTED")
save_dir.mkdir(exist_ok=True)
with torch.no_grad():
    for raw_path, model in path2model.items():
        mesh_path = raw_path.split("/World/test/")[-1]
        mesh_path = mesh_path.replace("/", "-")

        if "Robotiq_2F_85" in raw_path:
            name = mesh_path.split("-")[3]
            # print(name)

            # pos, rot = robot_ee_transform[raw_path]

            # model._xyz = math_utils.transform_points(
            #     model._xyz + pos.to(model._xyz.device),
            #     pos.to(model._xyz.device) * 0.0, rot.to(model._xyz.device))

        else:

            name = mesh_path.split("-")[1]
            print("unchange name", name)

        model.save_ply(save_dir / f"{name}.ply")

# geom = o3d.geometry.PointCloud()
# geom.points = o3d.utility.Vector3dVector(pcd)
# o3d.visualization.draw_geometries([geom] + list(meshes.values()))
