# CuRobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    get_world_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel, CudaRobotModelConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from sympy import true
import torch
import numpy as np
from curobo.geom.types import Cuboid, WorldConfig

try:
    from curobo.util.usd_helper import UsdHelper
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
except:
    print(
        "CuRobo USD Helper not available. Ensure you have the required dependencies installed."
    )

from typing import Dict, List, Optional, Union

from curobo.geom.types import (
    Mesh,
    WorldConfig,
)

# import isaacsim.core.utils.prims as prim_utils
from tools.curobo_ik_planner import IKPlanner
import omni
try:
    from isaaclab.assets import RigidObject, Articulation
    import isaaclab.utils.math as math_utils
except:
    pass
import pymeshlab


def get_prim_world_pose(cache, prim, inverse):
    world_transform: Gf.Matrix4d = cache.GetLocalToWorldTransform(prim)
    # get scale:
    scale: Gf.Vec3d = Gf.Vec3d(
        *(v.GetLength() for v in world_transform.ExtractRotationMatrix()))
    scale = list(scale)
    t_mat = world_transform.RemoveScaleShear()
    if inverse:
        t_mat = t_mat.GetInverse()

    translation: Gf.Vec3d = t_mat.ExtractTranslation()
    rotation: Gf.Rotation = t_mat.ExtractRotation()
    q = rotation.GetQuaternion()
    orientation = [q.GetReal()] + list(q.GetImaginary())
    t_mat = (Pose.from_list(
        list(translation) + orientation,
        TensorDeviceType()).get_matrix().view(4, 4).cpu().numpy())

    return t_mat, scale


def get_mesh_attrs(
    prim,
    cache=None,
    apply_trasform=False,
) -> Mesh:
    # read cube information
    # scale = prim.GetAttribute("size").Get()
    try:
        points = list(prim.GetAttribute("points").Get())
        points = [np.ravel(x) for x in points]

        faces = list(prim.GetAttribute("faceVertexIndices").Get())

        face_count = list(prim.GetAttribute("faceVertexCounts").Get())

        faces = np.array(faces).reshape(-1, 3)
        if prim.GetAttribute("xformOp:scale").IsValid():
            scale = list(prim.GetAttribute("xformOp:scale").Get())
        else:
            scale = [1.0, 1.0, 1.0]
        size = prim.GetAttribute("size").Get()
        if size is None:
            size = 1
        scale = [s * size for s in scale]

        points = np.array(points)

        if apply_trasform:  # kitchen can be transformed since unmoved in the scene

            mat, scale = get_prim_world_pose(cache, prim)

            # # compute position and orientation on cuda:
            # tensor_mat = torch.as_tensor(mat)

            # tensor_mat[:3, :3] *= torch.as_tensor(scale)
            # transformed_points = (tensor_mat[:3, :3] @ points.T).T + tensor_mat[:3,
            #                                                                     3]
            w = prim.GetAttribute("xformOp:orient").Get().GetReal()
            x, y, z = prim.GetAttribute("xformOp:orient").Get().GetImaginary()

            pos = torch.as_tensor(prim.GetAttribute("xformOp:translate").Get())

            orientation = torch.as_tensor([w, x, y, z])
            transformed_points = math_utils.transform_points(
                torch.as_tensor(points) * scale[0], pos, orientation)

        else:
            mat, scale = get_prim_world_pose(cache, prim)
            transformed_points = points * scale

        mesh = pymeshlab.Mesh(vertex_matrix=transformed_points,
                              face_matrix=np.array(faces, dtype=np.int32))
        ms = pymeshlab.MeshSet()
        ms.add_mesh(mesh, 'my_mesh')
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=1000)
        current_mesh = ms.current_mesh()
        vertices = current_mesh.vertex_matrix()
        faces = current_mesh.face_matrix()

        return [
            str(prim.GetPath()),
            transformed_points.astype(np.float32), faces, scale
        ]
    except:

        return None


class MotionPlanner:

    def __init__(
        self,
        env=None,
        robot_file="franka.yml",
        world_file="collision_table.yml",
        robot_name="robot",
        collision_checker=False,
        only_paths=None,
        reference_prim_path=None,
        ignore_substring=None,
    ):
        # init sim env
        self.env = env
        self.robot_name = robot_name

        self.only_paths = only_paths
        self.reference_prim_path = reference_prim_path
        self.ignore_substring = ignore_substring

        # self.device = device
        self.tensor_args = TensorDeviceType()

        # # mod this later
        n_obstacle_cuboids = 10
        n_obstacle_mesh = 30

        robot_file = join_path(get_robot_configs_path(), robot_file)
        world_file = join_path(get_world_configs_path(), world_file)

        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            world_file,
            collision_checker_type=CollisionCheckerType.MESH,
            collision_cache={
                "obb": n_obstacle_cuboids,
                "mesh": n_obstacle_mesh
            },
            interpolation_dt=1 / 10)
        self.motion_gen = MotionGen(motion_gen_config)
        self.motion_gen.warmup(enable_graph=True)

        robot_cfg = load_yaml(join_path(get_robot_configs_path(),
                                        robot_file))["robot_cfg"]
        robot_cfg = RobotConfig.from_dict(robot_cfg, self.tensor_args)
        self.kin_model = CudaRobotModel(robot_cfg.kinematics)

        self.device = "cuda"
        # init collision checker
        self.collision_checker = collision_checker
        if self.collision_checker:
            self.init_collision_mesh()

    def plan_batch_motion(self,
                          qpos,
                          target_position,
                          target_quat,
                          return_ee_pose=True,
                          jpos_dim=7):

        start = JointState.from_position(qpos[..., :jpos_dim])

        goal = Pose(target_position, target_quat)

        result = self.motion_gen.plan_batch(start,
                                            goal,
                                            plan_config=MotionGenPlanConfig(
                                                enable_graph=False,
                                                max_attempts=2,
                                                enable_finetune_trajopt=True))

    def plan_motion(
        self,
        qpos,
        target_position,
        target_quat,
        return_ee_pose=True,
        jpos_dim=7,
    ):

        start = JointState.from_position(qpos[..., :jpos_dim])

        goal = Pose(target_position, target_quat)

        result = self.motion_gen.plan_single(
            start, goal, MotionGenPlanConfig(max_attempts=2))

        if result.success.item():
            traj = result.get_interpolated_plan()

        else:

            traj = None

        if traj is not None:
            print(
                f"Trajectory Generated: success {result.success.item()} | len {len(traj)} | optimized_dt {result.optimized_dt.item()}"
            )

        # replace joint position with ee pose
        ee_pose = None

        if return_ee_pose and traj is not None:
            ee_pose = self.kin_model.get_state(traj.position)

        return ee_pose, traj

    def update_world(self, cuboids):

        self.world = WorldConfig(cuboid=cuboids)

        self.motion_gen.update_world(self.world)

    def attach_obstacle(self, name, qpos):
        obstacle = name

        start_state = JointState.from_position(
            torch.as_tensor(qpos, dtype=torch.float32, device=self.device))
        self.motion_gen.attach_objects_to_robot(start_state, [obstacle])

    def detach_obstacle(self) -> None:
        self.motion_gen.detach_object_from_robot()

    def add_obstacle(self,
                     plan_grasp=False,
                     target_object_name=None,
                     target_collision_checker_name=[]) -> None:

        obstacles = {}
        obstacles["mesh"] = []
        robot = self.env.scene[self.robot_name]
        robot_root_state = robot._data.root_state_w[0, :7]
        if target_collision_checker_name is None:
            target_collision_checker_name = self.obstacles_mesh.keys()

        for key in target_collision_checker_name:
            if key not in self.obstacles_mesh.keys():
                continue

            meta_data = self.obstacles_mesh[key]
            prim_path = meta_data[0]

            prim_name = prim_path.split("/")[4]

            if plan_grasp:
                if target_object_name == prim_name:
                    continue
            if isinstance(self.env.scene[prim_name], RigidObject):
                object = self.env.scene[prim_name]

                object_root_state = object._data.root_state_w[0, :7]
            elif isinstance(self.env.scene[prim_name], Articulation):
                articulated_object = self.env.scene[prim_name]

                id, _ = articulated_object.find_bodies(prim_path.split("/")[5])
                object_root_state = articulated_object._data.body_state_w[
                    0, id[0], :7]
            else:
                continue

            # elif "collision" in prim_path:

            #     sub_prim = self.env.scene.stage.GetPrimAtPath(prim_path)

            #     # true_prim_name = '_'.join(prim_name.lower().split('_')[:2])
            #     # rigid_collections = self.env.scene[true_prim_name]
            #     # rigid_bodies_id, _ = rigid_collections.find_objects(prim_name)
            #     # object_root_state = rigid_collections._data.object_state_w[
            #     #     0, rigid_bodies_id[0], :7]
            #     world_transform = self.usd_help._xform_cache.GetLocalToWorldTransform(
            #         sub_prim)
            #     transform_matrix = torch.as_tensor(world_transform)
            #     transform_quat = math_utils.quat_from_matrix(
            #         transform_matrix[:3, :3])
            #     object_root_state = torch.cat(
            #         [transform_matrix[3, :3], transform_quat]).to(self.device)

            # else:

            #     continue

            robot2object_pos, robot2object_quat = math_utils.subtract_frame_transforms(
                robot_root_state[:3], robot_root_state[3:7],
                object_root_state[:3], object_root_state[3:7])

            m_data = Mesh(name=meta_data[0],
                          vertices=meta_data[1].tolist(),
                          faces=meta_data[2].tolist(),
                          pose=torch.cat([robot2object_pos,
                                          robot2object_quat]).tolist(),
                          scale=[1, 1, 1])
            obstacles["mesh"].append(m_data)

        world_model = WorldConfig(**obstacles)
        obstacles = world_model.get_collision_check_world()

        self.motion_gen.update_world(obstacles)

        # obstacles.save_world_as_mesh("obstacles.obj")

    def init_collision_mesh(
        self,
        timecode: float = 0,
    ):

        self.world = WorldConfig()
        self.motion_gen.update_world(self.world)
        self.curobo_ik = IKPlanner(self.env, device=self.device)

        # init usd helper
        self.usd_help = UsdHelper()
        self.usd_help.load_stage(self.env.scene.stage)
        self.usd_help.add_world_to_stage(self.world, base_frame="/World")

        self.usd_help._xform_cache.Clear()
        self.usd_help._xform_cache.SetTime(timecode)

        all_items = self.usd_help.stage.Traverse()

        self.obstacles_mesh = {}

        obstacles = {}
        obstacles["mesh"] = []

        for x in all_items:
            if self.robot_name in x.GetPath().pathString.split("/"):
                continue

            if self.only_paths is not None:
                if not any(
                    [str(x.GetPath()).startswith(k) for k in self.only_paths]):
                    continue

            if self.ignore_substring is not None:
                if any([k in str(x.GetPath()) for k in self.ignore_substring]):
                    continue

            if x.IsA(UsdGeom.Mesh) and "collision" not in str(
                    x.GetPath().pathString) and "visuals" not in str(
                        x.GetPath().pathString):

                m_data = get_mesh_attrs(
                    x,
                    cache=self.usd_help._xform_cache,
                )

                if m_data is not None:
                    self.obstacles_mesh[x.GetPath().pathString.split("/")
                                        [4]] = m_data
            elif "collision" in str(x.GetPath().pathString):
                sub_mesh = x.GetChildren()

                if sub_mesh is None or len(sub_mesh) == 0:

                    if x.IsA(UsdGeom.Mesh):
                        m_data = get_mesh_attrs(
                            x,
                            cache=self.usd_help._xform_cache,
                            apply_trasform=True,
                        )

                        self.obstacles_mesh[x.GetPath().pathString.split("/")
                                            [-2]] = m_data

                else:
                    points_buffer = []
                    faces_buffer = []
                    faces_count = 0
                    for mesh in sub_mesh:
                        if mesh.IsA(UsdGeom.Mesh):

                            m_data = get_mesh_attrs(
                                mesh,
                                cache=self.usd_help._xform_cache,
                                apply_trasform=True,
                            )
                            if m_data is not None:
                                points_buffer.append(m_data[1])
                                faces_buffer.append(m_data[2] + faces_count)
                                faces_count += len(m_data[2])
                                name = m_data[0]
                                scale = m_data[3]

                    if len(points_buffer) > 0:
                        final_m_data = [
                            name,
                            np.concatenate(points_buffer),
                            np.concatenate(faces_buffer), scale
                        ]

                        self.obstacles_mesh[x.GetPath().pathString.split("/")
                                            [-2]] = final_m_data
            else:
                subchildren = x.GetAllChildren()
                for child in subchildren:
                    if child.IsA(UsdGeom.Mesh):
                        m_data = get_mesh_attrs(
                            child,
                            cache=self.usd_help._xform_cache,
                            apply_trasform=True,
                        )

                        if m_data is not None:
                            self.obstacles_mesh[child.GetPath().pathString.
                                                split("/")[-3]] = m_data

        try:

            for name in self.obstacles_mesh.keys():
                if "visuals" in name:
                    self.obstacles_mesh.pop(name, None)
            self.obstacles_mesh.pop("collisions", None)
            self.obstacles_mesh.pop("visuals", None)

        except:
            pass

    def clear_obstacles(self):
        # self.motion_gen.clear_world_cache()
        self.motion_gen.update_world(self.world)

    def remove_obstacle(self, str):
        self.world.remove_obstacle(str)
        #self.motion_gen.update_world(self.world)
