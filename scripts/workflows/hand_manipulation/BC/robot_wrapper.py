from typing import List

import numpy as np
import numpy.typing as npt
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshio
import os

import trimesh
import pymeshlab
import isaaclab.utils.math as math_utils

import torch


class RobotWrapper:
    """
    This class does not take mimic joint into consideration
    """

    def __init__(self,
                 urdf_path: str,
                 mesh_synthesis: bool = True,
                 viz_robot: bool = False,
                 num_downsample_points=512,
                 ee_frame_name: str = "panda_link7"):

        # Create robot model and data

        self.pin_model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path)

        self.pin_data: pin.Data = self.pin_model.createData()
        self.ee_frame_name = ee_frame_name
        self.viz_robot = viz_robot
        self.mesh_synthesis = mesh_synthesis
        self.init_settings()
        self.num_downsample_points = num_downsample_points

    def init_settings(self):
        """ Initialize settings for the robot wrapper.
        """

        self.q0 = pin.neutral(self.pin_model)
        if self.pin_model.nv != self.pin_model.nq:
            raise NotImplementedError(
                "Can not handle robot with special joint.")

        self.frame_id_mapping: Dict[str, int] = {}
        for i, frame in enumerate(self.pin_model.frames):
            self.frame_id_mapping[frame.name] = i

        self.ee_frame_id = self.frame_id_mapping[self.ee_frame_name]
        self.visual_data = pin.GeometryData(self.visual_model)

        if self.viz_robot:
            self.visualizer = MeshcatVisualizer(self.pin_model,
                                                self.collision_model,
                                                self.visual_model)
            self.visualizer.initViewer(open=True)
            self.visualizer.loadViewerModel()

        if self.mesh_synthesis:

            self.init_meshcat_visualizer()

    def init_meshcat_visualizer(self, ):
        self.mesh_dict = {}

        for geom_obj, placement in zip(self.visual_model.geometryObjects,
                                       self.visual_data.oMg):
            print(f"Loading mesh for {geom_obj.name}")
            mesh_path = geom_obj.meshPath

            mesh_ext = os.path.splitext(mesh_path)[-1].lower()

            mesh = trimesh.util.concatenate(trimesh.load(mesh_path))
            vertices = mesh.vertices  # (N, 3)

            mesh = pymeshlab.Mesh(vertex_matrix=vertices,
                                  face_matrix=np.array(mesh.faces,
                                                       dtype=np.int32))

            # Create a MeshSet and add the mesh to it
            ms = pymeshlab.MeshSet()
            ms.add_mesh(mesh, 'my_mesh')
            ms.meshing_remove_duplicate_faces()
            ms.meshing_repair_non_manifold_edges()
            # ms.meshing_repair_non_manifold_vertices()
            ms.meshing_surface_subdivision_midpoint(iterations=2)
            current_mesh = ms.current_mesh()
            vertices = current_mesh.vertex_matrix()

            vertices = math_utils.fps_points(
                torch.tensor(vertices).unsqueeze(0),
                self.num_downsample_points
                if hasattr(self, 'num_downsample_points') else 512,
            ).cpu().numpy().squeeze(0)
            # print(vertices.shape[0], num_points)

            self.mesh_dict[geom_obj.name] = vertices

    def extract_meshcat_visualizer(self, ):
        # Extract transformed mesh vertices
        all_mesh_vertices = []
        pin.updateGeometryPlacements(self.pin_model, self.pin_data,
                                     self.visual_model, self.visual_data)
        for geom_obj, placement in zip(self.visual_model.geometryObjects,
                                       self.visual_data.oMg):

            vertices = self.mesh_dict[geom_obj.name]  # (N, 3)

            # Apply transformation
            R = placement.rotation
            t = placement.translation
            transformed_vertices = (R @ vertices.T).T + t.T
            all_mesh_vertices.append(transformed_vertices)
        return np.concatenate(all_mesh_vertices, axis=0)

    # -------------------------------------------------------------------------- #
    # Robot property
    # -------------------------------------------------------------------------- #
    @property
    def joint_names(self) -> List[str]:
        return list(self.pin_model.names)

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.pin_model.nqs
        return [
            name for i, name in enumerate(self.pin_model.names) if nqs[i] > 0
        ]

    @property
    def dof(self) -> int:
        return self.pin_model.nq

    @property
    def link_names(self) -> List[str]:
        link_names = []
        for i, frame in enumerate(self.pin_model.frames):
            link_names.append(frame.name)
        return link_names

    @property
    def joint_limits(self):
        lower = self.pin_model.lowerPositionLimit
        upper = self.pin_model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    # -------------------------------------------------------------------------- #
    # Query function
    # -------------------------------------------------------------------------- #
    def get_joint_index(self, name: str):
        return self.dof_joint_names.index(name)

    def get_link_index(self, name: str):
        if name not in self.link_names:
            raise ValueError(
                f"{name} is not a link name. Valid link names: \n{self.link_names}"
            )
        return self.pin_model.getFrameId(name, pin.BODY)

    def get_joint_parent_child_frames(self, joint_name: str):
        joint_id = self.pin_model.getFrameId(joint_name)
        parent_id = self.pin_model.frames[joint_id].parent
        child_id = -1
        for idx, frame in enumerate(self.pin_model.frames):
            if frame.previousFrame == joint_id:
                child_id = idx
        if child_id == -1:
            raise ValueError(f"Can not find child link of {joint_name}")

        return parent_id, child_id

    # -------------------------------------------------------------------------- #
    # Kinematics function
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        pin.forwardKinematics(self.pin_model, self.pin_data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.pin_model, self.pin_data,
                                                 link_id)
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.pin_model, self.pin_data,
                                                 link_id)
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(self, qpos,
                                           link_id: int) -> npt.NDArray:
        J = pin.computeFrameJacobian(self.pin_model, self.pin_data, qpos,
                                     link_id)
        return J

    def get_sim_joints_names(self):

        self.sim_joints_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7', 'j1', 'j0', 'j2',
            'j3', 'j12', 'j13', 'j14', 'j15', 'j5', 'j4', 'j6', 'j7', 'j9',
            'j8', 'j10', 'j11'
        ]

    def get_real_joints_names(self):

        self.real_joints_names = [
            'panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4',
            'panda_joint5', 'panda_joint6', 'panda_joint7', 'j1', 'j12', 'j5',
            'j9', 'j0', 'j13', 'j4', 'j8', 'j2', 'j14', 'j6', 'j10', 'j3',
            'j15', 'j7', 'j11'
        ]

    def sim2real_joint_mapping(self):
        self.get_sim_joints_names()
        self.get_real_joints_names()
        self.sim2real_index = [
            self.sim_joints_names.index(name)
            for name in self.real_joints_names
        ]
        self.real2sim_index = [
            self.real_joints_names.index(name)
            for name in self.sim_joints_names
        ]

    def compute_ik(self, ee_pose: pin.SE3, init_qpos):
        oMdes = ee_pose
        qpos = init_qpos

        if qpos.shape[0] == 7:
            qpos = np.concatenate([qpos, np.zeros(16)])

        for k in range(100):
            pin.forwardKinematics(self.pin_model, self.pin_data, qpos)
            ee_pose = pin.updateFramePlacement(self.pin_model, self.pin_data,
                                               self.ee_frame_id)
            J = pin.computeFrameJacobian(self.pin_model, self.pin_data, qpos,
                                         self.ee_frame_id)
            iMd = ee_pose.actInv(oMdes)
            err = pin.log(iMd).vector
            if np.linalg.norm(err) < 1e-3:
                # print(k, np.linalg.norm(err))
                break

            v = J.T.dot(np.linalg.solve(J.dot(J.T) + 1e-5, err))
            qpos = pin.integrate(self.pin_model, qpos, v * 0.05)
        return qpos

    def compute_ee_pose(self, qpos) -> pin.SE3:
        if qpos.shape[0] == 7:
            qpos = np.concatenate([qpos, np.zeros(10)])
        pin.forwardKinematics(self.pin_model, self.pin_data, np.array(qpos))
        ee_pose: pin.SE3 = pin.updateFramePlacement(self.pin_model,
                                                    self.pin_data,
                                                    self.ee_frame_id)
        return ee_pose

    def step(self, qpos: npt.NDArray):
        """
        Update robot kinematics and show in Meshcat.
        """
        # if not self.viz_robot:
        #     return

        # # Forward kinematics
        # self.compute_forward_kinematics(qpos)

        # # # Update visual geometry placements
        # pin.updateGeometryPlacements(self.pin_model, self.pin_data,
        #                              self.visual_model, self.visual_data)

        # Display in Meshcat
        self.visualizer.display(qpos)


if __name__ == "__main__":
    # Example usage
    robot_wrapper = RobotWrapper(
        "source/assets/robot/franka/urdf/franka_description/robots/panda_arm_hand.urdf",
        viz_robot=True,
        mesh_synthesis=False,
    )

    while True:
        q = pin.neutral(robot_wrapper.pin_model)
        robot_wrapper.step(q)
