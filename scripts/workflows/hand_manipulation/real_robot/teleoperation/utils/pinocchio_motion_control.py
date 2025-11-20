from threading import Lock
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pinocchio as pin
import yaml

import numpy.typing as npt


class PinocchioMotionControl:

    def __init__(
        self,
        yaml_path: str,
        viz_robot: bool = True,
    ):

        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)

        self.qpos = None
        self.dt = config["robot_cfg"]["dt"]
        self.ee_name = config["robot_cfg"]["kinematics"]["ee_link"]
        self._qpos_lock = Lock()

        if not viz_robot:
            self.model: pin.Model = pin.buildModelFromUrdf(
                str(config["robot_cfg"]["urdf_path"]))
        else:

            self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
                str(config["robot_cfg"]["urdf_path"]))
            self.visual_data = pin.GeometryData(self.visual_model)

        self.ik_damping = float(
            config["robot_cfg"]["kinematics"]["ik_damping"])
        self.ik_eps = float(config["robot_cfg"]["kinematics"]["eps"])
        self.data: pin.Data = self.model.createData()
        frame_mapping: Dict[str, int] = {}

        for i, frame in enumerate(self.model.frames):
            frame_mapping[frame.name] = i

        self.frame_mapping = frame_mapping
        self.ee_frame_id = frame_mapping[self.ee_name]

    def step(self,
             pos: Optional[np.ndarray],
             quat: Optional[np.ndarray],
             repeat=1):
        xyzw = np.array([quat[1], quat[2], quat[3], quat[0]])

        pose_vec = np.concatenate([pos, xyzw])
        oMdes = pin.XYZQUATToSE3(pose_vec)
        with self._qpos_lock:
            qpos = self.qpos.copy()

        for k in range(100 * repeat):
            pin.forwardKinematics(self.model, self.data, qpos)
            ee_pose = pin.updateFramePlacement(self.model, self.data,
                                               self.ee_frame_id)
            J = pin.computeFrameJacobian(self.model, self.data, qpos,
                                         self.ee_frame_id)
            iMd = ee_pose.actInv(oMdes)

            err = pin.log(iMd).vector
            if np.linalg.norm(err) < self.ik_eps:
                break

            v = J.T.dot(np.linalg.solve(J.dot(J.T) + self.ik_damping, err))
            qpos = pin.integrate(self.model, qpos, v * self.dt)

        # self.set_current_qpos(qpos)
        return qpos

    def compute_ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        pin.forwardKinematics(self.model, self.data, qpos)
        oMf: pin.SE3 = pin.updateFramePlacement(self.model, self.data,
                                                self.ee_frame_id)
        xyzw_pose = pin.SE3ToXYZQUAT(oMf)

        return np.concatenate([
            xyzw_pose[:3],
            np.array([xyzw_pose[6], xyzw_pose[3], xyzw_pose[4], xyzw_pose[5]]),
        ])

    def get_current_qpos(self) -> np.ndarray:
        with self._qpos_lock:
            return self.qpos

    def set_current_qpos(self, qpos: np.ndarray):
        with self._qpos_lock:
            self.qpos = qpos
            pin.forwardKinematics(self.model, self.data, self.qpos)
            self.ee_pose = pin.updateFramePlacement(self.model, self.data,
                                                    self.ee_frame_id)

    def get_dof(self) -> int:
        return pin.neutral(self.model).shape[0]

    def get_timestep(self) -> float:
        return self.dt

    def get_joint_names(self) -> List[str]:
        # Pinocchio by default add a dummy joint name called "universe"
        names = list(self.model.names)
        return names[1:]

    def is_use_gpu(self) -> bool:
        return False

    def get_current_ee_pose(self, cur_qpos) -> np.ndarray:
        self.cur_ee_pose = self.compute_ee_pose(cur_qpos)

        return self.cur_ee_pose

    # -------------------------------------------------------------------------- #

    # Kinematics function
    # -------------------------------------------------------------------------- #

    def compute_forward_kinematics(self, qpos: npt.NDArray):

        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data,
                                                 link_id)
        return pose.homogeneous

    def get_link_pose_inv(self, link_id: int) -> npt.NDArray:
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data,
                                                 link_id)
        return pose.inverse().homogeneous

    def compute_single_link_local_jacobian(self, qpos,
                                           link_id: int) -> npt.NDArray:
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J
