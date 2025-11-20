from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from loop_rate_limiters import RateLimiter

import mink

FRANK_MJCF_PATH = Path(
    "/home/ensu/Documents/weird/IsaacLab/submodule/mink/examples/franka_emika_panda/mjx_scene.xml"
)
# IK parameters
SOLVER = "quadprog"
POS_THRESHOLD = 1e-4
ORI_THRESHOLD = 1e-4
MAX_ITERS = 20
from mink import SO3


class MinkFrankaSolver:

    def __init__(self, frequency=200.0):
        self.model = mujoco.MjModel.from_xml_path(FRANK_MJCF_PATH.as_posix())
        self.data = mujoco.MjData(self.model)
        self.configuration = mink.Configuration(self.model)
        self.end_effector_task = mink.FrameTask(
            frame_name="link7",
            frame_type="body",
            position_cost=1.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        self.posture_task = mink.PostureTask(model=self.model, cost=1e-2)
        self.tasks = [self.end_effector_task, self.posture_task]

        rate = RateLimiter(frequency=frequency, warn=False)
        self.dt = rate.dt

    def converge_ik(self, ):
        """
        Runs up to 'max_iters' of IK steps. Returns True if position and orientation
        are below thresholds, otherwise False.
        """
        for _ in range(MAX_ITERS):
            vel = mink.solve_ik(self.configuration, self.tasks, self.dt,
                                SOLVER, 1e-3)
            self.configuration.integrate_inplace(vel, self.dt)

            # Only checking the first FrameTask here (end_effector_task).
            # If you want to check multiple tasks, sum or combine their errors.
            err = self.tasks[0].compute_error(self.configuration)
            pos_achieved = np.linalg.norm(err[:3]) <= POS_THRESHOLD
            ori_achieved = np.linalg.norm(err[3:]) <= ORI_THRESHOLD

            if pos_achieved and ori_achieved:
                return True
        return False

    def solve_ik(self, qpos, targat_pos, target_quat):
        # --- Important: set initial target for posture ---
        cur_qpos = np.zeros_like(self.configuration.q)
        cur_qpos[:len(qpos)] = qpos

        self.configuration.update(cur_qpos)
        self.posture_task.set_target_from_configuration(self.configuration)

        T_wt = mink.SE3.from_rotation_and_translation(
            rotation=SO3(target_quat),
            translation=targat_pos,
        )
        self.end_effector_task.set_target(T_wt)

        # Attempt to converge IK
        self.converge_ik()

        return self.configuration.q[:7]

    def fk(self, q, frame_name="link7", frame_type="body"):
        """
        Compute FK: world → frame pose from joint config q.
        
        Args:
            model: mujoco.MjModel
            data: mujoco.MjData
            q: np.ndarray of shape (n_joints,)
            frame_name: name of end-effector (e.g. "link7" or "attachment_site")
            frame_type: "body" or "site"
        Returns:
            mink.SE3 object (position + orientation)
        """
        # Set joint positions
        self.data.qpos[:len(q)] = q
        mujoco.mj_forward(self.model, self.data)

        if frame_type == "body":
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    frame_name)
            pos = self.data.xpos[bid].copy()
            quat = self.data.xquat[bid].copy()  # wxyz
        elif frame_type == "site":
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                    frame_name)
            pos = self.data.site_xpos[sid].copy()
            quat = self.data.site_xquat[sid].copy()
        else:
            raise ValueError(f"Unknown frame_type {frame_type}")

        return pos, quat


if __name__ == "__main__":
    # main()
    mink_solver = MinkFrankaSolver()
    qpos = np.array([
        0.17301081120967865, 0.045076385140419006, -0.347432941198349,
        -2.188567876815796, 1.1745686531066895, 1.5173358917236328,
        0.18716195225715637
    ])
    pos, quat = mink_solver.fk(qpos)

    print("EE position:", pos)

    print("EE orientation (quat, wxyz):", quat)

    # Attempt to converge IK
    joint_pos = mink_solver.solve_ik(
        qpos,
        np.array([0.50, 0.0, 0.50]),
        np.array([0.0, 9.2460e-01, -3.8094e-01, 0.0]),
    )

    print("Final q:", joint_pos)
