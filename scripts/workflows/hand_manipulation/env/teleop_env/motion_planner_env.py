from tools.curobo_planner import IKPlanner, MotionPlanner

import torch

import source.isaaclab.isaaclab.utils.math as math_utils


class ArmMotionPlannerEnv:

    def __init__(
        self,
        env=None,
        args_cli=None,
        env_cfg=None,
        collision_checker=False,
    ):

        if args_cli is not None:

            self.add_left_hand = args_cli.add_left_hand
            self.add_right_hand = args_cli.add_right_hand
        if env_cfg is not None:
            if env_cfg["params"]["arm_type"] == "ur5e":
                self.robot_file = "ur5e.yml"
                self.ee_name = "wrist_3_link"
            elif env_cfg["params"]["arm_type"] == "franka":
                self.robot_file = "franka_hand.yml"
                self.ee_name = "panda_link7"
        else:
            self.robot_file = "franka_hand.yml"

        self.env = env
        self.device = "cuda" if env is None else env.unwrapped.device
        self.args_cli = args_cli
        self.env_cfg = env_cfg

        if self.env is not None:

            self.init_settings()
        robot_name = None if env is None else f"{self.hand_side}_hand"

        self.motion_planner = MotionPlanner(
            self.env,
            self.robot_file,
            collision_checker=collision_checker,
            robot_name=robot_name)
        self.ik_planner = IKPlanner(self.env,
                                    device=self.device,
                                    robot_file=self.robot_file)

    def init_settings(self):
        if self.add_left_hand:
            self.hand_side = "left"
        elif self.add_right_hand:
            self.hand_side = "right"

        self.num_arm_joints = self.env_cfg["params"]["num_arm_joints"]
        # self.num_hand_joints = self.env_cfg["params"]["num_hand_joints"]
        if self.add_left_hand:
            self.leftpalm2wrist, self.leftwrist2palm = self.init_delta_transform(
                "left")
        elif self.add_right_hand:
            self.rightpalm2wrist, self.rightwrist2palm = self.init_delta_transform(
                "right")

    def init_delta_transform(self, hand_side):
        palm_state = self.env.scene[
            f"{hand_side}_palm_lower"]._data.root_state_w[:, :7]
        wrist_state = self.env.scene[
            f"{hand_side}_{self.ee_name}"]._data.root_state_w[:, :7]
        palm2wrist = torch.cat(math_utils.subtract_frame_transforms(
            palm_state[:, :3], palm_state[:, 3:7], wrist_state[:, :3],
            wrist_state[:, 3:7]),
                               dim=1)
        wrist2palm = torch.cat(math_utils.subtract_frame_transforms(
            wrist_state[:, :3], wrist_state[:, 3:7], palm_state[:, :3],
            palm_state[:, 3:7]),
                               dim=1)

        return palm2wrist, wrist2palm

    def plan_motion(self, ee_pose, apply_offset=True, arm_qpos=None):

        if arm_qpos is None:
            arm_qpos = self.env.scene[
                f"{self.hand_side}_hand"].root_physx_view.get_dof_positions()
        if apply_offset:
            palm2wrist = getattr(self, f"{self.hand_side}palm2wrist")

            wrist_ee_pose = torch.cat(math_utils.combine_frame_transforms(
                ee_pose[:, :3], ee_pose[:, 3:7], palm2wrist[:len(ee_pose), :3],
                palm2wrist[:len(ee_pose), 3:7]),
                                      dim=1)
        else:
            wrist_ee_pose = ee_pose

        # wrist_ee_pose[:, :3] -= self.env.scene[
        #     f"{self.hand_side}_hand"]._data.root_state_w[:len(ee_pose), :3]

        plan_ee_pose, traj = self.motion_planner.plan_motion(
            arm_qpos[0, :self.num_arm_joints].unsqueeze(0),
            wrist_ee_pose[0, :3].unsqueeze(0), wrist_ee_pose[0,
                                                             3:7].unsqueeze(0))
        wrist2palm = getattr(self, f"{self.hand_side}wrist2palm")

        if plan_ee_pose is None:
            return None, None

        # arm_pose = torch.cat(
        #     [plan_ee_pose.ee_position, plan_ee_pose.ee_quaternion], dim=1)

        palm_ee_pose = torch.cat(
            [plan_ee_pose.ee_position, plan_ee_pose.ee_quaternion], dim=1)

        # palm_ee_pose = torch.cat(math_utils.combine_frame_transforms(
        #     palm_ee_pose[:, :3], palm_ee_pose[:, 3:7], wrist2palm[:, :3],
        #     wrist2palm[:, 3:7]),
        #                          dim=1)

        torch.cuda.empty_cache()

        return palm_ee_pose, traj.position

    def ik_plan_motion(self, ee_pose, apply_offset=False):

        if apply_offset:

            palm2wrist = getattr(self, f"{self.hand_side}palm2wrist")

            wrist_ee_pose = torch.cat(math_utils.combine_frame_transforms(
                ee_pose[:, :3], ee_pose[:, 3:7], palm2wrist[:, :3],
                palm2wrist[:, 3:7]),
                                      dim=1)

        else:

            ee_position, ee_quaternion = math_utils.combine_frame_transforms(
                ee_pose[:, :3], ee_pose[:, 3:7],
                torch.tensor([[0.0000, 0.0000, 0.1070]],
                             device="cuda").repeat_interleave(len(ee_pose), 0),
                torch.tensor([[1., 0., 0., 0.]],
                             device="cuda").repeat_interleave(len(ee_pose), 0))

            init_arm_qpos = self.ik_planner.plan_motion(
                ee_position, ee_quaternion)
        # self.ik_planner.ik_solver.fk(torch.zeros((1, 7)).to(self.device))

        return init_arm_qpos

    # def execute_motion(self, arm_qpos):
    #     target_qpos = torch.zeros(
    #         (self.env.num_envs,
    #          self.num_arm_joints + self.num_hand_joints)).to(self.device)

    #     for i in range(arm_qpos.shape[0]):
    #         target_qpos[:, -self.num_arm_joints:] = arm_qpos[i]
    #         obs, reward, terminate, time_out, info = self.env.step(target_qpos)

    #     self.env.reset()

    #
