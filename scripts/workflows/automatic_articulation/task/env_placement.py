import torch

import isaaclab.utils.math as math_utils
from scripts.workflows.automatic_articulation.utils.process_action import get_robottip_pose, curobo2robot_actions
from curobo.types.robot import JointState


class PlacementEnv:

    def __init__(self,
                 env,
                 planner,
                 use_relative_pose=False,
                 collision_checker=False,
                 env_config=None):
        self.env = env
        self.device = env.device
        self.robot = env.scene["robot"]
        self.kitchen = env.scene["kitchen"]
        self.use_relative_pose = use_relative_pose
        self.collision_checker = collision_checker
        self.planner = planner

        self.env_config = env_config
        self.init_setting()

    def init_setting(self):

        self.target_handle_name = self.kitchen.cfg.articulation_cfg[
            "target_drawer"]

        self.handle_id, handle_name = self.kitchen.find_bodies(
            "drawer_" + self.target_handle_name.split("_")[1])

        self.target_joint_type = self.kitchen.cfg.articulation_cfg[
            "target_joint_type"]

        self.placement_offset = torch.as_tensor(
            self.env_config["params"]["Task"]["placement"]
            ["placement_offset"]).to(self.device)
        self.open_gripper = self.env_config["params"]["Task"]["placement"][
            "open_gripper_horizon"]
        self.grasp_object_name = self.env.scene[
            "kitchen"].cfg.articulation_cfg["target_object"]
        self.placement_region = self.env_config["params"]["Task"][
            "placement"].get("placement_region", None)
        self.lower_horizon = self.env_config["params"]["Task"][
            "placement"].get("lower_horizon", 0)

    def get_target_placement_pos(self):
        handle_location = self.kitchen._data.body_state_w[0][
            self.handle_id][:, :3]
        placement_pose = handle_location.clone()

        # if self.target_joint_type == "prismatic":
        placement_pose += self.placement_offset[:3]

        robot_dof_pos = self.robot.root_physx_view.get_dof_positions()

        placement_quat = self.placement_offset[3:7].unsqueeze(0)
        robot_root_pose = self.robot._data.root_state_w
        curobo_position, curobo_quat = math_utils.subtract_frame_transforms(
            robot_root_pose[:, :3], robot_root_pose[:, 3:7], placement_pose,
            placement_quat)

        return curobo_position, curobo_quat, robot_dof_pos

    def get_target_placement_traj(self,
                                  current_ee_pose=None,
                                  target_object_pose=None,
                                  target_object_quat=None):
        curobo_position, curobo_quat, robot_dof_pos = self.get_target_placement_pos(
        )

        if current_ee_pose is not None:

            curobo_ik = self.planner.curobo_ik

            robot_dof_pos = curobo_ik.plan_motion(
                current_ee_pose[:3].clone(),
                current_ee_pose[3:7].clone()).to(self.device).squeeze(0)[:, :8]

        if target_object_pose is not None:

            robot_base_pose = self.robot._data.root_state_w
            target_object_pose = target_object_pose.to(
                self.device) - robot_base_pose[0, :3]

            curobo_position = target_object_pose.unsqueeze(0)
        if target_object_quat is not None:

            curobo_quat = target_object_quat

        ee_pose, traj = self.planner.plan_motion(robot_dof_pos,
                                                 curobo_position, curobo_quat)

        if ee_pose is None:
            return None

        self.planner.clear_obstacles()
        curobo_target_positions = ee_pose.ee_position
        curobo_targe_quaternion = ee_pose.ee_quaternion
        curobo_target_ee_pos = torch.cat([
            curobo_target_positions, curobo_targe_quaternion,
            torch.zeros(len(curobo_targe_quaternion), 1).to(self.device)
        ],
                                         dim=1)
        _, self.target_ee_traj = curobo2robot_actions(curobo_target_ee_pos,
                                                      self.device)

        self.target_ee_traj[:, -1] = -1  # close gripper

        self.lower_gripper_pose = self.target_ee_traj[-1].unsqueeze(0).repeat(
            self.lower_horizon, 1)
        self.lower_gripper_pose[:, 2] -= torch.arange(self.lower_horizon).to(
            self.env.device) * 0.01  # open gripper
        if self.lower_horizon > 0:
            gripper_open_ee_pose = self.lower_gripper_pose[-1].unsqueeze(
                0).repeat(self.open_gripper, 1)
        else:
            gripper_open_ee_pose = self.target_ee_traj[-1].unsqueeze(0).repeat(
                self.open_gripper, 1)

        if self.env_config["params"]["Task"]["placement"].get(
                "open_offset", None):

            gripper_open_ee_pose[10:, :3] += torch.as_tensor([
                self.env_config["params"]["Task"]["placement"]["open_offset"]
            ]).to(self.device).repeat_interleave(self.open_gripper - 10, 0)
        else:
            gripper_open_ee_pose[:int(self.open_gripper / 2),
                                 0] += torch.arange(0,
                                                    int(self.open_gripper / 2),
                                                    1).to(self.device) * 0.0
            gripper_open_ee_pose[int(self.open_gripper / 2):,
                                 0] += 0.01 * self.open_gripper / 2 * 0.0
        gripper_open_ee_pose[int(self.open_gripper / 2):, -1] = 1

        self.target_ee_traj = torch.cat(
            [self.target_ee_traj, gripper_open_ee_pose], dim=0)

        self.reach_length = len(self.target_ee_traj)
        self.count_steps = 0
        return True

    def success_or_not(self, observation):
        if self.placement_region is not None:

            bb_region = self.placement_region["bound_region"]
            object_pose = observation["policy"][
                f"{self.grasp_object_name}_root_pose"][0]

            placement_or_not = (object_pose[0] > bb_region[0]
                                and object_pose[1] > bb_region[1]
                                and object_pose[2] > bb_region[2]
                                and object_pose[0] < bb_region[3]
                                and object_pose[1] < bb_region[4]
                                and object_pose[2] < bb_region[5])
          
            
        else:
            placement_or_not = abs(
                observation["policy"]["drawer_pose"][0, 2] -
                observation["policy"][f"{self.grasp_object_name}_pose"][0, 2]
            ) < 0.07
        if self.env_config["params"]["Task"]["placement"].get(
                "placement_object", None):
            bb_region = self.placement_region["bound_region"]
            object_pos = observation["policy"][self.grasp_object_name +
                                               "_root_pose"]
            object_quat = object_pos[:, 3:7]
            object_angles = math_utils.euler_xyz_from_quat(object_quat)
            placement_or_not = (abs(object_angles[0]) > object_pose[6]) & (
                object_pos[:, 2] > -0.10) & (object_pos[:, 1] > bb_region[1])
            

        return placement_or_not
