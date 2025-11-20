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

        self.use_relative_pose = use_relative_pose
        self.collision_checker = collision_checker
        self.planner = planner

        self.env_config = env_config

        self.placement_object = None
        self.placement_object_name = None
        self.target_object = None
        self.target_object_name = None
        self.init_setting()

    def init_setting(self):

        self.placement_offset = torch.as_tensor(
            self.env_config["params"]["Task"]["placement"]
            ["placement_offset"]).to(self.device)
        self.open_gripper_horizon = self.env_config["params"]["Task"][
            "placement"]["open_gripper_horizon"]
        self.gripper_length_offset = torch.as_tensor(
            self.env_config["params"]["Task"]["gripper_length_offset"]).to(
                self.device)
        self.init_ee_pos = torch.as_tensor(
            self.env_config["params"]["Task"]["placement"]["init_ee_pos"]).to(
                self.device)
        self.refine_horizon = self.env_config["params"]["Task"]["placement"][
            "refine_horizon"]

    def get_target_placement_traj(self,
                                  current_ee_pose=None,
                                  target_object_pose=None):
        # extarct the  placement object from the default root state
        if target_object_pose is None:
            curobo_tip_pose = self.placement_object._data.root_state_w[0, :7]
        else:
            curobo_tip_pose = target_object_pose
        default_top_pose = self.placement_object._data.default_root_state[
            0, :7]

        curobo_tip_pose[3:7] = math_utils.quat_mul(
            curobo_tip_pose[3:7],
            math_utils.quat_inv(default_top_pose[3:7]),
        )

        robot_dof_pos = self.robot.root_physx_view.get_dof_positions()
        curobo_tip_pose[3:7] = math_utils.quat_mul(
            curobo_tip_pose[3:7],
            self.init_ee_pos[3:7],
        )

        curobo_ee_pose = math_utils.combine_frame_transforms(
            curobo_tip_pose[:3], curobo_tip_pose[3:7],
            self.gripper_length_offset[:3], self.gripper_length_offset[3:7])

        if current_ee_pose is not None:

            curobo_ik = self.planner.curobo_ik

            robot_dof_pos = curobo_ik.plan_motion(
                current_ee_pose[:3].clone(),
                current_ee_pose[3:7].clone()).to(self.device).squeeze(0)[:, :8]

        if self.collision_checker:
            self.planner.add_obstacle()

            for collision_object in self.planner.motion_gen.world_model.objects:

                if self.target_object_name not in collision_object.name:
                    continue

                attach_object_name = collision_object.name

            self.planner.motion_gen.attach_objects_to_robot(
                JointState.from_position(robot_dof_pos.view(1, -1)),
                [attach_object_name])

        curobo_ee_pose = torch.cat(curobo_ee_pose)

        ee_pose, traj = self.planner.plan_motion(robot_dof_pos,
                                                 curobo_ee_pose[:3],
                                                 curobo_ee_pose[3:7])
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

        # refine the gripper pose

        refine_gripper_ee_pose = curobo_ee_pose.unsqueeze(0).repeat(
            self.refine_horizon, 1)
        refine_gripper_ee_pose = torch.cat([
            refine_gripper_ee_pose,
            torch.ones(len(refine_gripper_ee_pose)).unsqueeze(1).to(
                self.device) * -1
        ],
                                           dim=1)
        # gripper openning
        gripper_open_ee_pose = self.target_ee_traj[-1].unsqueeze(0).repeat(
            self.open_gripper_horizon, 1)
        gripper_open_ee_pose[:, -1] = 1
        self.target_ee_traj = torch.cat([
            self.target_ee_traj, refine_gripper_ee_pose, gripper_open_ee_pose
        ],
                                        dim=0)

        self.reach_length = len(self.target_ee_traj)
        self.count_steps = 0
        return True

    def success_or_not(self, observation):

        return self.target_object.data.root_state_w[0][2] > (
            self.placement_object.data.root_state_w[0][2] + 0.02)
