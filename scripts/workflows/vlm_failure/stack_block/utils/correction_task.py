import torch

import copy
import numpy as np
import isaaclab.utils.math as math_utils


class CorrectGraspEnv:

    def __init__(self, ):
        pass

    def high_level_plan(self, last_obs, new_pickup_object_name,
                        new_placement_object_name):
        self.multi_env.assign_objects(new_pickup_object_name,
                                      new_placement_object_name)
        self.multi_env.env_grasp.reset(last_obs, resample=True)
        grasp_action = self.multi_env.env_grasp.target_ee_traj
        self.multi_env.env_placement.get_target_placement_traj(
            grasp_action[-1])
        placement_action = self.multi_env.env_placement.target_ee_traj
        trajectory = torch.cat([grasp_action, placement_action], dim=0)

        final_obs = self.multi_env.step_env(trajectory, last_obs)["policy"]

        success_or_not = final_obs[self.gt_target_object_name][0][2] > (
            final_obs[self.gt_placement_object_name][0][2] + 0.02)
        if not success_or_not:
            self.current_failure_reasoning = f"Your pick up the {new_pickup_object_name} and place it onto the {new_placement_object_name}."

        return success_or_not

    def low_level_plan(
        self,
        last_obs,
        xyz_grasp_offset=[0, 0, 0.0],
        rpy_grasp_offset=[0, 0, 0],
    ):
        action = copy.deepcopy(np.asarray(self.demo["actions"]))

        if "+" in self.llm_feedback:
            factor = -1
        elif "-" in self.llm_feedback:
            factor = 1

        start_frame = int(len(action) / 2)

        target_grasp_pose = action[start_frame +
                                   self.multi_env.curobo_planner_length - 1]
        refine_pose = action[start_frame +
                             self.multi_env.curobo_planner_length +
                             self.multi_env.env_grasp.approach_horizon +
                             self.multi_env.env_grasp.close_gripper_horizon -
                             1]

        target_grasp_pose[:3] += np.abs(xyz_grasp_offset) * factor
        refine_pose[:3] += np.abs(xyz_grasp_offset) * factor
        curobo_tip_pose = self.multi_env.env_placement.placement_object._data.root_state_w[
            0, :7]

        rpy_grasp_offset = torch.as_tensor(rpy_grasp_offset).to(
            self.multi_env.device)
        orientations = math_utils.quat_from_euler_xyz(rpy_grasp_offset[0],
                                                      rpy_grasp_offset[1],
                                                      rpy_grasp_offset[2])

        target_grasp_pose[3:7] = math_utils.shortest_angles(
            math_utils.quat_mul(
                orientations.unsqueeze(0),
                torch.as_tensor([target_grasp_pose[3:7]
                                 ]).to(self.multi_env.device)),
            self.multi_env.env)[0].cpu().numpy()

        refine_pose[3:7] = target_grasp_pose[3:7]
        self.multi_env.env_grasp.robot_pose = torch.as_tensor(
            refine_pose[:7]).to(self.multi_env.device)
        grasp_action = self.multi_env.env_grasp.reset_planner_grasp(
            0,
            torch.as_tensor(target_grasp_pose).to(self.multi_env.device),
            sample_grasp=False)
        if grasp_action is None:
            return False, None
        self.multi_env.env_placement.get_target_placement_traj(
            grasp_action[-1], curobo_tip_pose)
        placement_action = self.multi_env.env_placement.target_ee_traj
        ee_traj = torch.cat([grasp_action, placement_action], dim=0)
        final_obs = self.multi_env.step_env(ee_traj, last_obs)["policy"]
        success_or_not = self.multi_env.env_placement.success_or_not(final_obs)
        print(self.llm_feedback)
        # if not success_or_not:

        #     gt_grasp_pose = action[self.multi_env.curobo_planner_length - 1]
        #     delta_error = abs(gt_grasp_pose[:3] - target_grasp_pose[:3])
        #     axes = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']
        #     max_deviation_index = np.argmax(
        #         delta_error)  # Index of the maximum deviation
        #     max_deviation_axis = axes[max_deviation_index]
        #     offset_sign = "+" if (
        #         target_grasp_pose[max_deviation_index]
        #     ) - gt_grasp_pose[max_deviation_index] > 0 else "-"
        # self.current_failure_reasoning = f"The robot failed to grasp the object due to the {offset_sign}{max_deviation_axis} offset."

        return success_or_not, final_obs

    # def gen_pan(self,
    #             high_level_policy=False,
    #             low_level_policy=True,
    #             last_obs=None):
    #     if high_level_policy:
    #         new_pickup_object_name = self.multi_env.target_object_name
    #         new_placement_object_name = self.multi_env.placement_object_name
    #         success_or_not = self.high_level_plan(new_pickup_object_name,
    #                                               new_placement_object_name,
    #                                               last_obs)
    #         self.llm_feedback = (
    #             f"It is a high level failure, you should pick up the {new_pickup_object_name} and place onto {new_placement_object_name}"
    #         )
    #     if low_level_policy:

    #         success_or_not, final_obs = self.low_level_plan(
    #             last_obs, [0, 0, -0.01])

    #     return success_or_not, final_obs
