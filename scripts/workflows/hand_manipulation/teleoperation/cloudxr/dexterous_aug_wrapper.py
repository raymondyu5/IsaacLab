import isaaclab.utils.math as math_utils

import torch


class DexterousAugWrapper:

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
        begin_index=4,
        skip_steps=1,
    ):
        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.device = env.device
        self.num_envs = env.num_envs
        self.use_delta_pose = False if "Rel" not in self.args_cli.task else True
        self.begin_index = begin_index

        self.skip_steps = skip_steps
        self.task = ("place" if "Place" in self.args_cli.task else
                     "open" if "Open" in self.args_cli.task else "grasp")
        self.augment = args_cli.augment
        self.augment_times = args_cli.augment_times
        if args_cli.add_left_hand:
            self.hand_side = "left"
        elif args_cli.add_right_hand:
            self.hand_side = "right"

        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

        from scripts.workflows.hand_manipulation.env.teleop_env.motion_planner_env import ArmMotionPlannerEnv
        self.arm_motion_env = ArmMotionPlannerEnv(
            self.env,
            self.args_cli,
            self.env_cfg,
        )

    def reset_augment_env(self,
                          next_init_joint_pose,
                          next_pick_object_pose,
                          next_place_object_pose=None,
                          robot_actions=None,
                          env_mask=None):

        demo_pick_object_pose = next_pick_object_pose.clone()
        demo_init_joint_pose = next_init_joint_pose.clone()

        reset_pick_object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[
                ..., :7].clone()

        self.delta_pick_object_pose = torch.cat(
            math_utils.subtract_frame_transforms(
                demo_pick_object_pose[:, :3], demo_pick_object_pose[:, 3:7],
                reset_pick_object_pose[:, :3], reset_pick_object_pose[:, 3:7]),
            dim=1)

        # data = math_utils.combine_frame_transforms(demo_pick_object_pose[:, :3], demo_pick_object_pose[:, 3:7],self.delta_pick_object_pose[:, :3], self.delta_pick_object_pose[:, 3:7])

        if next_place_object_pose is not None:
            demo_place_object_pose = next_place_object_pose.clone()
            reset_place_object_pose = self.env.scene[
                f"{self.hand_side}_hand_place_object"]._data.root_state_w[
                    ..., :7]
            self.delta_place_object_pose = torch.cat(
                math_utils.subtract_frame_transforms(
                    demo_place_object_pose[:, :3], demo_place_object_pose[:,
                                                                          3:7],
                    reset_place_object_pose[:, :3],
                    reset_place_object_pose[:, 3:7]),
                dim=1)
        self.demo_robot_actions = robot_actions.clone()
        self.demo_mask = env_mask
        self.lift_pick_object = torch.zeros_like(self.env_ids).to(
            self.device).bool()

    def reset_multi_env(
        self,
        init_actions,
        next_init_joint_pose,
        env_ids,
    ):
        for i in range(10):

            self.env.scene[
                f"{self.hand_side}_hand"].root_physx_view.set_dof_positions(
                    next_init_joint_pose, indices=env_ids)

            if self.use_delta_pose:

                new_obs, rewards, terminated, time_outs, extras = self.env.step(
                    torch.zeros(self.env.action_space.shape).to(self.device))
            else:

                new_obs, rewards, terminated, time_outs, extras = self.env.step(
                    init_actions)
        self.init_pick_object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[
                ..., :7].clone()

        return new_obs

    def step_aug_action(self, action):
        mask_action = action[~self.demo_mask].clone()

        cur_pick_object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[..., :7]

        self.lift_pick_object = self.lift_pick_object | (
            (cur_pick_object_pose[:, 2] - self.init_pick_object_pose[:, 2])
            > 0.5)

        to_pick = ~self.lift_pick_object
        # if to_pick.sum() > 0:

        action[..., :7] = torch.cat(math_utils.combine_frame_transforms(
            action[:, :3],
            action[:, 3:7],
            self.delta_pick_object_pose[:, :3],
            self.delta_pick_object_pose[:, 3:7],
        ),
                                    dim=1)

        if self.task == "place":
            cur_place_object_pose = self.env.scene[
                f"{self.hand_side}_hand_place_object"]._data.root_state_w[
                    ..., :7][self.demo_mask]

        action[~self.demo_mask] = mask_action

        return action
