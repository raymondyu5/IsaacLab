from isaaclab.envs import mdp
from isaaclab.utils.math import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import numpy as np
from scripts.workflows.hand_manipulation.utils.cloudxr.utils import reset_root_state_uniform
import torch
import isaaclab.utils.math as math_utils

import copy
import math

from scripts.workflows.hand_manipulation.teleoperation.cloudxr.dexterous_aug_wrapper import DexterousAugWrapper


class TeleoperationProcessor(DexterousAugWrapper):

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
        begin_index=4,
        skip_steps=1,
    ):

        super().__init__(
            env,
            env_cfg,
            args_cli,
            begin_index,
            skip_steps,
        )

        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_cfg,
            save_path=args_cli.save_path,
            load_path=args_cli.load_path,
            save_zarr=True)
        self.rollout_times = 0

        self.init_data_buffer()

        self.init_setting()

    def init_data_buffer(self):

        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

    def init_rigid_object_setting(self):

        self.rigid_object_setting = self.env_cfg["params"].get(
            "RigidObject", {})

        self.pick_object_list = self.env_cfg["params"]["multi_cluster_rigid"][
            f"{self.hand_side}_hand_object"]["objects_list"]

        self.rigid_object_list = list(self.env_cfg["params"].get(
            "RigidObject", {}).keys())
        if self.task == "place":

            self.place_object_list = self.env_cfg["params"][
                "multi_cluster_rigid"][f"{self.hand_side}_hand_place_object"][
                    "objects_list"]

        self.root_state = torch.as_tensor(self.rigid_object_setting[
            self.rigid_object_list[0]]["pos"]).unsqueeze(0).to(
                self.device).repeat_interleave(self.num_envs, dim=0)

        root_quat = obtain_target_quat_from_multi_angles(
            self.rigid_object_setting[self.rigid_object_list[0]]["rot"]
            ["axis"], self.rigid_object_setting[
                self.rigid_object_list[0]]["rot"]["angles"])

        root_quat = torch.as_tensor(root_quat).unsqueeze(0).to(
            self.device).repeat_interleave(self.num_envs, dim=0)
        self.root_state = torch.cat([self.root_state, root_quat],
                                    dim=1).to(self.device)

    def init_setting(self):

        init_ee_pose = torch.as_tensor(
            self.env_cfg["params"]["init_ee_pose"]).to(
                self.device).unsqueeze(0)
        self.num_hand_joint = self.env_cfg["params"]["num_hand_joints"]

        init_pose = torch.cat([
            init_ee_pose,
            torch.zeros(1, self.num_hand_joint).to(self.device)
        ],
                              dim=1)
        self.init_rigid_object_setting()

        self.init_ik_joint_pose = self.arm_motion_env.ik_plan_motion(
            init_pose.to(torch.float32)).reshape(-1, 7).repeat_interleave(
                self.num_envs, 0)

        self.init_ik_joint_pose = torch.cat([
            self.init_ik_joint_pose,
            torch.zeros((self.num_envs, self.num_hand_joint)).to(self.device)
        ],
                                            dim=1)

        self.init_actions = []
        if self.add_left_hand:
            self.init_actions.append(init_pose)
        if self.add_right_hand:
            self.init_actions.append(init_pose)
        self.init_actions = torch.cat(self.init_actions,
                                      dim=1).repeat_interleave(self.num_envs,
                                                               dim=0).to(
                                                                   self.device)

        self.raw_data = self.collector_interface.raw_data["data"]
        self.demo_index = 0
        self.num_hand_joints = self.env_cfg["params"]["num_hand_joints"]
        arm_action_bound = torch.as_tensor(
            self.env_cfg["params"]["Task"]["action_range"]).to(self.device)
        arm_action_bound = torch.as_tensor(
            self.env_cfg["params"]["Task"]["action_range"]).to(self.device)

        arm_action_bound = torch.stack([
            torch.tensor(
                [-arm_action_bound[0]] * 3 + [-arm_action_bound[1]] * 3,
                device=self.device),
            torch.tensor([arm_action_bound[0]] * 3 + [arm_action_bound[1]] * 3,
                         device=self.device)
        ],
                                       dim=1)
        hand_action_bound = self.env.scene[f"right_hand"]._data.joint_limits[
            0, -self.num_hand_joint:]
        self.action_bound = torch.cat([arm_action_bound, hand_action_bound],
                                      dim=0)
        self.init_data()

    def selective_skip(self, arr: torch.Tensor, ratio: float, block: int = 10):

        keep = int(math.ceil(block * ratio))  # number to keep in each block
        skip = block - keep  # number to skip

        pattern = torch.cat([
            torch.ones(keep, dtype=torch.bool),
            torch.zeros(skip, dtype=torch.bool)
        ])

        repeat = math.ceil(arr.shape[0] / len(pattern))
        mask = pattern.repeat(repeat)[:arr.shape[0]]

        return arr[mask.to(arr.device)]

    def init_data(self):

        self.demo_action = []
        self.init_joint_pose = []
        from collections import defaultdict
        self.pick_object_init_pose = []
        self.pick_object_counts = defaultdict(int)
        self.pick_demo_list = []
        self.whole_pick_object_pose = []

        for name in self.pick_object_list:

            self.pick_object_counts[name] = 0

        if self.task == "place":
            self.place_object_init_pose = []

        for i in range(len(self.raw_data.keys())):

            demo_obs = self.raw_data[f"demo_{i}"]["obs"]
            if self.use_delta_pose:
                demo_action = torch.zeros(250, 22).to(self.device)
            else:

                demo_action = torch.zeros(250, 23).to(self.device)

            init_pick_object_pose = None
            init_place_object_pose = None

            for rigid_object in self.rigid_object_list:

                rigid_object_pose = demo_obs.get(f"{rigid_object}_pose", None)

                if rigid_object_pose is not None:
                    init_rigid_object_pose = rigid_object_pose[
                        self.begin_index]

                    if init_rigid_object_pose[2] > -0.10:

                        if rigid_object in self.pick_object_list:
                            self.whole_pick_object_pose.append(
                                rigid_object_pose)
                            init_pick_object_pose = torch.as_tensor(
                                init_rigid_object_pose).unsqueeze(0).clone()
                            pick_object_name = rigid_object

                        if rigid_object in self.place_object_list:
                            init_place_object_pose = torch.as_tensor(
                                init_rigid_object_pose).unsqueeze(0).clone()

            if init_pick_object_pose is not None:
                if self.task == "place" and init_place_object_pose is None:

                    continue

                self.pick_object_counts[pick_object_name] += 1
                self.pick_demo_list.append(pick_object_name)
                self.pick_object_init_pose.append(init_pick_object_pose)
                if self.task == "place":
                    self.place_object_init_pose.append(init_place_object_pose)
            else:
                continue
            robot_action = torch.as_tensor(
                np.array(self.raw_data[f"demo_{i}"]["actions"]
                         [self.begin_index:])).to(self.device)
            skip_step = np.floor(1 / (robot_action.shape[0] / 250) * 10) / 10

            if skip_step < 1.0:

                robot_action = self.selective_skip(robot_action,
                                                   ratio=skip_step)

            demo_action[:robot_action.shape[0]] = robot_action

            demo_action[robot_action.shape[0]:] = robot_action[-1]
            self.demo_action.append(demo_action.unsqueeze(0))
            init_joint_pose = demo_obs.get(f"{self.hand_side}_hand_joint_pos",
                                           None)

            self.init_joint_pose.append(
                torch.as_tensor(
                    init_joint_pose[self.begin_index]).unsqueeze(0))

        self.init_joint_pose = torch.cat(self.init_joint_pose,
                                         dim=0).to(self.device)

        self.demo_action = torch.cat(self.demo_action, dim=0)
        self.pick_object_init_pose = torch.cat(self.pick_object_init_pose,
                                               dim=0).to(self.device)
        if self.task == "place":
            self.place_object_init_pose = torch.cat(
                self.place_object_init_pose, dim=0).to(self.device)
        assert len(self.demo_action) == len(self.init_joint_pose) == len(
            self.pick_demo_list) == len(self.place_object_init_pose)

        self.sim_env_object_list = ["test" for i in range(self.num_envs)]

        self.pick_demo_list = np.array(self.pick_demo_list, dtype=object)

        sim_env_object_array = np.array(self.sim_env_object_list, dtype=object)

        for index, object_name in enumerate(self.pick_object_list):
            mask = (self.env_ids.cpu().numpy() %
                    len(self.pick_object_list)) == index
            sim_env_object_array[mask] = object_name

        self.sim_env_object_list = sim_env_object_array.tolist()

    def extract_demo(self, ):

        init_joint_pose = self.init_ik_joint_pose.clone()

        init_pick_object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[..., :7]
        init_pick_object_pose[:, :3] -= self.env.scene.env_origins
        self.env_mask = torch.zeros(self.num_envs).to(self.device).bool()

        if self.task == "place":
            init_place_object_pose = self.env.scene[
                f"{self.hand_side}_hand_place_object"]._data.root_state_w[
                    ..., :7]
            init_place_object_pose[:, :3] -= self.env.scene.env_origins

        self.robot_actions = self.init_actions.clone().unsqueeze(
            1).repeat_interleave(self.demo_action.shape[1], dim=1)
        for index, object_name in enumerate(self.pick_object_list):

            # set up maks
            env_mask = (self.env_ids.cpu().numpy() %
                        len(self.pick_object_list)) == index

            demo_mask = np.ones(len(self.init_joint_pose)).astype(bool)

            # set up env indices
            env_indices = np.where(env_mask)[0].tolist()
            selected_demo_id = np.where(
                self.pick_demo_list == object_name)[0].tolist()
            if len(selected_demo_id) == 0:
                continue
            selected_demo_id = np.array(selected_demo_id)
            final_selected_demo_id = selected_demo_id[:len(env_indices)]

            demo_mask[final_selected_demo_id] = False

            # set up init joint pose and object pose
            init_joint_pose[env_indices[:len(
                final_selected_demo_id
            )]] = self.init_joint_pose[final_selected_demo_id].clone()

            self.init_joint_pose = self.init_joint_pose[demo_mask]

            init_pick_object_pose[env_indices[:len(
                final_selected_demo_id
            )]] = self.pick_object_init_pose[final_selected_demo_id].clone()
            self.pick_object_init_pose = self.pick_object_init_pose[demo_mask]
            self.pick_demo_list = self.pick_demo_list[demo_mask]

            if self.task == "place":
                init_place_object_pose[env_indices[:len(
                    final_selected_demo_id)]] = self.place_object_init_pose[
                        final_selected_demo_id].clone()
                self.place_object_init_pose = self.place_object_init_pose[
                    demo_mask]

            self.env_mask[env_indices[:len(final_selected_demo_id)]] = True

            self.robot_actions[
                env_indices[:len(final_selected_demo_id)]] = self.demo_action[
                    final_selected_demo_id].clone()
            self.demo_action = self.demo_action[demo_mask]

        init_pick_object_pose[:, :3] += self.env.scene.env_origins
        if self.task == "place":
            init_place_object_pose[:, :3] += self.env.scene.env_origins
        self.next_init_joint_pose = init_joint_pose.clone()
        self.next_pick_object_pose = init_pick_object_pose.clone()
        if self.task == "place":
            self.next_place_object_pose = init_place_object_pose.clone()
        self.rollout_times = 0

    def reset_demo_env(self):

        if len(self.init_joint_pose) == 0:
            return None
        if not self.augment or (self.augment and
                                (self.rollout_times == 0
                                 or self.rollout_times == self.augment_times)):
            self.extract_demo()

        self.rollout_times += 1
        new_obs = self.reset_multi_env(
            self.init_actions,
            self.next_init_joint_pose,
            self.env_ids,
        )

        if not self.augment:

            self.env.scene[
                f"{self.hand_side}_hand_object"].write_root_link_pose_to_sim(
                    self.next_pick_object_pose.clone())

            if self.task == "place":
                self.env.scene[
                    f"{self.hand_side}_hand_place_object"].write_root_link_pose_to_sim(
                        self.next_place_object_pose)

        # else:
        self.reset_augment_env(
            self.next_init_joint_pose,
            self.next_pick_object_pose,
            next_place_object_pose=self.next_place_object_pose
            if self.task == "place" else None,
            robot_actions=self.robot_actions,
            env_mask=self.env_mask,
        )

        return new_obs
