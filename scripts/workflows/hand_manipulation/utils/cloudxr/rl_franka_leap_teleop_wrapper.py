from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
import numpy as np
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer
import isaaclab.utils.math as math_utils
import copy


class RLBCReplayDatawrapper:

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
    ):

        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.device = env.device
        self.num_envs = env.num_envs
        self.use_delta_pose = False if "Rel" not in self.args_cli.task else True
        self.hand_side = "right" if self.add_right_hand else "left"

        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_cfg,
            save_path=args_cli.save_path,
            load_path=args_cli.load_path,
        )
        self.raw_data = self.collector_interface.raw_data["data"]

        self.init_reload_data_buffer()
        self.init_setting()
        self.init_reload_data()

    def init_reload_data_buffer(self):

        self.obs_buffer = []
        self.actions_buffer = []
        self.does_buffer = []
        self.rewards_buffer = []

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

        self.init_actions = []
        if self.add_left_hand:
            self.init_actions.append(init_pose)
        if self.add_right_hand:
            self.init_actions.append(init_pose)
        self.init_actions = torch.cat(self.init_actions,
                                      dim=1).repeat_interleave(self.num_envs,
                                                               dim=0).to(
                                                                   self.device)

        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

        self.raw_data = self.collector_interface.raw_data["data"]

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

    def reset_robot_joints(self, ):

        self.env.scene[
            f"{self.hand_side}_hand"].root_physx_view.set_dof_positions(
                self.init_joint_poses, indices=self.env_ids)

    def reset_rigid_objects(self, ):

        reset_pose = self.rigid_objects_poses.clone()

        reset_pose[..., :3] += self.env.scene.env_origins
        self.env.scene[f"{self.hand_side}_hand_object"].write_root_pose_to_sim(
            reset_pose, env_ids=self.env_ids)

    def init_reload_data(self):
        self.all_actions = torch.zeros(self.env.action_space.shape).to(
            self.device).unsqueeze(0).repeat_interleave(
                self.env.max_episode_length - 20, dim=0)
        self.rigid_objects_poses = torch.zeros((self.env.num_envs, 7),
                                               device=self.device)
        self.init_joint_poses = torch.zeros((
            self.env.num_envs,
            self.env.scene[f"{self.hand_side}_hand"].data.joint_pos.shape[-1]),
                                            device=self.device)

        for index, demo_name in enumerate(self.raw_data.keys()):

            demo_data = self.raw_data[demo_name]
            num_actions = len(demo_data["actions"])

            self.all_actions[
                :num_actions,
                index,
            ] = torch.as_tensor(np.array(demo_data["actions"])).to(
                self.device).clone()

            self.all_actions[num_actions:, index,
                             -self.num_hand_joints:] = self.all_actions[
                                 num_actions - 1, index,
                                 -self.num_hand_joints:].clone()
            self.all_actions[num_actions:, 2] = 0.6
            self.rigid_objects_poses[index, :] = torch.as_tensor(
                demo_data["obs"][f"{self.hand_side}_manipulated_object_pose"]
                [0]).to(self.device).clone()
            self.init_joint_poses[index, :] = torch.as_tensor(
                demo_data["obs"][f"{self.hand_side}_hand_joint_pos"][0]).to(
                    self.device).clone()

    def reset_env(self, ):

        self.init_reload_data_buffer()

        self.env.reset()
        self.reset_rigid_objects()

        for i in range(10):

            self.reset_robot_joints()
            obs, rewards, terminated, time_outs, extras = self.env.step(
                torch.zeros(self.env.action_space.shape).to(self.device))

        return obs

    def lift_or_not(self, ):
        target_object_state = self.env.scene[
            f"{self.hand_side}_hand_object"].data.root_state_w[..., :7]
        success_flag = target_object_state[:, 2] > 0.20
        if success_flag.sum() > 0:

            index = torch.nonzero(success_flag, as_tuple=True)[0]

            self.filter_out_data(index)
        return success_flag

        return success_flag

    def filter_out_data(self, index):

        obs_buffer = []
        actions_buffer = []
        rewards_buffer = []
        does_buffer = []
        for i in range(len(self.obs_buffer)):
            per_obs = self.obs_buffer[i]
            per_obs_dict = {}
            for obs_key in list(per_obs["policy"].keys()):
                per_obs_dict[obs_key] = per_obs["policy"][obs_key][index]

            obs_buffer.append(per_obs_dict)
            actions_buffer.append(self.actions_buffer[i][index])
            rewards_buffer.append(self.rewards_buffer[i][index])
            does_buffer.append(self.does_buffer[i])

        self.collector_interface.add_demonstraions_to_buffer(
            obs_buffer,
            actions_buffer,
            rewards_buffer,
            does_buffer,
        )

    def step_reload_data(self):
        self.rew_buffer = []
        self.obs_buffer = []
        self.actions_buffer = []
        self.next_obs_buffer = []
        self.does_buffer = []

        last_obs = self.reset_env()

        for action in self.all_actions:

            raw_action = math_utils.denormalize_action(action.clone(),
                                                       self.action_bound,
                                                       self.env.step_dt)
            raw_action[:, -self.num_hand_joints:] /= self.env.step_dt

            new_obs, rewards, terminated, time_outs, extras = self.env.step(
                raw_action)
            done = terminated | time_outs

            self.obs_buffer.append(last_obs)
            self.actions_buffer.append(action.clone())
            self.does_buffer.append(done)
            self.rewards_buffer.append(rewards)

            last_obs = copy.deepcopy(new_obs)

        success_or_not = self.lift_or_not()
        success_rate = success_or_not.sum().float() / success_or_not.shape[0]

        print("Success rate of the demonstration: ", success_rate)
