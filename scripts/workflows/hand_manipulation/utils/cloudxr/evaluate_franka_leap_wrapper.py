import torch
import isaaclab.utils.math as math_utils
from isaaclab.envs import mdp
from isaaclab.utils.math import create_rotation_matrix_from_view, obtain_target_quat_from_multi_angles
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import numpy as np
from scripts.workflows.hand_manipulation.utils.cloudxr.utils import reset_root_state_uniform
import copy


class EvaluateFrankaLeapWrapper:

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

        self.target_object_name = f"{self.hand_side}_hand_object"
        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_cfg,
            save_path=args_cli.save_path,
            load_path=args_cli.load_path,
            normalize_action=args_cli.normalize_action,
        )

        self.init_data_buffer()

        self.init_setting()

    def init_data_buffer(self):

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

        for i in range(10):
            if self.use_delta_pose:
                self.env.step(
                    torch.zeros(self.env.action_space.shape).to(self.device))

            else:

                self.env.step(self.init_actions)

        self.env_ids = torch.arange(self.env.num_envs).to(self.device)

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

    def reset_robot_joints(self,
                           demo_obs=None,
                           hand_side="right",
                           init_joint_pose=None):
        if demo_obs is not None:
            init_joint_pose = demo_obs.get(f"{hand_side}_hand_joint_pos", None)

        if init_joint_pose is not None:

            self.env.scene[
                f"{hand_side}_hand"].root_physx_view.set_dof_positions(
                    torch.as_tensor(init_joint_pose[0]).unsqueeze(0).to(
                        self.device).repeat_interleave(self.num_envs, dim=0),
                    indices=self.env_ids)

    def reset_rigid_objects(self, demo_obs):

        init_rigid_object_pose = demo_obs[
            f"{self.hand_side}_manipulated_object_pose"]

        reset_pose = torch.as_tensor(
            init_rigid_object_pose[0]).unsqueeze(0).to(self.device).repeat(
                self.num_envs, 1)
        reset_pose[..., :3] += self.env.scene.env_origins
        self.env.scene[f"{self.hand_side}_hand_object"].write_root_pose_to_sim(
            reset_pose, env_ids=self.env_ids)

    def reset_env(self, ):
        demo_obs = self.raw_data[f"demo_{self.demo_index}"]["obs"]
        self.reset_robot_joints(demo_obs, "right")
        self.reset_robot_joints(demo_obs, "left")

        self.demo_action = torch.as_tensor(
            np.array(self.raw_data[f"demo_{self.demo_index}"]["actions"])).to(
                self.device)
        self.init_data_buffer()

        self.env.reset()
        self.reset_rigid_objects(demo_obs)

        for i in range(10):
            if self.use_delta_pose:
                self.reset_robot_joints(demo_obs, "right")
                self.reset_robot_joints(demo_obs, "left")
                obs, rewards, terminated, time_outs, extras = self.env.step(
                    torch.zeros(self.env.action_space.shape).to(self.device))
            else:

                obs, rewards, terminated, time_outs, extras = self.env.step(
                    self.demo_action[0].to(
                        self.device).unsqueeze(0).repeat_interleave(
                            self.num_envs, dim=0))

        return obs

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

    def lift_or_not(self, index=None):
        target_object_state = self.env.scene[
            self.target_object_name].data.root_state_w[..., :7]
        success_flag = target_object_state[:, 2] > 0.3

        if success_flag.sum() > 0:
            if self.args_cli.save_path is not None:
                if index is None:
                    index = torch.nonzero(success_flag, as_tuple=True)[0]

                self.filter_out_data(index)
        return success_flag

    def replay(self, ):

        self.reset_env()

        for action in self.demo_action:
            raw_action = action.unsqueeze(0).repeat_interleave(self.num_envs,
                                                               dim=0).clone()
            if self.args_cli.normalize_action:
                raw_action = self.collector_interface.unnormalize(
                    raw_action,
                    self.collector_interface.action_stats["action"])
            raw_action = math_utils.denormalize_action(
                action.unsqueeze(0).clone(), self.action_bound,
                self.env.step_dt)
            raw_action[:, -self.num_hand_joints:] /= self.env.step_dt

            obs, rewards, terminated, time_outs, extras = self.env.step(
                raw_action)
            done = terminated | time_outs

            self.obs_buffer.append(obs)
            self.actions_buffer.append(
                action.unsqueeze(0).repeat_interleave(self.num_envs, dim=0))
            self.does_buffer.append(done)
            self.rewards_buffer.append(rewards)
        sucess_flag = self.lift_or_not(index=[0])
        self.demo_index += 1

    def open_loop_policy(self, policy):
        last_obs = self.reset_env()

        actions = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["actions"]
        actions = np.array(actions)
        obs_buffer = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["obs"]

        total_frame = actions.shape[0]

        for i in range(total_frame):

            obs_dict = {}
            for key in obs_buffer.keys():

                obs_dict[key] = torch.as_tensor(obs_buffer[key][i]).to(
                    self.device)

            if self.args_cli.normalize_action:
                noramlized_action = policy(obs_dict)
                action = self.collector_interface.unnormalize(
                    noramlized_action,
                    self.collector_interface.action_stats["action"])
            else:
                action = policy(obs_dict)

            action = torch.as_tensor(action).to(self.device)

            raw_action = math_utils.denormalize_action(
                action.unsqueeze(0).clone(), self.action_bound,
                self.env.step_dt)
            raw_action[:, -self.num_hand_joints:] /= self.env.step_dt

            next_obs, rewards, terminated, time_outs, extras = self.env.step(
                raw_action)

            last_obs = copy.deepcopy(next_obs)

        self.demo_index += 1

        sucess_flag = self.lift_or_not()
        return sucess_flag

    def close_loop_policy(
        self,
        policy,
    ):

        last_obs = self.reset_env()

        # actions = self.collector_interface.raw_data["data"][
        #     f"demo_{self.demo_index}"]["actions"]
        # actions = np.array(actions)
        obs_buffer = self.collector_interface.raw_data["data"][
            f"demo_{self.demo_index}"]["obs"]

        # total_frame = actions.shape[0]
        total_frame = 120

        for i in range(total_frame):
            # if i % 3 == 0:

            #     last_obs["policy"]["right_hand_joint_pos"] = torch.as_tensor(
            #         np.array(obs_buffer["right_hand_joint_pos"][i])).to(
            #             self.device).unsqueeze(0)

            if self.args_cli.normalize_action:
                noramlized_action = policy(last_obs["policy"])
                action = self.collector_interface.unnormalize(
                    noramlized_action,
                    self.collector_interface.action_stats["action"])
            else:
                action = policy(last_obs["policy"])

            action = torch.as_tensor(action).to(self.device)
            raw_action = math_utils.denormalize_action(
                action.unsqueeze(0).clone(), self.action_bound,
                self.env.step_dt)
            raw_action[:, -self.num_hand_joints:] /= self.env.step_dt

            next_obs, rewards, terminated, time_outs, extras = self.env.step(
                raw_action)

            last_obs = copy.deepcopy(next_obs)

        self.demo_index += 1

        sucess_flag = self.lift_or_not()
        return sucess_flag
