from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import torch
import numpy as np
from scripts.workflows.open_policy.utils.buffer_utils import h5py_group_to_dict
import isaaclab.utils.math as math_utils
import copy

import imageio
import sys

sys.path.append("submodule/stable-baselines3/")

from scripts.workflows.open_policy.task.BCPPOBuffer import OnlineBCBuffer


class BCReplayDatawrapper:

    def __init__(
        self,
        env,
        env_cfg,
        args_cli,
        reload_data: bool = False,
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
        bc_args_cli = copy.deepcopy(args_cli)
        bc_args_cli.log_dir = args_cli.bc_dir

        self.collector_interface = MultiDatawrapper(
            bc_args_cli,
            env_cfg,
            save_path=bc_args_cli.save_path,
            load_path=bc_args_cli.load_path,
        )
        self.raw_data = self.collector_interface.raw_data["data"]

        self.init_reload_data_buffer()
        self.init_setting()

        if reload_data:

            self.init_reload_data()
            self.step = self.step_reload_data
        else:
            self.step = self.step_data

    def step_data(self):
        obs_buffer = []
        next_obs_buffer = []
        actions_buffer = []
        rew_buffer = []
        next_obs_buffer = []
        does_buffer = []

        for index, demo_name in enumerate(self.raw_data.keys()):
            obs = self.raw_data[demo_name]["obs"]
            actions = self.raw_data[demo_name]["actions"]
            rewards = self.raw_data[demo_name]["rewards"]
            dones = self.raw_data[demo_name]["dones"]

            obs_buffer.append(self.process_obs(obs, numpy_already=True)[None])
            next_obs_buffer.append(
                self.process_obs(obs, numpy_already=True)[None])
            actions_buffer.append(np.array(actions)[None])
            rew_buffer.append(np.array(rewards)[None])
            does_buffer.append(np.array(dones)[None])

        self.rollout_buffer = OnlineBCBuffer(
            self.env,
            rewards.shape[0],
            self.env.observation_space["policy"],
            self.env.action_space,
            self.device,
            rewards.shape[0],
        )
        all_obs = np.concatenate(obs_buffer, 0).transpose(1, 0, 2)
        all_next_obs = np.concatenate(next_obs_buffer, 0).transpose(1, 0, 2)
        all_actions = np.concatenate(actions_buffer, 0).transpose(1, 0, 2)
        all_does = np.concatenate(does_buffer, 0).transpose(1, 0)
        all_rews = np.concatenate(rew_buffer, 0).transpose(1, 0)
        terminates = np.zeros_like(all_rews)
        terminates[:, -1] = 1

        self.rollout_buffer.add(all_obs, all_next_obs, all_actions, all_rews,
                                terminates, all_does)

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
            self.all_actions[num_actions:, 2] = 0.5
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

        return self.process_obs(obs)

    def update_buffer(self, obs_buffer, next_obs_buffer, actions_buffer,
                      rew_buffer, does_buffer, success_or_not):

        obs = np.concatenate(obs_buffer, 0).transpose(1, 0, 2)[success_or_not]
        next_obs = np.concatenate(next_obs_buffer,
                                  0).transpose(1, 0, 2)[success_or_not]

        dones = np.concatenate(does_buffer, 0).transpose(
            1, 0)[success_or_not].astype(np.float16)
        actions = np.concatenate(actions_buffer,
                                 0).transpose(1, 0, 2)[success_or_not]

        rews = np.concatenate(rew_buffer, 0).transpose(1, 0)[success_or_not]

        terminates = np.zeros_like(rews)
        terminates[:, -1] = 1

        self.rollout_buffer = OnlineBCBuffer(
            self.env,
            rews.shape[0],
            self.env.observation_space["policy"],
            self.env.action_space,
            self.device,
            rews.shape[0],
        )

        self.rollout_buffer.add(obs, next_obs, actions, rews, terminates,
                                dones)

    def process_obs(
        self,
        obs_dict: torch.Tensor | dict[str, torch.Tensor],
        concatenate_obs: bool = True,
        numpy_already: bool = False,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        # Sb3 doesn't support asymmetric observation spaces, so we only use "policy"

        obs = obs_dict["policy"] if "policy" in obs_dict else obs_dict

        if not concatenate_obs:
            dict_obs = dict()

            for key, value in obs.items():

                dict_obs[key] = value.cpu().numpy()
            return dict_obs

        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            if numpy_already:
                obs = np.concatenate([v for v in obs.values()], axis=-1)
            else:

                obs = torch.cat([v for v in obs.values()],
                                dim=-1).detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):

            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")

        return obs

    def lift_or_not(self, ):
        target_object_state = self.env.scene[
            f"{self.hand_side}_hand_object"].data.root_state_w[..., :7]
        success_flag = target_object_state[:, 2] > 0.20

        return success_flag

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

            self.rew_buffer.append(rewards[None].cpu().numpy())

            new_obs = self.process_obs(new_obs)

            self.obs_buffer.append(last_obs[None])
            self.actions_buffer.append(action[None].cpu().numpy())
            self.next_obs_buffer.append(copy.deepcopy(new_obs[None]))
            self.does_buffer.append(done[None].cpu().numpy())

            last_obs = copy.deepcopy(new_obs)

        success_or_not = self.lift_or_not()
        success_rate = success_or_not.sum().float() / success_or_not.shape[0]

        print("Success rate of the demonstration: ", success_rate)

        self.update_buffer(self.obs_buffer, self.next_obs_buffer,
                           self.actions_buffer, self.rew_buffer,
                           self.does_buffer,
                           success_or_not.cpu().numpy())

    # def test_demo(self, test_env):
    #     print("Testing demo")
    #     self.reset()
    #     test_env.reset()
    #     self.reset()

    #     for action in self.actions_buffer:

    #         new_obs, rew, terminate, timeouts, infos, _ = test_env.step(
    #             torch.as_tensor(action[0]).to(self.device))

    #     height = self.env.scene[
    #         self.target_manipulated_object].data.root_state_w[:, 2]
    #     success_or_not = height > 0.20
    #     success_rate = success_or_not.sum().float() / success_or_not.shape[0]

    #     print("Success rate of the demonstration: ", success_rate)
