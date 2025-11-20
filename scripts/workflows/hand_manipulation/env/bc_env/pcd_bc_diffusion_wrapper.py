import numpy as np

from scripts.workflows.hand_manipulation.env.bc_env.zarr_replay_env_wrapper import ZarrReplayWrapper
import sys
import os
from torchvision import transforms

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    TemporalEnsembleBufferAction, TemporalEnsembleBufferObservation,
    TemporalEnsembleImageObservation)
import torch
import copy

from isaaclab.envs import mdp
from isaaclab.managers import SceneEntityCfg
from tools.visualization_utils import vis_pc, visualize_pcd

import isaaclab.utils.math as math_utils
from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer, filter_out_data


class PCDBCDiffusionWrapper(ZarrReplayWrapper):

    def __init__(self, env, env_cfg, args_cli, replay_env=None):

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
        self.demo_index = 0
        # if args_cli.save_path is not None:

        self.num_arm_actions = 6

        self.load_diffusion_model()
        super().__init__(env,
                         env_cfg,
                         args_cli,
                         zarr_cfg=self.zarr_cfg,
                         num_pcd=self.image_dim[-1])

        self.temporal_action_buffer = TemporalEnsembleBufferAction(
            num_envs=self.env.num_envs,
            horizon_K=self.policy.horizon,
            action_dim=self.action_dim,
        )
        self.temporal_obs_buffer = TemporalEnsembleBufferObservation(
            num_envs=self.env.num_envs,
            horizon_K=self.policy.n_obs_steps,
            obs_dim=self.obs_dim,
        )
        self.temporal_image_buffer = TemporalEnsembleImageObservation(
            num_envs=self.env.num_envs,
            horizon_K=self.policy.n_obs_steps,
            obs_dim=self.obs_dim,
        )
        if self.args_cli.analysis:
            self.contruct_meshgrid_transforms()

    def contruct_meshgrid_transforms(self):

        object_range = self.env_cfg["params"]["multi_cluster_rigid"][
            f"{self.hand_side}_hand_object"]["pose_range"]
        range_list = [
            object_range.get(key, (0.0, 0.0))
            for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]

        N, M, K = 3, 3, 3  # number of splits you want

        x_range = torch.linspace(-0.0, 0.10, N)
        y_range = torch.linspace(-0.05, 0.05, M)
        yaw_range = torch.linspace(-0.5, 0.5, K)

        # Create 3D meshgrid
        X, Y, Yaw = torch.meshgrid(x_range, y_range, yaw_range, indexing="ij")

        # Flatten into coordinate triples
        self.object_coords = torch.stack(
            [X.flatten(), Y.flatten(), Yaw.flatten()], dim=1).to(self.device)
        default_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.default_root_state[0, :2]

        self.object_coords[..., :2] += default_pose[:2]

    def load_diffusion_model(self):

        checkpoint = os.path.join(
            self.args_cli.diffusion_path, "checkpoints",
            f"{self.args_cli.diffusion_checkpoint}.ckpt")

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

        cfg = payload['cfg']

        cfg.policy.num_inference_steps = 3
        cfg._target_ = "scripts.workflows.hand_manipulation.utils.diffusion.train_cfm_pcd_policy.TrainCFMUnetPCDWorkspace"
        cls = hydra.utils.get_class(cfg._target_)

        workspace = cls(cfg, )
        workspace: BaseWorkspace
        workspace.load_payload(payload, exclude_keys=None, include_keys=None)

        # get policy from workspace
        self.policy = workspace.model
        if cfg.training.use_ema:
            self.policy = workspace.ema_model

        device = torch.device(self.device)
        self.policy.to(device)
        self.policy.eval()

        self.chunk_size = self.policy.n_obs_steps

        self.obs_key = [f"{self.hand_side}_hand_joint_pos", f"{self.hand_side}_ee_pose"]
        cfg.dataset.obs_key = self.obs_key

        self.image_key = ["seg_pc"]
        self.obs_dim = cfg.shape_meta.obs.agent_pos.shape[0]
        self.action_dim = cfg.shape_meta.action.shape[0]

        self.image_dim = cfg.shape_meta.obs.seg_pc.shape
        cfg.dataset.image_key = self.image_key

        self.zarr_cfg = cfg

    def get_demo_obs(self, index):
        obs_demo = []

        for key in self.obs_key:

            obs_demo.append(self.raw_data[key][index])
        obs_demo = torch.tensor(np.concatenate(obs_demo,
                                               axis=0)).to(self.device)

        # for key in self.obs_key:

        #     obs_demo.append(self.raw_data[key][index])
        # obs_demo = torch.tensor(np.concatenate(obs_demo,
        #                                        axis=0)).to(self.device)

        self.temporal_obs_buffer.add_obs(index, obs_demo)

        image_demo = []
        for key in self.image_key:
            image_demo.append(self.raw_data[key][index])

        image_demo = torch.tensor(np.concatenate(image_demo,
                                                 axis=0)).to(self.device)
        self.temporal_image_buffer.add_obs(index, image_demo)

    def reset(self, analysis_mode=False):
        self.env.reset()

        distractor_name = self.env_cfg["params"].get("distractor_name", [])
        init_rigid_object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.reset_root_state
        if len(distractor_name) > 0:
            for distractor in distractor_name:

                dist = 100
                while True:

                    distractor_root_state = self.env.scene[
                        distractor]._data.reset_root_state
                    dist = torch.linalg.norm(distractor_root_state[..., :2] -
                                             init_rigid_object_pose[..., :2],
                                             dim=-1)
                    if dist > 0.20:
                        self.env.scene[
                            f"{self.hand_side}_hand_object"].write_root_pose_to_sim(
                                init_rigid_object_pose[..., :7],
                                env_ids=self.env_ids)
                        break

                    mdp.reset_root_state_uniform(
                        self.env,
                        self.env_ids,
                        pose_range=self.env_cfg["params"]["RigidObject"]
                        [distractor]["pose_range"],
                        velocity_range={},
                        asset_cfg=SceneEntityCfg(distractor))

        if analysis_mode:

            reset_pose = self.env.scene[
                f"{self.hand_side}_hand_object"]._data.reset_root_state.clone(
                )
            reset_pose[..., :2] = self.object_coords[self.demo_index][
                ..., :2] + self.env.scene.env_origins[:, :2]

            reset_pose[..., 3:7] = math_utils.quat_from_euler_xyz(
                torch.zeros(1).to(self.device),
                torch.zeros(1).to(self.device),
                torch.as_tensor(
                    self.object_coords[self.demo_index][2])).repeat(
                        self.env.num_envs, 1)
            self.env.scene[
                f"{self.hand_side}_hand_object"].write_root_pose_to_sim(
                    reset_pose[..., :7], env_ids=self.env_ids)

        for i in range(20):
            self.reset_robot_joints()
            if self.use_delta_pose:
                actions = torch.zeros(self.env.action_space.shape,
                                      dtype=torch.float32,
                                      device=self.device)
            else:

              
                actions = torch.as_tensor(
                    self.env_cfg["params"].get("init_ee_pose")).to(
                        self.device).unsqueeze(0).repeat_interleave(
                            self.env.unwrapped.num_envs, dim=0)
                actions = torch.concat([
                    actions,
                    torch.zeros((self.env.unwrapped.num_envs, 16),
                                dtype=torch.float32,
                                device=self.device)
                ],
                                       dim=-1)
            next_obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)
        # self.finger_lpfilter.reset()
        # self.finger_lpfilter(
        #     torch.zeros(self.env.num_envs, 16).to(self.device))
        self.init_object_height = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, 2]

        self.image_buffer = []

        return next_obs

    def open_loop_evaluate(self):

        self.reset_env()
        self.temporal_obs_buffer.reset(self.raw_data["actions"].shape[0],
                                       self.env.num_envs)

        self.temporal_image_buffer.reset(self.raw_data["seg_pc"].shape[0],
                                         self.raw_data["seg_pc"].shape[1:],
                                         self.env.num_envs)
        self.temporal_action_buffer.reset(self.raw_data["actions"].shape[0],
                                          self.env.num_envs)

        print("open_loop_evaluate")
        demo_action = self.raw_data["actions"]

        with torch.no_grad():

            for i in range(demo_action.shape[0]):
                self.get_demo_obs(i)
                obs_chunk = self.temporal_obs_buffer.compute_obs().clone()
                image_chunk = self.temporal_image_buffer.compute_obs().clone()

                obs_dict = {
                    "agent_pos": obs_chunk,
                    "seg_pc": image_chunk,
                }
                predict_action = self.policy.predict_action(
                    obs_dict)["action_pred"]
                # for _ in range(predict_action.shape[1]):

                self.temporal_action_buffer.add_prediction(i, predict_action)
                # hand_action = self.temporal_action_buffer.compute_action()
                # self.env.step(predict_action[:, 0, :])
                # import pdb
                # pdb.set_trace()
              
                next_obs, rewards, terminated, time_outs, extras = self.env.step(
                    torch.as_tensor(demo_action[i]).to(
                        self.device).unsqueeze(0).repeat_interleave(
                            self.num_envs, dim=0))

        self.demo_index += 1
        return self.evaluate_success(next_obs)

    def get_eval_obs(self, obs, index):
        obs_demo = []

        for key in self.obs_key:

            obs_data = obs[key].clone()

            obs_demo.append(obs_data)

        obs_demo = torch.cat(obs_demo, dim=1)

        # for key in self.obs_key:

        #     obs_demo.append(self.raw_data[key][index])
        # obs_demo = torch.tensor(np.concatenate(obs_demo,
        #                                        axis=0)).to(self.device)

        self.temporal_obs_buffer.add_obs(index, obs_demo)

        image_demo = []
        for key in self.image_key:

            seg_pcd = obs[key]
            sampled_pcd = []
            for pcd in seg_pcd:

                points_index = torch.randperm(pcd.shape[-2]).to(self.device)

                sampled_pcd.append(pcd[:, points_index[:self.image_dim[-1]]])
            sampled_pcd = torch.cat(sampled_pcd, dim=0)
            image_demo.append(sampled_pcd)

        image_demo = torch.cat(image_demo, dim=0).permute(0, 2, 1)
        self.temporal_image_buffer.add_obs(index, image_demo)

    # def close_loop_evaluate(self):

    #     self.temporal_obs_buffer.reset(160, self.env.num_envs)
    #     self.temporal_action_buffer.reset(160, self.env.num_envs)
    #     self.temporal_image_buffer.reset(160, self.image_dim,
    #                                      self.env.num_envs)
    #     last_obs = self.reset()
    #     # last_obs = self.reset_env()
    #     self.image_buffer = []

    #     print("close_loop_evaluate")
    #     with torch.no_grad():

    #         for i in range(160):
    #             self.get_eval_obs(last_obs["policy"], i)
    #             obs_chunk = self.temporal_obs_buffer.compute_obs().clone()
    #             image_chunk = self.temporal_image_buffer.compute_obs().clone()

    #             obs_dict = {
    #                 "agent_pos": obs_chunk,
    #                 "seg_pc": image_chunk,
    #             }

    #             predict_action = self.policy.predict_action(
    #                 obs_dict)["action_pred"]

    #             self.temporal_action_buffer.add_prediction(i, predict_action)
    #             # hand_action = self.temporal_action_buffer.compute_action()
    #             # next_obs, rewards, terminated, time_outs, extras = self.env.step(
    #             #     hand_action)

    #             if not self.use_delta_pose:

    #                 ee_link_pose = self.env.scene[
    #                     f"{self.hand_side}_panda_link7"]._data.root_state_w.clone(
    #                     )
    #                 ee_link_pose[:, :3] -= self.env.scene.env_origins
    #                 xyz_pose, quat_pose = math_utils.apply_delta_pose(
    #                     ee_link_pose[:, :3], ee_link_pose[:, 3:7],
    #                     predict_action[:, 0, :6])

    #                 action = torch.cat([
    #                     xyz_pose, quat_pose,
    #                     predict_action[:, 0, -self.num_hand_joint:]
    #                 ],
    #                                    dim=-1).to(self.device)

    #             else:
    #                 action = predict_action[:, 0]

    #             next_obs, rewards, terminated, time_outs, extras = self.env.step(
    #                 action)
    #             try:

    #                 self.image_buffer.append(
    #                     last_obs["policy"]["rgb"][:, 0].cpu().numpy())
    #             except:
    #                 pass

    #             last_obs = copy.deepcopy(next_obs)
    #             # o3d = vis_pc(image_chunk[0, 0, :3].transpose(0,
    #             #                                              1).cpu().numpy())
    #             # # o3d = vis_pc(last_obs["policy"]["seg_pc"][0, :, :3].cpu().numpy())
    #             # visualize_pcd([o3d])
    #     self.demo_index += 1
    #     return self.env.scene[
    #         f"{self.hand_side}_hand_object"].data.root_state_w[..., 2] > 0.3

    def close_loop_evaluate(self, analysis_mode=False):

        last_obs = self.reset(analysis_mode)
        self.last_diffusion_obs = self.get_diffusion_obs(last_obs["policy"])

        # last_obs = self.reset_env()  # Commented out: zarr replay not needed for evaluation
        self.image_buffer = []
        reset_buffer(self)
        sparse_rewards = torch.zeros(self.num_envs).to(self.device)

        print("close_loop_evaluate")
        with torch.no_grad():

            for i in range(160):

                next_obs, rewards, terminated, time_outs, extras, predict_action = self.step_diffusion(
                )
                sparse_rewards += rewards
                if self.collector_interface is not None:
                    update_buffer(self,
                                  None,
                                  last_obs["policy"],
                                  predict_action[:, 0].clone(),
                                  rewards,
                                  terminated,
                                  time_outs,
                                  convert_to_cpu=False)

                try:

                    self.image_buffer.append(
                        last_obs["policy"]["rgb"][:, 0].cpu().numpy())
                except:
                    pass
                last_obs = copy.deepcopy(next_obs)

                # import pdb
                # pdb.set_trace()

                # o3d = vis_pc( last_obs["policy"]["seg_pc"][0][0, :, :3].cpu().numpy())
                # visualize_pcd([o3d])

        self.demo_index += 1
        object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, :3].clone()
        object_pose[:, :3] -= self.env.scene.env_origins

        if "Place" in self.args_cli.task:
            target_object_state = self.env.scene[
                f"{self.hand_side}_hand_place_object"].data.root_state_w[
                    ..., :7]
            pick_object_state = self.env.scene[
                f"{self.hand_side}_hand_object"].data.root_state_w[..., :7]
            success = torch.linalg.norm(target_object_state[:, :2] -
                                        pick_object_state[:, :2],
                                        dim=-1) < 0.10

        else:

            lift_or_not = (object_pose[:, 2] > 0.20)
            # overhigh_or_not = (object_pose[:, 2] < 0.60)
            # outofbox_or_not = ((object_pose[:, 0] < 0.65) &
            #                    (object_pose[:, 0] > 0.3) &
            #                    (object_pose[:, 1] < 0.3) &
            #                    (object_pose[:, 1] > -0.3))
            success = lift_or_not  #& overhigh_or_not & outofbox_or_not
        if self.collector_interface is not None:

            self.collector_interface.add_demonstraions_to_buffer(
                self.obs_buffer, self.action_buffer, self.rewards_buffer,
                self.does_buffer)

        return success

    def contruct_reward(self, ):

        ## object height reward

        object_height = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, 2]
        height_reward = torch.clip(
            (object_height - self.init_object_height - 0.02) / object_height,
            0, 1) * 2
        palm_pose = self.env.scene[
            f"{self.hand_side}_palm_lower"]._data.root_state_w

        ### palm height reward
        nonlift = height_reward < 0.05
        height_diff = torch.abs(palm_pose[:, 2] - object_height)

        max_dist = 0.2  # e.g., 10 cm
        norm_diff = torch.clamp(height_diff / max_dist, 0.0, 1.0)

        # convert to reward: 1 when aligned, 0 when far apart
        close_reward = (1.0 - norm_diff) * nonlift.float()
        close_reward = torch.where(nonlift, close_reward,
                                   torch.ones_like(close_reward))

        ### finger pose reward

        finger_action = self.env.scene[
            f"{self.hand_side}_hand"]._data.joint_pos[..., -16:].clone()

        # normalize finger closeness: assume joint range [0=open, 1=closed]
        finger_norm = torch.clamp(finger_action, 0.0, 1.0)

        # --- Distance thresholds ---
        # if palm is farther than 5 cm → encourage open fingers (reward 0 if closing)
        far_mask = (height_diff > 0.05)

        # if palm is close (<= 3 cm) → encourage closing
        close_mask = (height_diff < 0.03)

        # between 3–5 cm → interpolate linearly
        mid_mask = (~far_mask) & (~close_mask)

        # --- Finger reward ---
        # encourage open when far
        open_reward = (1.0 - finger_norm.mean(dim=-1)) * far_mask.float()

        # encourage close when very near
        close_reward_fingers = finger_norm.mean(dim=-1) * close_mask.float()

        # interpolate in mid zone
        mid_alpha = (0.05 - height_diff) / (0.05 - 0.03
                                            )  # goes 0→1 as dist 5→3 cm
        mid_alpha = torch.clamp(mid_alpha, 0.0, 1.0)
        mid_reward = (finger_norm.mean(dim=-1) * mid_alpha) * mid_mask.float()

        finger_reward = open_reward + close_reward_fingers + mid_reward

        #### total reward
        reward = height_reward + close_reward + finger_reward
        return reward

    def step_diffusion(self):

        predict_action = self.policy.predict_action(
            self.last_diffusion_obs)["action_pred"]
        robot_action = predict_action[:, 0].clone()
        if self.args_cli.use_relative_finger_pose:

            robot_action[:, -16:] += self.env.scene[
                f"{self.hand_side}_hand"]._data.joint_pos[:, -16:].clone()

        # robot_action[:, -16:] += (torch.rand(
        #     (self.env.num_envs, 16)).to(self.device) * 2 - 1) * 0.10

        # if self.env.unwrapped.episode_length_buf[
        #         0] > 90 and self.env.unwrapped.episode_length_buf[0] < 95:

        #     robot_action[:, -16:] -= 0.1

        next_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
            robot_action)
        object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, :3].clone()
        object_pose[:, :3] -= self.env.scene.env_origins
        rewards = (object_pose[:, 2] > 0.20)
        handmade_rewards = self.contruct_reward()

        self.last_diffusion_obs = self.get_diffusion_obs(next_obs["policy"])
        return next_obs, rewards, terminated, time_outs, extras, predict_action

    def analyze_evaluation(self, ):

        success = self.close_loop_evaluate(analysis_mode=True)
        return torch.mean(success.float()).item()
