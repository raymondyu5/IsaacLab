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
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper

from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer, filter_out_data


class StateBCDiffusionWrapper(ZarrReplayWrapper):

    def __init__(self, env, env_cfg, args_cli, replay_env=None):

        self.env = env
        self.args_cli = args_cli
        self.env_cfg = env_cfg
        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.device = env.unwrapped.device
        self.num_envs = env.unwrapped.num_envs
        self.use_delta_pose = False if "Rel" not in self.args_cli.task else True
        self.hand_side = "right" if self.add_right_hand else "left"
        self.task = "place" if "Place" in self.args_cli.task else "grasp"

        self.target_object_name = f"{self.hand_side}_hand_object"
        self.demo_index = 3

        self.num_arm_actions = 6
        self.action_framework = args_cli.action_framework

        self.load_diffusion_model()
        super().__init__(
            env,
            env_cfg,
            args_cli,
            zarr_cfg=self.zarr_cfg,
        )

        self.temporal_action_buffer = TemporalEnsembleBufferAction(
            num_envs=self.env.unwrapped.num_envs,
            horizon_K=self.policy.horizon,
            action_dim=self.action_dim,
        )
        self.temporal_obs_buffer = TemporalEnsembleBufferObservation(
            num_envs=self.env.unwrapped.num_envs,
            horizon_K=self.policy.n_obs_steps,
            obs_dim=self.obs_dim,
        )

    def load_diffusion_model(self):

        checkpoint = os.path.join(
            self.args_cli.diffusion_path, "checkpoints",
            f"{self.args_cli.diffusion_checkpoint}.ckpt")

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

        cfg = payload['cfg']

        cfg.policy.num_inference_steps = 3
        cfg._target_ = "scripts.workflows.hand_manipulation.utils.diffusion.train_cfm_lowdim_policy.TrainCFMUnetLowdimWorkspace"
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
        self.obs_key = cfg.dataset.obs_key
        self.image_key = cfg.dataset.image_key

        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim

        self.add_time_stamp = cfg.dataset.get("add_time_stamp", False)

        self.zarr_cfg = cfg

    def get_demo_obs(self, index):
        obs_demo = []

        if self.add_time_stamp:
            obs_demo.append([index])

        for key in self.obs_key:

            obs_demo.append(self.raw_data[key][index])

        obs_demo = torch.tensor(np.concatenate(obs_demo,
                                               axis=0)).to(self.device)

        self.temporal_obs_buffer.add_obs(index, obs_demo)

    def reset(self):
        next_obs, _ = self.env.unwrapped.reset()
        reset_buffer(self)

        distractor_name = self.env_cfg["params"].get("distractor_name", [])
        init_rigid_object_pose = self.env.unwrapped.scene[
            f"{self.hand_side}_hand_object"]._data.reset_root_state
        if len(distractor_name) > 0:
            for distractor in distractor_name:

                dist = 100
                while True:

                    distractor_root_state = self.env.unwrapped.scene[
                        distractor]._data.reset_root_state
                    dist = torch.linalg.norm(distractor_root_state[..., :2] -
                                             init_rigid_object_pose[..., :2],
                                             dim=-1)
                    if dist > 0.20:
                        self.env.unwrapped.scene[
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

        for i in range(10):
            self.reset_robot_joints()

            if self.use_delta_pose:

                actions = torch.zeros(self.env.unwrapped.action_space.shape,
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
            next_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
                actions)

        return next_obs

    def open_loop_evaluate(self):

        last_obs = self.reset_env()
        self.temporal_obs_buffer.reset(self.raw_data["actions"].shape[0],
                                       self.env.unwrapped.num_envs)

        self.temporal_action_buffer.reset(self.raw_data["actions"].shape[0],
                                          self.env.unwrapped.num_envs)

        print("open_loop_evaluate")
        demo_action = self.raw_data["actions"]
        with torch.no_grad():

            for i in range(demo_action.shape[0]):
                self.get_demo_obs(i)
                obs_chunk = self.temporal_obs_buffer.compute_obs().clone()
                obs_dict = {
                    "obs": obs_chunk,
                }

                # if i > 30:

                # obs_dict = self.get_diffusion_obs(last_obs["policy"])

                predict_action = self.policy.predict_action(
                    obs_dict)["action_pred"]
                # for _ in range(predict_action.shape[1]):

                self.temporal_action_buffer.add_prediction(i, predict_action)
                # hand_action = self.temporal_action_buffer.compute_action()
                # next_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
                #     predict_action[:, 0, :])
                # last_obs = copy.deepcopy(next_obs)
                next_obs, rewards, terminated, time_outs, extras = self.env.step(
                    torch.as_tensor(demo_action[i]).to(
                        self.device).unsqueeze(0).repeat_interleave(
                            self.num_envs, dim=0))

        self.demo_index += 1
        success = self.evaluate_success(next_obs)
        return success

    def get_eval_obs(self, obs, index):
        obs_demo = []

        if self.add_time_stamp:
            obs_demo.append(
                torch.as_tensor([index]).unsqueeze(0).to(self.device))

        for key in self.obs_key:

            # if key in []:
            #     obs_demo.append(
            #         torch.as_tensor(self.raw_data[key][index]).unsqueeze(0).to(
            #             self.device))
            # else:

            obs_demo.append(obs[key])

        obs_demo = torch.cat(obs_demo, dim=1)

        self.temporal_obs_buffer.add_obs(index, obs_demo)
        # for key in self.obs_key:

        #     obs_demo.append(self.raw_data[key][index])
        # obs_demo = torch.tensor(np.concatenate(obs_demo,
        #                                        axis=0)).to(self.device)

        # self.temporal_obs_buffer.add_obs(index, obs_demo)

    # def close_loop_evaluate(self):

    # self.temporal_obs_buffer.reset(180, self.env.unwrapped.num_envs)
    # self.temporal_action_buffer.reset(180, self.env.unwrapped.num_envs)

    # last_obs = self.reset()
    # # last_obs = self.reset_env()
    # self.image_buffer = []

    # print("close_loop_evaluate")
    # with torch.no_grad():

    #     for i in range(180):
    #         self.get_eval_obs(last_obs["policy"], i)
    #         obs_chunk = self.temporal_obs_buffer.compute_obs().clone()

    #         obs_dict = {
    #             "obs": obs_chunk,
    #         }

    #         predict_action = self.policy.predict_action(
    #             obs_dict)["action_pred"]

    #         self.temporal_action_buffer.add_prediction(i, predict_action)

    #         next_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
    #             predict_action[:, 0])
    #         last_obs = copy.deepcopy(next_obs)

    # self.demo_index += 1
    # return self.env.unwrapped.scene[
    #     f"{self.hand_side}_hand_object"].data.root_state_w[..., 2] > 0.3

    def close_loop_evaluate(self):

        last_obs = self.reset()
        # last_obs = self.reset_env()

        self.last_diffusion_obs = self.get_diffusion_obs(last_obs["policy"])

        self.image_buffer = []

        print("close_loop_evaluate")

        with torch.no_grad():

            for i in range(160):
                next_obs, rewards, terminated, time_outs, extras, predict_action = self.step_diffusion(
                )

                update_buffer(
                    self,
                    next_obs,
                    last_obs,
                    predict_action,
                    rewards,
                    terminated,
                    time_outs,
                )

                last_obs = copy.deepcopy(next_obs)

        self.demo_index += 1

        success = self.evaluate_success(self)
        # index = torch.nonzero(success, as_tuple=True)[0]
        # if self.args_cli.save_path is not None and len(index) > 0:
        #     filter_out_data(self, index.tolist(), save_data=True)

        return success

    def step_diffusion(self):

        predict_action = self.policy.predict_action(
            self.last_diffusion_obs)["action_pred"]
        rollout_action = predict_action[:, 0, :]

        if self.args_cli.revert_action:

            if "Rel" in self.args_cli.task:
                link_ee_pos = self.env.unwrapped.scene[
                    f"{self.hand_side}_panda_link7"]._data.root_state_w[:, :7]
                link_ee_pos[:, :3] -= self.env.unwrapped.scene.env_origins
                delta_pose = math_utils.extract_delta_pose(
                    rollout_action, link_ee_pos)
                rollout_action = torch.cat(
                    [delta_pose, rollout_action[:, -self.num_hand_joint:]],
                    dim=-1)

        next_obs, rewards, terminated, time_outs, extras = self.env.unwrapped.step(
            rollout_action)

        self.last_diffusion_obs = self.get_diffusion_obs(next_obs["policy"])

        return next_obs, rewards, terminated, time_outs, extras, rollout_action
