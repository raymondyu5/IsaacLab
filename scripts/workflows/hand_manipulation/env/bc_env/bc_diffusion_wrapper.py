from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper

import numpy as np

from scripts.workflows.hand_manipulation.env.rl_env.replay_rl_wrapper import ReplayRLWrapper
import sys
import os

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    TemporalEnsembleBufferAction, TemporalEnsembleBufferObservation)
import torch
import copy


class BCDiffusionWrapper(ReplayRLWrapper):

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
        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_cfg,
            save_path=args_cli.save_path,
            load_path=args_cli.load_path,
        )

        self.demo_index = self.collector_interface.traj_count
        # self.collector_interface.init_collector_interface(
        #     save_path=f"demo_{self.demo_index}")
        self.num_arm_actions = 6

        self.init_setting()
        super().__init__(
            env,
            env_cfg,
            args_cli,
        )
        self.load_diffusion_model()
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

    def load_diffusion_model(self):

        checkpoint = os.path.join(self.args_cli.diffusion_path, "checkpoints",
                                  "latest.ckpt")

        payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

        cfg = payload['cfg']

        cfg.policy.num_inference_steps = 3
        cfg._target_ = "scripts.workflows.hand_manipulation.utils.diffusion.policy.train_cfm_unet_lowdim_policy.CFMnetLowdimPolicy"
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
        self.obs_dim = cfg.obs_dim
        self.action_dim = cfg.action_dim

    def get_demo_obs(self, obs, index):
        obs_demo = []

        for key in obs.keys():
            if key in self.obs_key:

                obs_demo.append(obs[key][index])
        obs_demo = torch.tensor(np.concatenate(obs_demo,
                                               axis=0)).to(self.device)

        self.temporal_obs_buffer.add_obs(index, obs_demo)

    def reset(self):
        next_obs, _ = self.env.reset()
        for i in range(20):
            actions = torch.zeros(self.env.action_space.shape,
                                  dtype=torch.float32,
                                  device=self.device)
            next_obs, rewards, terminated, time_outs, extras = self.env.step(
                actions)

        return next_obs

    def open_loop_evaluate(self):

        last_obs = self.reset_env()
        obs = self.raw_data[f"demo_{self.demo_index}"]["obs"]
        self.temporal_obs_buffer.reset(self.demo_action.shape[0],
                                       self.env.num_envs)
        self.temporal_action_buffer.reset(self.demo_action.shape[0],
                                          self.env.num_envs)
        print("open_loop_evaluate")
        with torch.no_grad():

            for i in range(self.demo_action.shape[0]):
                self.get_demo_obs(obs, i)
                obs_chunk = self.temporal_obs_buffer.compute_obs().clone()
                obs_dict = {
                    "obs": obs_chunk,
                }
                predict_action = self.policy.predict_action(
                    obs_dict)["action_pred"]
                # for _ in range(predict_action.shape[1]):

                self.temporal_action_buffer.add_prediction(i, predict_action)
                hand_action = self.temporal_action_buffer.compute_action()
                # self.env.step(predict_action[:, 0, :])
                self.env.step(hand_action)

        self.demo_index += 1

    def get_eval_obs(self, obs, index):
        obs_demo = []

        for key in self.obs_key:

            obs_demo.append(obs[key])

        obs_demo = torch.cat(obs_demo, dim=1)

        self.temporal_obs_buffer.add_obs(index, obs_demo)

    def close_loop_evaluate(self):

        self.temporal_obs_buffer.reset(180, self.env.num_envs)
        self.temporal_action_buffer.reset(180, self.env.num_envs)
        last_obs = self.reset()

        print("close_loop_evaluate")
        with torch.no_grad():

            for i in range(180):
                self.get_eval_obs(last_obs["policy"], i)
                obs_chunk = self.temporal_obs_buffer.compute_obs().clone()
                obs_dict = {
                    "obs": obs_chunk,
                }
                predict_action = self.policy.predict_action(
                    obs_dict)["action_pred"]

                self.temporal_action_buffer.add_prediction(i, predict_action)
                hand_action = self.temporal_action_buffer.compute_action()
                # next_obs, rewards, terminated, time_outs, extras = self.env.step(
                #     hand_action)
                next_obs, rewards, terminated, time_outs, extras = self.env.step(
                    predict_action[:, 0])

                last_obs = copy.deepcopy(next_obs)
        self.demo_index += 1

        object_pose = self.env.scene[
            f"{self.hand_side}_hand_object"]._data.root_state_w[:, :3]
        object_pose[:, :3] = self.env.scene.env_origins
        lift_or_not = (object_pose[:, 2] > 0.30)
        overhigh_or_not = (object_pose[:, 2] < 0.60)
        outofbox_or_not = ((object_pose[:, 0] < 0.65) &
                           (object_pose[:, 0] > 0.3) &
                           (object_pose[:, 1] < 0.3) &
                           (object_pose[:, 1] > -0.3))
        success = lift_or_not & overhigh_or_not & outofbox_or_not

        return success
