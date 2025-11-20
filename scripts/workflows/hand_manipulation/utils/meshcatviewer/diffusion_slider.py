import numpy as np
import torch
from scripts.workflows.hand_manipulation.utils.meshcatviewer.button_wrapper import ButtonWrapper

import numpy as np

import torch

import sys
import os

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    extract_finger_joints, temporal_ensemble_finger_joints)
import time


class DiffusionSliderApp:

    def __init__(self,
                 raw_sliders,
                 root,
                 action_buffer=None,
                 diffusion_path=None,
                 hand_side="right",
                 device="cuda",
                 slider_function=None,
                 group_count=0,
                 num_slider=1):

        # Define joint names and dimensions
        self.raw_joint_names = [
            'j1', 'j0', 'j2', 'j3', 'j12', 'j13', 'j14', 'j15', 'j5', 'j4',
            'j6', 'j7', 'j9', 'j8', 'j10', 'j11'
        ]
        self.isaac_joint_names = [
            'j1', 'j12', 'j5', 'j9', 'j0', 'j13', 'j4', 'j8', 'j2', 'j14',
            'j6', 'j10', 'j3', 'j15', 'j7', 'j11'
        ]
        self.retarget2pin = [
            self.isaac_joint_names.index(name) for name in self.raw_joint_names
        ]
        self.num_hand_joints = len(self.raw_joint_names)
        self.diffusion_path = diffusion_path
        self.hand_side = hand_side
        self.device = device
        self.raw_sliders = raw_sliders
        self.root = root
        self.slider_function = slider_function
        self.action_buffer = action_buffer[:100]
        self.num_demos = len(self.action_buffer) if action_buffer else 0

        self.joint_limits = torch.as_tensor(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.46999997, 2.4429998], [-1.047, 1.047],
             [-1.047, 1.047], [-0.5059999, 1.8849999], [-1.2, 1.8999999],
             [-0.5059999, 1.8849999], [-0.5059999, 1.8849999],
             [-0.366, 2.0419998], [-1.34, 1.8799999], [-0.366, 2.0419998],
             [-0.366, 2.0419998]],
            dtype=torch.float32).to(self.device)

        self.group_count = self.load_diffusion_model(group_count)
        if self.action_buffer is not None:
            self.group_count = self.init_reconstruct_slides(
                self.group_count, num_slider)
        else:
            self.group_count = group_count

    def init_reconstruct_slides(self, group_count, num_slider):
        self.button_wrapper = ButtonWrapper(self.root,
                                            num_demos=self.num_demos,
                                            raw_sliders=self.raw_sliders)
        group_count = self.button_wrapper.create_demo_control_panel(
            group_count, num_slider, name="Diffusion Reconstruction")
        self.recontruct_diffusion_actions = [
            [] for _ in range(len(self.diffusion_models))
        ]

        for index, diffusion_model in enumerate(self.diffusion_models):
            with torch.no_grad():
                self.raw_dfifusion_actions = []

                for action_buffer in self.action_buffer:
                    start_time = time.time()
                    raw_hand_pose = torch.as_tensor(
                        action_buffer[..., -self.num_hand_joints:]).to(
                            self.device).to(dtype=torch.float32)

                    num_chunks = raw_hand_pose.shape[
                        0] - diffusion_model.horizon + 1
                    obs_list = [
                        raw_hand_pose[i:i + diffusion_model.horizon].reshape(
                            1, -1, self.num_hand_joints)
                        for i in range(0, num_chunks - 1)
                    ]

                    actions_chunks = [
                        raw_hand_pose[i:i + diffusion_model.horizon].reshape(
                            1, -1, self.num_hand_joints)
                        for i in range(1, num_chunks)
                    ]

                    obs_list = torch.cat(obs_list, dim=0).to(self.device)
                    actions_chunks = torch.cat(actions_chunks,
                                               dim=0).to(self.device)
                    obs_dict = {"obs": obs_list}

                    predict_action = temporal_ensemble_finger_joints(
                        extract_finger_joints(
                            diffusion_model.predict_action(obs_dict)
                            ["action_pred"], self.joint_limits))
                    actions_chunks = temporal_ensemble_finger_joints(
                        extract_finger_joints(actions_chunks,
                                              self.joint_limits))

                    self.recontruct_diffusion_actions[index].append(
                        predict_action.cpu().numpy())
                    self.raw_dfifusion_actions.append(
                        actions_chunks.cpu().numpy())

                    end_time = time.time()
                    print(
                        f"Diffusion reconstruction time: {end_time - start_time:.4f} seconds"
                    )

        return group_count

    def load_diffusion_model(self, group_count):
        self.diffusion_sliders = []
        self.diffusion_models = []
        self.diffusion_model_setting = []
        for index, diffusion_path in enumerate(self.diffusion_path):

            diffusion_sliders = []

            checkpoint = os.path.join(diffusion_path, "checkpoints",
                                      "latest.ckpt")

            payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

            cfg = payload['cfg']
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg, output_dir=diffusion_path)
            workspace: BaseWorkspace
            workspace.load_payload(payload,
                                   exclude_keys=None,
                                   include_keys=None)

            # get policy from workspace
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model

            device = torch.device(self.device)
            policy.to(device)
            policy.eval()
            self.diffusion_models.append(policy)
            self.diffusion_model_setting.append(cfg)
            # self.slider_function(f"{model_name}_dim_{latent_dim}",
            #                      [f"z[{i}]" for i in range(latent_dim)], -1, 1,
            #                      diffusion_sliders)
            # self.diffusion_sliders.append(diffusion_sliders)
            group_count += 1
        return group_count

    def get_diffusion_values(self):

        actions = []
        diffusion_reconstructed_actions = {"vae_reconstructed": []}
        for index, diffusion_model in enumerate(self.diffusion_models):
            # diffusion_slider = self.diffusion_sliders[index]

            # diffusion_values = np.array([v.get() for v in diffusion_slider])
            # action_range = self.diffusion_model_setting[index]

            # diffusion_values = (diffusion_values + 1) / 2 * (
            #     action_range[1] - action_range[0]) + action_range[0]

            # diffusion_values = torch.as_tensor(diffusion_values).unsqueeze(
            #     0).to(self.device).to(dtype=torch.float32)
            # with torch.no_grad():
            #     reconstructed_actions = diffusion_model.decode_action(
            #         diffusion_values)

            #     if self.diffusion_model_setting[index][2] is not None:
            #         reconstructed_actions = dataset_denrormalizer(
            #             reconstructed_actions,
            #             self.diffusion_model_setting[index][3],
            #             self.diffusion_model_setting[index][4])
            #     reconstructed_actions = reconstructed_actions[0].cpu().numpy()[
            #         self.retarget2pin].tolist()

            # actions.append(reconstructed_actions)

            if self.action_buffer is not None:
                target_demo_data = self.raw_dfifusion_actions[
                    self.button_wrapper.demo_count % self.num_demos]
                num_frames = len(target_demo_data)

                recontruct_diffusion_action = self.recontruct_diffusion_actions[
                    index][self.button_wrapper.demo_count %
                           self.num_demos][self.button_wrapper.frame_index %
                                           num_frames]
                raw_action = self.raw_dfifusion_actions[
                    self.button_wrapper.demo_count %
                    self.num_demos][self.button_wrapper.frame_index %
                                    num_frames][..., -self.num_hand_joints:]

                diffusion_reconstructed_actions["vae_reconstructed"].append(
                    np.concatenate([
                        np.concatenate([
                            recontruct_diffusion_action[self.retarget2pin],
                            [1]
                        ]),
                        np.concatenate([raw_action[self.retarget2pin], [1]])
                    ]).tolist())

                if self.button_wrapper.play_demo:

                    self.button_wrapper.frame_index += 1

        return {"vae": actions} | diffusion_reconstructed_actions
