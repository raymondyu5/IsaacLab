import numpy as np
import torch
from scripts.workflows.hand_manipulation.utils.meshcatviewer.button_wrapper import ButtonWrapper

import sys

sys.path.append("submodule/benchmark_VAE/src")
import yaml
from pythae.models import AutoModel
import os
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, dataset_normalizer)


class VAESliderApp:

    def __init__(self,
                 raw_sliders,
                 root,
                 action_buffer=None,
                 vae_path=None,
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
        self.vae_path = vae_path
        self.hand_side = hand_side
        self.device = device
        self.raw_sliders = raw_sliders
        self.root = root
        self.slider_function = slider_function
        self.action_buffer = action_buffer
        self.num_demos = len(self.action_buffer) if action_buffer else 0

        self.group_count = self.load_vae(group_count)
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
            group_count, num_slider, name="VAE Reconstruction")
        self.recontruct_vae_actions = [[] for _ in range(len(self.vae_models))]

        for index, vae_model in enumerate(self.vae_models):
            with torch.no_grad():
                for action_buffer in self.action_buffer:
                    action_bufer = torch.as_tensor(action_buffer).to(
                        self.device).to(
                            dtype=torch.float32)[..., -self.num_hand_joints:]
                    if self.vae_model_setting[index][2] is not None:
                        normlized_hand_pose = dataset_normalizer(
                            action_bufer, self.vae_model_setting[index][3],
                            self.vae_model_setting[index][4])

                        recontructed_hand_pose = dataset_denrormalizer(
                            vae_model({
                                "data": normlized_hand_pose
                            }).recon_x, self.vae_model_setting[index][3],
                            self.vae_model_setting[index][4])
                    else:
                        recontructed_hand_pose = vae_model.decode_action(
                            vae_model({
                                "data": action_bufer
                            }).z)

                        # recontructed_hand_pose = vae_model({
                        #     "data": action_bufer
                        # }).recon_x

                    self.recontruct_vae_actions[index].append(
                        recontructed_hand_pose.cpu().numpy())

        return group_count

    def load_vae(self, group_count):

        self.vae_sliders = []
        self.vae_models = []
        self.vae_model_setting = []

        for index, vae_path in enumerate(self.vae_path):
            vae_sliders = []

            all_dirs = [
                d for d in os.listdir(vae_path)
                if os.path.isdir(os.path.join(vae_path, d))
            ]
            last_training = sorted(all_dirs)[-1]

            vae_model = AutoModel.load_from_folder(os.path.join(
                vae_path, last_training, 'final_model'),
                                                   device=self.device)
            vae_model.eval()

            model_name = vae_model.model_name
            self.vae_models.append(vae_model.to(self.device))

            with open(f"{vae_path}/model_config.yaml", "r") as f:
                model_config = yaml.safe_load(f)

                action_mean = torch.as_tensor(model_config["action_mean"]).to(
                    self.device)
                action_std = torch.as_tensor(model_config["action_std"]).to(
                    self.device)
                data_normalizer = model_config["data_normalizer"]
                max_latent_value = np.array(model_config["max_latent_value"])
                min_latent_value = np.array(model_config["min_latent_value"])
                embedding_dim = model_config["embedding_dim"]

                self.vae_model_setting.append([
                    min_latent_value, max_latent_value, data_normalizer,
                    action_mean, action_std
                ])

            self.slider_function(f"{model_name}_dim_{embedding_dim}",
                                 [f"z[{i}]" for i in range(embedding_dim)], -1,
                                 1, vae_sliders)
            self.vae_sliders.append(vae_sliders)
            group_count += 1
        return group_count

    def get_vae_values(self):

        actions = []
        vae_reconstructed_actions = {"vae_reconstructed": []}
        for index, vae_model in enumerate(self.vae_models):
            vae_slider = self.vae_sliders[index]

            vae_values = np.array([v.get() for v in vae_slider])
            action_range = self.vae_model_setting[index]

            vae_values = (vae_values + 1) / 2 * (
                action_range[1] - action_range[0]) + action_range[0]

            vae_values = torch.as_tensor(vae_values).unsqueeze(0).to(
                self.device).to(dtype=torch.float32)
            with torch.no_grad():
                reconstructed_actions = vae_model.decode_action(vae_values)

                if self.vae_model_setting[index][2] is not None:
                    reconstructed_actions = dataset_denrormalizer(
                        reconstructed_actions,
                        self.vae_model_setting[index][3],
                        self.vae_model_setting[index][4])
                reconstructed_actions = reconstructed_actions[0].cpu().numpy()[
                    self.retarget2pin].tolist()

            actions.append(reconstructed_actions)

            if self.action_buffer is not None:
                target_demo_data = self.action_buffer[
                    self.button_wrapper.demo_count % self.num_demos]
                num_frames = len(target_demo_data)

                recontruct_vae_action = self.recontruct_vae_actions[index][
                    self.button_wrapper.demo_count %
                    self.num_demos][self.button_wrapper.frame_index %
                                    num_frames][self.retarget2pin]
                raw_action = self.action_buffer[
                    self.button_wrapper.demo_count % self.num_demos][
                        self.button_wrapper.frame_index %
                        num_frames][...,
                                    -self.num_hand_joints:][self.retarget2pin]

                vae_reconstructed_actions["vae_reconstructed"].append(
                    np.concatenate([recontruct_vae_action,
                                    raw_action]).tolist())

                if self.button_wrapper.play_demo:

                    self.button_wrapper.frame_index += 1

        return {"vae": actions} | vae_reconstructed_actions
