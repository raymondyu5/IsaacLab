import numpy as np

import torch
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper
import sys

sys.path.append("submodule/benchmark_VAE/src")
from pythae.models import AutoModel
import os
import yaml

from scripts.workflows.hand_manipulation.utils.vae.vae_plot import plot_action
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, dataset_normalizer)
from scripts.workflows.hand_manipulation.env.datagen.vae_verify_wrapper import VAEVerifyWrapper
from scripts.workflows.hand_manipulation.env.datagen.act_vae_verify_wrapper import ACTVAEVerifyWrapper
from scripts.workflows.hand_manipulation.env.datagen.diffusion_verify_wrapper import DiffusionVerifyWrapper

from scripts.workflows.hand_manipulation.env.datagen.latent_diffusion_verify_wrapper import LatentDiffusionVerifyWrapper


class VAEWrapper:

    def __init__(
        self,
        args_cli,
        env_config,
        env,
    ):
        self.args_cli = args_cli
        self.env_config = env_config
        self.env = env
        self.device = env.unwrapped.device
        self.actions_buffer = None

        self.add_left_hand = args_cli.add_left_hand
        self.add_right_hand = args_cli.add_right_hand
        self.num_hand_joints = self.env_config["params"]["num_hand_joints"]

        self.collector_interface = MultiDatawrapper(
            args_cli,
            env_config,
            load_path=args_cli.load_path,
            save_path=args_cli.save_path,
        )
        if self.args_cli.save_path is not None:
            self.collector_interface.init_collector_interface()
        self.num_count = 0
        if self.add_left_hand:
            self.hand_side = "left"
        elif self.add_right_hand:
            self.hand_side = "right"
        self.raw_data = self.collector_interface.raw_data["data"]

        self.num_data = len(self.raw_data)
        self.demo_count = 0
        self.load_all_actions()

        if self.args_cli.model_name == "vae":
            self.verify_env = VAEVerifyWrapper(env, args_cli)

            self.random_play = self.verify_env.random_play_vae
            self.verify_model = self.verify_env.verify_vae

            setattr(self.verify_env, "actions_buffer", self.actions_buffer)
            setattr(self.verify_env, "raw_data", self.raw_data)

            self.verify_env.load_vae_model()
        elif self.args_cli.model_name == "diffusion":
            self.verify_env = DiffusionVerifyWrapper(env, args_cli)
            self.random_play = self.verify_env.random_play_diffusion
            self.verify_model = self.verify_env.verify_diffusion
            setattr(self.verify_env, "action_buffer", self.raw_action_buffer)
            setattr(self.verify_env, "raw_data", self.raw_data)
        elif self.args_cli.model_name == "latent_diffusion":
            self.verify_env = LatentDiffusionVerifyWrapper(env, args_cli)
            self.random_play = self.verify_env.random_play_diffusion
            self.verify_model = self.verify_env.verify_diffusion
            setattr(self.verify_env, "action_buffer", self.raw_action_buffer)
            setattr(self.verify_env, "raw_data", self.raw_data)
        elif self.args_cli.model_name == "vae_act":
            self.verify_env = ACTVAEVerifyWrapper(env, args_cli)
            self.random_play = self.verify_env.random_play_vae
            self.verify_model = self.verify_env.verify_vae
            setattr(self.verify_env, "action_buffer", self.raw_action_buffer)
            setattr(self.verify_env, "raw_data", self.raw_data)

    def load_all_actions(self):

        actions_buffer = []
        for i in range(self.num_data):
            if self.add_left_hand:
                actions_buffer.append(
                    torch.as_tensor(
                        np.array(self.raw_data[f"demo_{i}"]["actions"][
                            ..., -self.num_hand_joints:])).to(self.device))
            elif self.add_right_hand:
                actions_buffer.append(
                    torch.as_tensor(
                        np.array(self.raw_data[f"demo_{i}"]["actions"][
                            ..., -self.num_hand_joints:])).to(self.device))

        self.actions_buffer = torch.cat(actions_buffer, dim=0)
        self.raw_action_buffer = actions_buffer
