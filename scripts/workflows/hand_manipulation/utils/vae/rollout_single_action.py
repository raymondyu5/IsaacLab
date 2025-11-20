import os
import sys
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append("submodule/benchmark_VAE/src")
from pythae.models import AutoModel
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import init_chunk_data
import re
import math
from scripts.workflows.utils.robomimc_collector import RobomimicDataCollector, sample_train_test
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    load_config, load_data)


class RolloutLatent:

    def __init__(self, args_cli, num_hand_joints=16, device=None):
        self.args_cli = args_cli
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")
        self.hand_side = args_cli.hand_side
        self.num_hand_joints = num_hand_joints

        self.init_model()

    def init_model(self):

        vae_dir = self.args_cli.log_dir

        all_dirs = [
            d for d in os.listdir(vae_dir)
            if os.path.isdir(os.path.join(vae_dir, d))
        ]

        last_training = sorted(all_dirs)[-1]

        model_dir = os.path.join(vae_dir, last_training, "final_model")

        collector_interface = RobomimicDataCollector(
            "convert_action", os.path.dirname(self.args_cli.log_dir),
            "latent_action", 200000)
        collector_interface.reset()

        vae_model = AutoModel.load_from_folder(
            model_dir, device=self.device).eval().to(self.device)

        vae_config = load_config(vae_dir, self.device)
        raw_actions, all_raw_actions = load_data(self.args_cli.data_dir,
                                                 self.device)
        unique_id = []
        with torch.no_grad():

            for data in all_raw_actions:
                result = vae_model({
                    "data":
                    torch.as_tensor(data)[1:].to(self.device).to(torch.float32)
                })

                if vae_model.model_name in ["VQVAE"]:
                    latent_actions = result.quantized_indices
                    latent_actions = latent_actions.reshape(
                        len(latent_actions), -1)
                    num_embeddings = vae_model.quantizer.num_embeddings
                    latent_actions = (latent_actions /
                                      (num_embeddings - 1)) * 2 - 1
                    unique_id.append(result.quantized_indices.cpu().numpy())

                else:
                    latent_actions = result.z
                num_sequences = len(data) - 1
                for i in range(num_sequences):

                    collector_interface.add("actions",
                                            latent_actions[i].unsqueeze(0))
                    collector_interface.add(
                        f"obs/state",
                        torch.as_tensor(data)[i].to(self.device).to(
                            torch.float32).unsqueeze(0))

                    dones = torch.tensor([False], device=self.device)
                    if i == num_sequences - 1:
                        dones[-1] = True
                    reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)

                    collector_interface.flush(reset_env_ids)

                print(np.unique(np.concatenate(unique_id, axis=0)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/data_0604/right_vae_state_act/",
    )

    parser.add_argument(
        "--hand_side",
        type=str,
        default="right",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="logs/data_0604/raw_right_data.hdf5",
    )
    args_cli, hydra_args = parser.parse_known_args()

    vae = RolloutLatent(args_cli)
