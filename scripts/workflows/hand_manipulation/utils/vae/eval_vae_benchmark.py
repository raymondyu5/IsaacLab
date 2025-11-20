import sys

sys.path.append("submodule/benchmark_VAE/src")
import yaml
from pythae.models import AutoModel
import os
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, dataset_normalizer, load_config, load_data)

import h5py
import numpy as np
import torch

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math
import re


class EVALVAEBENCHMARK:

    def __init__(self,
                 args_cli,
                 num_hand_joints=16,
                 device="cuda" if torch.cuda.is_available() else "cpu"):

        self.args_cli = args_cli
        self.num_hand_joints = num_hand_joints
        self.hand_side = args_cli.hand_side
        self.device = device
        self.joint_limits = torch.as_tensor(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.46999997, 2.4429998], [-1.047, 1.047],
             [-1.047, 1.047], [-0.5059999, 1.8849999], [-1.2, 1.8999999],
             [-0.5059999, 1.8849999], [-0.5059999, 1.8849999],
             [-0.366, 2.0419998], [-1.34, 1.8799999], [-0.366, 2.0419998],
             [-0.366, 2.0419998]], ).to(device).to(torch.float32)

        self.all_actions, _ = load_data(self.args_cli.data_dir, self.device)

        self.init_model()

    def init_model(self, ):
        vae_dir = os.path.dirname(self.args_cli.log_dir +
                                  f"/{self.hand_side}_vae/")
        vae_list = os.listdir(vae_dir)
        print("vae_list:", vae_list)
        vae_loss = {}
        vae_error_hist = {}

        for vae_source_file in vae_list:
            if not os.path.isdir(os.path.join(vae_dir, vae_source_file)):

                continue
            for file in os.listdir(os.path.join(vae_dir, vae_source_file)):
                vae_file = vae_source_file + "/" + file
                if not os.path.isdir(os.path.join(vae_dir, vae_file)):
                    continue

                vae_checkpoint = os.path.join(vae_dir, vae_file)
                all_dirs = [
                    d for d in os.listdir(vae_checkpoint)
                    if os.path.isdir(os.path.join(vae_checkpoint, d))
                ]
                last_training = sorted(all_dirs)[-1]
                try:

                    vae_model = AutoModel.load_from_folder(
                        os.path.join(vae_checkpoint, last_training,
                                     'final_model'),
                        device=self.device).to(self.device)
                except:
                    print(f"vae model {vae_checkpoint} not found")
                    continue
                vae_model.eval()
                vae_config = load_config(vae_checkpoint)
                recontructed_pose, ratio = self.recontruct_vae_actions(
                    vae_model, vae_config)

                diff = self.extract_finger_joints(
                    recontructed_pose) - self.extract_finger_joints(
                        self.all_actions.clone())
                wrapped_diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
                sum_error = torch.sum(torch.abs(wrapped_diff), dim=-1) / 16
                # sum_error = torch.sum(abs(recontructed_pose -
                #                           self.all_actions.clone()),
                #                       dim=-1)

                std = torch.std(recontructed_pose - self.all_actions.clone(),
                                dim=-1,
                                keepdim=True)
                mean_error = torch.mean(sum_error).item()
                std_error = torch.mean(std).item()

                loss = torch.mean(sum_error).item()

                if vae_model.model_name in ["HRQVAE", "VQVAE"]:
                    key = f"{vae_model.model_name}_latent_{vae_config[-1]}_emb_{vae_model.quantizer.num_embeddings}"
                else:
                    key = f"{vae_model.model_name}_latent_{vae_config[-1]}"
                vae_error_hist[key] = sum_error.detach().cpu().numpy()

                vae_type = vae_model.model_name
                embedding_dim = vae_config[-1]
                latent_key = f"embedding_dim_{embedding_dim}"

                if vae_type not in vae_loss:
                    vae_loss[vae_type] = {"embedding_dim": {}}

                if vae_type not in ["HRQVAE", "VQVAE"]:
                    # Non-quantized VAE (e.g., AE)
                    vae_loss[vae_type]["embedding_dim"][latent_key] = {
                        "mean": mean_error,
                        "std": std_error,
                        "ratio": ratio,
                    }
                else:
                    # Quantized VAE (e.g., VQVAE, HRQVAE)
                    num_embeddings = vae_model.quantizer.num_embeddings
                    emb_key = f"num_embeddings_{num_embeddings}"

                    if latent_key not in vae_loss[vae_type]["embedding_dim"]:
                        vae_loss[vae_type]["embedding_dim"][latent_key] = {
                            "num_embeddings": {}
                        }

                    vae_loss[vae_type]["embedding_dim"][latent_key][
                        "num_embeddings"][emb_key] = {
                            "mean": mean_error,
                            "std": std_error,
                            "ratio": ratio,
                        }
        self.visualize_vae_hist(vae_error_hist)
        self.visualize_vae_loss(vae_loss)

    def extract_finger_joints(self, joints):

        raw_joints = (joints + 1) / 2 * (
            self.joint_limits[:, 1] -
            self.joint_limits[:, 0]) + self.joint_limits[:, 0]
        return raw_joints

    def recontruct_vae_actions(self, vae_model, vae_config):

        if vae_config[2] is not None:
            raw_actions = dataset_normalizer(self.all_actions.clone(),
                                             vae_config[3], vae_config[4])
        else:
            raw_actions = self.all_actions.clone()
        ratio = 0
        with torch.no_grad():
            result = vae_model({"data": raw_actions})
            if vae_model.model_name in ["HRQVAE", "VQVAE"]:

                flat = result.quantized_indices.view(
                    -1, result.quantized_indices.shape[-1])  # shape (24788, 8)

                # Step 2: Count unique rows (pairs of 8 indices)
                unique_rows = torch.unique(flat, dim=0)
                num_unique = unique_rows.shape[0]
                num_embeddings = vae_model.quantizer.num_embeddings
                ratio = num_unique / num_embeddings

            if vae_config[2] is not None:

                recontructed_hand_pose = dataset_denrormalizer(
                    result.recon_x, vae_config[3], vae_config[4])
            else:
                recontructed_hand_pose = result.recon_x
        return recontructed_hand_pose, ratio

    def visualize_vae_hist(self, vae_error_hist):

        # Step 1: Parse keys like "VQVAE_latent_8_emb_64"
        grouped = {}
        for name, errors in vae_error_hist.items():
            match = re.match(r"(.*)_latent_(\d+)(?:_emb_(\d+))?", name)
            if not match:
                continue
            vae_type, embedding_dim, num_embeddings = match.groups()
            key = f"{vae_type}_latent_{embedding_dim}"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append({
                "name":
                name,
                "vae_type":
                vae_type,
                "embedding_dim":
                int(embedding_dim),
                "num_embeddings":
                int(num_embeddings) if num_embeddings else None,
                "errors":
                errors
            })

        # Step 2: For each group (one embedding_dim), make one figure
        for group_key, entries in grouped.items():
            entries.sort(
                key=lambda x: x["num_embeddings"])  # Sort for nice layout

            num_plots = len(entries)
            cols = 4
            rows = math.ceil(num_plots / cols)

            fig, axes = plt.subplots(rows,
                                     cols,
                                     figsize=(5 * cols, 4 * rows),
                                     sharex=False,
                                     sharey=False)
            axes = axes.flatten()

            # Compute shared Y-axis upper bound (for visual comparison)
            global_max_count = max(
                np.histogram(e["errors"], bins=50)[0].max() for e in entries)
            y_max = global_max_count * 1.1

            # Plot each subplot with its own X range
            for idx, entry in enumerate(entries):
                ax = axes[idx]
                errors = entry["errors"]
                bins = np.linspace(np.min(errors), np.max(errors), 50)
                x_min, x_max = bins[0], bins[-1]

                ax.hist(errors, bins=bins, color='skyblue', edgecolor='black')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(0, y_max)

                title = f"Embeddings: {entry['num_embeddings']}" if entry[
                    'num_embeddings'] is not None else "VAE"
                ax.set_title(title, fontsize=10)
                ax.set_xlabel(f"Sum Error\n[{x_min:.2f}, {x_max:.2f}]")
                ax.set_ylabel(f"Freq\n[0, {int(y_max)}]")
                ax.grid(True)

            # Hide unused axes
            for idx in range(len(entries), len(axes)):
                fig.delaxes(axes[idx])

            fig.suptitle(f"Sum Error Distribution for {group_key}",
                         fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            save_path = os.path.join(
                self.args_cli.log_dir + f"/{self.hand_side}_vae/",
                f"{group_key}_hist.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

    def visualize_vae_loss(self, vae_loss):

        # Step 1: Flatten the dictionary into rows
        rows = []
        for vae_type, outer in vae_loss.items():
            for latent_key, value in outer["embedding_dim"].items():
                embedding_dim = int(latent_key.replace("embedding_dim_", ""))
                if isinstance(value, dict) and "num_embeddings" in value:
                    for emb_key, metrics in value["num_embeddings"].items():
                        num_embeddings = int(
                            emb_key.replace("num_embeddings_", ""))
                        mean_loss = metrics.get("mean", None)
                        std_loss = metrics.get("std", None)
                        ratio = metrics.get("ratio", None)
                        rows.append([
                            vae_type, embedding_dim, num_embeddings, mean_loss,
                            std_loss, ratio
                        ])
                else:
                    # Handle non-quantized VAEs
                    mean_loss = value.get("mean", None) if isinstance(
                        value, dict) else None
                    std_loss = value.get("std", None) if isinstance(
                        value, dict) else None
                    rows.append([
                        vae_type, embedding_dim, 1, mean_loss, std_loss, None
                    ])

        # Step 2: Create DataFrame
        df = pd.DataFrame(rows,
                          columns=[
                              "VAE Type", "Latent Dim", "Num Embeddings",
                              "Mean Loss", "Std Loss", "Ratio"
                          ])

        # Step 3: Visualize
        for metric in ["Mean Loss", "Std Loss", "Ratio"]:
            for vae_type in df["VAE Type"].unique():
                plot_df = df[df["VAE Type"] == vae_type]

                # Skip empty or unavailable metric
                if metric not in plot_df.columns or plot_df[metric].isnull(
                ).all():
                    continue

                # If model does not use embeddings → line plot
                if plot_df["Num Embeddings"].isnull().all():
                    plt.figure()
                    plt.errorbar(plot_df["Latent Dim"],
                                 plot_df[metric],
                                 yerr=plot_df["Std Loss"]
                                 if metric == "Mean Loss" else None,
                                 fmt='-o')
                    plt.title(f"{metric} vs Latent Dim for {vae_type}")
                    plt.xlabel("Latent Dim")
                    plt.ylabel(metric)
                    plt.grid(True)
                    plt.tight_layout()
                    save_name = f"{vae_type}_{metric.replace(' ', '_').lower()}_vs_latent.png"
                    plt.savefig(os.path.join(
                        self.args_cli.log_dir + f"/{self.hand_side}_vae",
                        save_name),
                                bbox_inches='tight')
                    plt.close()
                    continue

                # Else → heatmap
                plot_df = plot_df[plot_df["Num Embeddings"].notna()]
                if plot_df.empty:
                    continue

                heatmap_data = plot_df.pivot(index="Latent Dim",
                                             columns="Num Embeddings",
                                             values=metric)

                plt.figure(figsize=(10, 6))
                sns.heatmap(heatmap_data,
                            annot=True,
                            fmt=".3f",
                            cmap="viridis")
                plt.title(f"{metric} Heatmap for {vae_type}")
                plt.ylabel("Latent Dim")
                plt.xlabel("Num Embeddings")
                plt.tight_layout()
                save_name = f"{vae_type}_{metric.replace(' ', '_').lower()}.png"
                plt.savefig(os.path.join(
                    self.args_cli.log_dir + f"/{self.hand_side}_vae",
                    save_name),
                            bbox_inches='tight')
                plt.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs/data_0604",
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

    vae = EVALVAEBENCHMARK(args_cli)
