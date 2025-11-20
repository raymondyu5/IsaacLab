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


class EVALVAEBENCHMARK:

    def __init__(self, args_cli, num_hand_joints=16, device=None):
        self.args_cli = args_cli
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")
        self.hand_side = args_cli.hand_side
        self.num_hand_joints = num_hand_joints
        self.joint_limits = self._init_joint_limits()  #.to(self.device)
        self.init_model()

    def _init_joint_limits(self):
        return torch.tensor(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.47, 2.443], [-1.047, 1.047], [-1.047, 1.047],
             [-0.506, 1.885], [-1.2, 1.9], [-0.506, 1.885], [-0.506, 1.885],
             [-0.366, 2.042], [-1.34, 1.88], [-0.366, 2.042], [-0.366, 2.042]],
            dtype=torch.float32)

    def init_model(self):
        vae_root = self.args_cli.log_dir
        vae_loss, vae_error_hist, act_vae_error_hist = {}, {}, {}

        for vae_source in os.listdir(vae_root):
            vae_dir = os.path.join(vae_root, vae_source)

            all_dirs = [
                d for d in os.listdir(vae_dir)
                if os.path.isdir(os.path.join(vae_dir, d))
            ]
            if len(all_dirs) == 0:

                continue

            for subdir in os.listdir(vae_dir):
                all_dirs = [
                    d for d in os.listdir(os.path.join(vae_dir, subdir))
                    if os.path.isdir(os.path.join(vae_dir, subdir + "/" + d))
                ]
                if len(all_dirs) == 0:

                    continue
                last_training = sorted(all_dirs)[-1]
                model_dir = os.path.join(vae_dir, subdir, last_training,
                                         "final_model")

                if not os.path.isdir(model_dir): continue

                try:
                    vae_model = AutoModel.load_from_folder(
                        model_dir, device=self.device).eval().to(self.device)
                except:
                    print(f"VAE model not found: {model_dir}")
                    continue

                config_path = os.path.join(vae_dir, subdir)
                vae_config = self.load_config(config_path)
                self.init_data(vae_config[-1])

                recon, ratio = self.reconstruct(vae_model, vae_config)

                original = self.train_dataset[0].squeeze(1)

                diff = self.extract_finger_joints(
                    recon.cpu()) - self.extract_finger_joints(original.cpu())
                wrapped = (diff + torch.pi) % (2 * torch.pi) - torch.pi

                sum_error = (torch.abs(wrapped).sum(dim=-1) / 16)
                mean_error = sum_error.mean().item()
                torch.cuda.empty_cache()

                std_error = recon.sub(original).reshape(
                    recon.shape[0], -1).std(dim=1).mean().item()

                key = self.build_key(vae_model, vae_config)
                vae_type = vae_model.model_name
                chunk_key = f"chunk_{vae_config[-1]}"
                latent_key = f"embedding_dim_{vae_config[-2]}"

                act_vae_error_hist[key] = sum_error.cpu().numpy().reshape(
                    sum_error.shape[0], -1)

                vae_error_hist[key] = torch.abs(wrapped).cpu().numpy()
                self.update_loss_dict(vae_loss, vae_type, chunk_key,
                                      latent_key, key, mean_error, std_error,
                                      ratio, vae_model, torch.abs(wrapped))
                del vae_model
                torch.cuda.empty_cache()

        self.visualize_vae_hist(vae_error_hist)

        self.visualize_vae_loss(vae_loss)

        self.plot_act_error_details(act_vae_error_hist)

    def extract_finger_joints(self, joints):
        scale = self.joint_limits[:, 1] - self.joint_limits[:, 0]
        return (joints + 1) / 2 * scale + self.joint_limits[:, 0]

    def reconstruct(self, model, config, batch_size=2048):
        recon_x_list = []
        latent_list = []

        with torch.no_grad():
            action_chunk = self.train_dataset[0]
            state = self.train_dataset[1]

            for start in range(0, len(action_chunk), batch_size):
                end = start + batch_size
                batch_action = action_chunk[start:end].to(self.device)
                batch_state = state[start:end].to(self.device)

                result = model({
                    "data": {
                        "action_chunk": batch_action,
                        "state": batch_state
                    }
                })

                recon_x_list.append(result.recon_x)
                latent_list.append(result.z)

            # Concatenate reconstructions
            recon_x = torch.cat(recon_x_list, dim=0)
            latent_list = torch.cat(latent_list, dim=0)

            # Assume latent_list is already loaded and has shape [217388, 1, 2]
            latent_xy = latent_list.squeeze(1)  # shape becomes [217388, 2]

            # Convert to NumPy for plotting (optional but recommended)
            latent_xy_np = latent_xy.cpu().numpy()

            # Plot
            # plt.figure(figsize=(6, 6))
            # plt.scatter(latent_xy_np[:, 0],
            #             latent_xy_np[:, 1],
            #             s=0.1,
            #             alpha=0.5)
            # plt.xlabel('Latent Dim 1')
            # plt.ylabel('Latent Dim 2')
            # plt.title('Latent Space XY Plot')
            # x_min, x_max = np.min(latent_xy_np[:, 0]), np.max(latent_xy_np[:,
            #                                                                0])
            # y_min, y_max = np.min(latent_xy_np[:, 1]), np.max(latent_xy_np[:,
            #                                                                1])
            # print(
            #     f"Latent space range: x [{x_min:.2f}, {x_max:.2f}], y [{y_min:.2f}, {y_max:.2f}]"
            # )
            # plt.xlim(x_min, x_max)
            # plt.ylim(y_min, y_max)
            # plt.grid(True)
            # plt.axis('equal')
            # plt.show()

            # Compute reconstruction ratio using the final result (or optionally from recon_x)
            ratio = self._compute_ratio(
                model, result)  # Note: this uses only the last batch's result

            if config[2] is None:
                return recon_x, ratio
            else:
                return dataset_denrormalizer(recon_x, config[3],
                                             config[4]), ratio

    def _compute_ratio(self, model, result):
        if model.model_name not in ["HRQVAE", "VQVAE"]:
            return 0
        flat = result.quantized_indices.view(
            -1, result.quantized_indices.shape[-1])

        return torch.unique(flat,
                            dim=0).shape[0] / model.quantizer.num_embeddings

    def build_key(self, model, config):
        embedding_dim, chunk_size = config[-2], config[-1]
        if model.model_name in ["HRQVAE", "VQVAE"]:
            return f"{model.model_name}_latent_{embedding_dim}_chunk_{chunk_size}_emb_{model.quantizer.num_embeddings}"
        return f"{model.model_name}_latent_{embedding_dim}_chunk_{chunk_size}"

    def update_loss_dict(self, loss_dict, vae_type, chunk_key, latent_key, key,
                         mean, std, ratio, model, raw_errors):
        loss_dict.setdefault(vae_type, {}).setdefault(chunk_key, {})
        if model.model_name in ["HRQVAE", "VQVAE"]:
            emb_key = f"num_embeddings_{model.quantizer.num_embeddings}"
            loss_dict[vae_type][chunk_key].setdefault(
                latent_key, {})[emb_key] = {
                    "mean": mean,
                    "std": std,
                    "ratio": ratio,
                    "raw_errors": raw_errors.cpu().numpy(),
                    "chunk_size": int(chunk_key.split('_')[-1])
                }
        else:
            loss_dict[vae_type][chunk_key][latent_key] = {
                "mean": mean,
                "std": std,
                "ratio": ratio,
                "raw_errors": raw_errors.cpu().numpy(),
                "chunk_size": int(chunk_key.split('_')[-1])
            }

    def load_config(self, model_dir):
        with open(f"{model_dir}/model_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        return [
            np.array(cfg["min_latent_value"]),
            np.array(cfg["max_latent_value"]), cfg["data_normalizer"],
            torch.tensor(cfg["action_mean"]).to(self.device),
            torch.tensor(cfg["action_std"]).to(self.device),
            cfg["embedding_dim"], cfg["chunk_size"]
        ]

    def init_data(self, chunk_size):
        self.train_dataset, *_ = init_chunk_data(self.args_cli.data_dir,
                                                 self.num_hand_joints,
                                                 chunk_size, 1.0, self.device)

    def visualize_vae_hist(self, vae_error_hist):
        import os

        # Step 1: Parse keys like "VQVAE_latent_8_chunk_10_emb_64"
        grouped = {}
        for name, errors in vae_error_hist.items():
            match = re.match(r"(.*)_latent_(\d+)_chunk_(\d+)(?:_emb_(\d+))?",
                             name)
            if not match:
                continue
            vae_type, embedding_dim, chunk_size, num_embeddings = match.groups(
            )
            group_key = f"{vae_type}_latent_{embedding_dim}_chunk_{chunk_size}"

            if group_key not in grouped:
                grouped[group_key] = []

            grouped[group_key].append({
                "name":
                name,
                "vae_type":
                vae_type,
                "embedding_dim":
                int(embedding_dim),
                "chunk_size":
                int(chunk_size),
                "num_embeddings":
                int(num_embeddings) if num_embeddings else None,
                "errors":
                errors
            })

        # Step 2: For each group (fixed embedding_dim + chunk), make one figure
        for group_key, entries in grouped.items():
            entries.sort(key=lambda x: x["num_embeddings"]
                         if x["num_embeddings"] is not None else -1)

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
                np.histogram(e["errors"].reshape(-1), bins=50)[0].max()
                for e in entries)
            y_max = global_max_count * 1.1

            all_errors = np.concatenate([e["errors"]
                                         for e in entries]).reshape(-1)

            for idx, entry in enumerate(entries):
                ax = axes[idx]
                errors = entry["errors"].reshape(-1)
                bins = np.linspace(np.min(errors), np.max(errors), 50)
                x_min, x_max = min(all_errors), max(all_errors)

                ax.hist(errors, bins=bins, color='skyblue', edgecolor='black')
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(0, y_max)

                title = f"Embeddings: {entry['num_embeddings']}" if entry[
                    'num_embeddings'] is not None else "VAE"
                ax.set_title(title, fontsize=10)
                ax.set_xlabel(f"Sum Error\n[{x_min:.2f}, {x_max:.2f}]")
                ax.set_ylabel(f"Freq\n[0, {int(y_max)}]")
                ax.grid(True)

            # Hide unused subplots
            for idx in range(len(entries), len(axes)):
                fig.delaxes(axes[idx])

            fig.suptitle(f"Sum Error Distribution for {group_key}",
                         fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            # Save under chunk-specific subfolder
            chunk_size = entries[0]["chunk_size"]
            save_dir = os.path.join(self.args_cli.log_dir,
                                    f"chunk_{chunk_size}")
            os.makedirs(save_dir, exist_ok=True)

            save_path = os.path.join(save_dir, f"{group_key}_hist.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

    def visualize_vae_loss(self, vae_loss):
        # Step 1: Flatten the dictionary into rows
        rows = []
        for vae_type, chunks in vae_loss.items():
            for chunk_key, latent_dict in chunks.items():
                chunk_size = int(chunk_key.replace("chunk_", ""))
                for latent_key, value in latent_dict.items():
                    embedding_dim = int(
                        latent_key.replace("embedding_dim_", ""))
                    if isinstance(value, dict) and all(
                            k.startswith("num_embeddings_")
                            for k in value.keys()):
                        # Quantized
                        for emb_key, metrics in value.items():
                            num_embeddings = int(
                                emb_key.replace("num_embeddings_", ""))
                            mean_loss = metrics.get("mean", None)
                            std_loss = metrics.get("std", None)
                            ratio = metrics.get("ratio", None)
                            rows.append([
                                vae_type, embedding_dim, chunk_size,
                                num_embeddings, mean_loss, std_loss, ratio
                            ])
                    else:
                        # Non-quantized
                        mean_loss = value.get("mean", None)
                        std_loss = value.get("std", None)
                        ratio = value.get("ratio", None)
                        rows.append([
                            vae_type, embedding_dim, chunk_size, 1, mean_loss,
                            std_loss, ratio
                        ])

        # Step 2: Create DataFrame
        df = pd.DataFrame(rows,
                          columns=[
                              "VAE Type", "Latent Dim", "Chunk Size",
                              "Num Embeddings", "Mean Loss", "Std Loss",
                              "Ratio"
                          ])

        # Step 3: Visualize, grouped by chunk size
        for chunk_size in sorted(df["Chunk Size"].dropna().unique()):
            df_chunk = df[df["Chunk Size"] == chunk_size]

            for metric in ["Mean Loss", "Std Loss", "Ratio"]:
                for vae_type in df_chunk["VAE Type"].unique():
                    plot_df = df_chunk[df_chunk["VAE Type"] == vae_type]

                    if metric not in plot_df.columns or plot_df[metric].isnull(
                    ).all():
                        continue

                    save_dir = os.path.join(self.args_cli.log_dir,
                                            f"chunk_{chunk_size}")
                    os.makedirs(save_dir, exist_ok=True)

                    if plot_df["Num Embeddings"].isnull().all() or (
                            plot_df["Num Embeddings"] == 0).all():
                        # Non-quantized: line plot vs latent dim
                        plt.figure()
                        plt.errorbar(plot_df["Latent Dim"],
                                     plot_df[metric],
                                     yerr=plot_df["Std Loss"]
                                     if metric == "Mean Loss" else None,
                                     fmt='-o')
                        plt.title(
                            f"{metric} vs Latent Dim for {vae_type} (chunk {chunk_size})"
                        )
                        plt.xlabel("Latent Dim")
                        plt.ylabel(metric)
                        plt.grid(True)
                        plt.tight_layout()
                        save_name = f"{vae_type}_{metric.replace(' ', '_').lower()}_chunk_{chunk_size}_vs_latent.png"
                        plt.savefig(os.path.join(save_dir, save_name),
                                    bbox_inches='tight')
                        plt.close()
                        continue

                    # Quantized: heatmap
                    plot_df = plot_df[plot_df["Num Embeddings"] > 0]
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
                    plt.title(
                        f"{metric} Heatmap for {vae_type} (chunk {chunk_size})"
                    )
                    plt.ylabel("Latent Dim")
                    plt.xlabel("Num Embeddings")
                    plt.tight_layout()
                    save_name = f"{vae_type}_{metric.replace(' ', '_').lower()}_chunk_{chunk_size}.png"
                    plt.savefig(os.path.join(save_dir, save_name),
                                bbox_inches='tight')
                    plt.close()

    def plot_act_error_details(self, act_vae_error_hist):
        for key, errors in act_vae_error_hist.items():
            chunk_size = errors.shape[1]
            cols = 5
            rows = int(np.ceil(chunk_size / cols))

            # === 1. Subplots for each time step ===
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axes = axes.flatten()

            for t in range(chunk_size):
                ax = axes[t]
                sns.histplot(errors[:, t],
                             bins=50,
                             kde=False,
                             ax=ax,
                             color='skyblue',
                             edgecolor='black')
                min_val = errors[:, t].min()
                max_val = errors[:, t].max()
                ax.set_title(f"Step {t}\n[{min_val:.3f}, {max_val:.3f}]")
                ax.set_xlabel("Error Value")
                ax.set_ylabel("Frequency")
                ax.set_xlim(errors.min(), errors.max())

                ax.grid(True)

            for t in range(chunk_size, len(axes)):
                fig.delaxes(axes[t])

            fig.suptitle(f"Action Error Distribution per Step\n{key}",
                         fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])

            save_dir = os.path.join(self.args_cli.log_dir,
                                    f"chunk_{chunk_size}")
            os.makedirs(save_dir, exist_ok=True)
            subplot_path = os.path.join(
                save_dir, f"{key}_act_error_per_step_subplots.png")
            plt.savefig(subplot_path, bbox_inches="tight")
            plt.close()

            # === 2. Combined histogram in one figure ===
            plt.figure(figsize=(12, 6))
            for t in range(chunk_size):
                sns.histplot(errors[:, t],
                             bins=50,
                             kde=False,
                             label=f"Step {t}",
                             alpha=0.6)

            plt.title(f"Overlayed Action Error Distribution\n{key}")
            plt.xlabel("Error Value")
            plt.ylabel("Frequency")
            plt.legend(title="Chunk Step",
                       bbox_to_anchor=(1.05, 1),
                       loc='upper left')
            plt.tight_layout()

            combined_path = os.path.join(
                save_dir, f"{key}_act_error_per_step_combined.png")
            plt.savefig(combined_path, bbox_inches="tight")
            plt.close()


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

    vae = EVALVAEBENCHMARK(args_cli)
