import os
import sys
import yaml
import torch
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import time

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import (
    dataset_denrormalizer, extract_finger_joints, TemporalEnsembleBufferAction,
    TemporalEnsembleBufferObservation)
import time

import re
import math
from scripts.workflows.hand_manipulation.utils.vae.data_normalizer import sliding_chunks, init_chunk_data


class EVALdiffusionBENCHMARK:

    def __init__(self, args_cli, num_hand_joints=16, device=None):
        self.args_cli = args_cli
        self.device = device or ("cuda"
                                 if torch.cuda.is_available() else "cpu")
        self.hand_side = args_cli.hand_side
        self.num_hand_joints = num_hand_joints
        self.joint_limits = self._init_joint_limits()
        self.init_model()

    def _init_joint_limits(self):
        return np.array(
            [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
             [-1.047, 1.047], [-0.47, 2.443], [-1.047, 1.047], [-1.047, 1.047],
             [-0.506, 1.885], [-1.2, 1.9], [-0.506, 1.885], [-0.506, 1.885],
             [-0.366, 2.042], [-1.34, 1.88], [-0.366, 2.042], [-0.366, 2.042]],
            dtype=np.float32)

    def init_model(self, batch_size=2048):
        diffusion_root = self.args_cli.log_dir
        diffusion_loss, diffusion_error_hist, act_diffusion_error_hist = {}, {}, {}
        all_action, all_states = self.init_data()

        for diffusion_source in os.listdir(diffusion_root):
            diffusion_dir = os.path.join(diffusion_root, diffusion_source)
            checkpoint = os.path.join(diffusion_dir, "checkpoints",
                                      "latest.ckpt")
            if os.path.exists(checkpoint) is False:
                print(f"Checkpoint {checkpoint} does not exist, skipping...")
                continue

            payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

            cfg = payload['cfg']

            cls = hydra.utils.get_class(cfg._target_)

            workspace = cls(cfg, args_cli=None)

            workspace: BaseWorkspace
            workspace.load_payload(payload,
                                   exclude_keys=None,
                                   include_keys=None)
            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model

            device = torch.device(self.device)
            policy.to(device)
            policy.eval()
            all_reconstructed = []
            for i in range(0, all_action.shape[0], batch_size):
                start = time.time()
                batch_obs = all_action[i:i + batch_size]  # shape (B, 30, 16)

                obs_dict = {
                    "obs": torch.tensor(batch_obs, device=device).float()
                }
                gt_action = all_action[i:i + batch_size]

                with torch.no_grad():
                    action_pred = policy.predict_action(obs_dict)[
                        "action_pred"]  # shape (B, 30, 16) or similar

                    reconstructed = extract_finger_joints(
                        action_pred.cpu().numpy(), self.joint_limits)

                all_reconstructed.append(reconstructed)

                print(
                    f"Reconstruction time for batch {i // batch_size}: {time.time() - start:.2f} seconds"
                )

            all_reconstructed = np.concatenate(all_reconstructed, axis=0)
            diff = np.abs(all_reconstructed -
                          extract_finger_joints(all_action, self.joint_limits)[
                              :,
                              :all_reconstructed.shape[1],
                          ])
            wrapped = (diff + torch.pi) % (2 * torch.pi) - torch.pi
            sum_error = (np.abs(wrapped).sum(axis=-1) / 16)
            mean_error = sum_error.mean().item()
            key = f"horizon_{cfg.horizon}_nobs_{cfg.n_obs_steps}_naction_{cfg.n_action_steps}_ininference_{cfg.policy.num_inference_steps}"

            act_diffusion_error_hist[key] = sum_error
            diffusion_error_hist[key] = wrapped

            std_error = wrapped.std(axis=1).mean().item()
            print("Diffusion Model:", key)
            print(f"Mean Error: {mean_error:.4f}, Std Error: {std_error:.4f}")

            diffusion_loss[key] = {
                "mean": mean_error,
                "std": std_error,
                "ratio": 0.0,
                "raw_errors": wrapped,
                "horizon": cfg.horizon,
                "n_obs_steps": cfg.n_obs_steps,
                "n_action_steps": cfg.n_action_steps,
                "num_inference_steps": cfg.policy.num_inference_steps,
            }

        self.visualize_diffusion_loss(diffusion_loss)
        self.visualize_diffusion_hist(diffusion_error_hist)

        self.plot_act_error_details(act_diffusion_error_hist)

    def reconstruct(self, model, config, batch_size=2048):
        recon_x_list = []

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

            # Concatenate reconstructions
            recon_x = torch.cat(recon_x_list, dim=0)

            # Compute reconstruction ratio using the final result (or optionally from recon_x)
            ratio = self._compute_ratio(
                model, result)  # Note: this uses only the last batch's result

            if config[2] is None:
                return recon_x, ratio
            else:
                return dataset_denrormalizer(recon_x, config[3],
                                             config[4]), ratio

    def _compute_ratio(self, model, result):
        if model.model_name not in ["HRQdiffusion", "VQdiffusion"]:
            return 0
        flat = result.quantized_indices.view(
            -1, result.quantized_indices.shape[-1])
        return torch.unique(
            flat,
            dim=0).shape[0] / model.quantizer.num_embeddings**flat.shape[-1]

    def load_config(self, model_dir):
        with open(f"{model_dir}/model_config.yaml", "r") as f:
            cfg = yaml.safe_load(f)
        return [
            np.array(cfg["min_latent_value"]),
            np.array(cfg["max_latent_value"]), cfg["data_normalizer"],
            torch.tensor(cfg["action_mean"]).to(self.device),
            torch.tensor(cfg["action_std"]).to(self.device), cfg["latent_dim"],
            cfg["chunk_size"]
        ]

    def init_data(self, chunk_size=40):

        data = h5py.File(self.args_cli.data_dir, "r")["data"]

        all_actions = []
        all_states = []

        for index in range(len(data)):

            raw_actions = data[f"demo_{index}"]["actions"][
                ..., -self.num_hand_joints:]
            chunks_actions, state = sliding_chunks(raw_actions, chunk_size)

            all_actions.append(chunks_actions)
            all_states.append(state)

        all_actions = np.concatenate(all_actions, axis=0)
        all_states = np.concatenate(all_states, axis=0)

        return all_actions, all_states

    def visualize_diffusion_hist(self, diffusion_error_hist):
        import os

        # Step 1: Group errors by horizon → (n_obs, n_action) → n_inference
        grouped = {}
        for name, errors in diffusion_error_hist.items():
            match = re.match(
                r"horizon_(\d+)_nobs_(\d+)_naction_(\d+)_ininference_(\d+)",
                name)
            if not match:
                continue

            horizon, n_obs, n_action, n_infer = map(int, match.groups())
            horizon_key = f"chunk_{horizon}"
            combo_key = f"nobs_{n_obs}_naction_{n_action}"

            if horizon_key not in grouped:
                grouped[horizon_key] = {}
            if combo_key not in grouped[horizon_key]:
                grouped[horizon_key][combo_key] = []

            grouped[horizon_key][combo_key].append({
                "name": name,
                "n_infer": n_infer,
                "errors": errors
            })

        # Step 2: Generate figures
        for horizon_key, combos in grouped.items():
            for combo_key, entries in combos.items():
                entries.sort(key=lambda x: x["n_infer"])
                num_plots = len(entries)
                cols = 4
                rows = math.ceil(num_plots / cols)

                fig, axes = plt.subplots(rows,
                                         cols,
                                         figsize=(5 * cols, 4 * rows),
                                         sharex=False,
                                         sharey=False)
                axes = axes.flatten() if isinstance(axes,
                                                    np.ndarray) else [axes]

                all_errors = np.concatenate([e["errors"]
                                             for e in entries]).reshape(-1)
                global_max_count = max(
                    np.histogram(e["errors"].reshape(-1), bins=50)[0].max()
                    for e in entries)
                y_max = global_max_count * 1.1

                x_min, x_max = min(all_errors), max(all_errors)

                for idx, entry in enumerate(entries):
                    ax = axes[idx]
                    errors = entry["errors"].reshape(-1)
                    bins = np.linspace(np.min(errors), np.max(errors), 50)

                    ax.hist(errors,
                            bins=bins,
                            color='skyblue',
                            edgecolor='black')
                    ax.set_xlim(x_min, x_max)
                    ax.set_ylim(0, y_max)

                    ax.set_title(f"Infer Steps: {entry['n_infer']}",
                                 fontsize=10)
                    ax.set_xlabel(f"Sum Error\n[{x_min:.2f}, {x_max:.2f}]")
                    ax.set_ylabel(f"Freq\n[0, {int(y_max)}]")
                    ax.grid(True)

                # Remove unused subplots
                for idx in range(len(entries), len(axes)):
                    fig.delaxes(axes[idx])

                fig.suptitle(
                    f"Sum Error Distribution\n{horizon_key} | {combo_key}",
                    fontsize=16)
                fig.tight_layout(rect=[0, 0, 1, 0.95])

                save_dir = os.path.join(self.args_cli.log_dir,
                                        f"{horizon_key}")

                os.makedirs(save_dir, exist_ok=True)

                save_path = os.path.join(save_dir, f"{combo_key}_hist.png")
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

    def visualize_diffusion_loss(self, diffusion_loss, log_dir="./logs"):
        # Step 1: Parse keys into rows
        rows = []
        for key, value in diffusion_loss.items():
            match = re.match(
                r"horizon_(\d+)_nobs_(\d+)_naction_(\d+)_ininference_(\d+)",
                key)

            if not match:
                print(f"Skipping unrecognized key: {key}")
                continue

            horizon, nobs, naction, ninfer = map(int, match.groups())
            mean_loss = value.get("mean")
            std_loss = value.get("std")
            ratio = value.get("ratio", None)
            diffusion_type = value.get("type", "diffusion")

            rows.append({
                "horizon": horizon,
                "n_obs_steps": nobs,
                "n_action_steps": naction,
                "n_inference_steps": ninfer,
                "mean_loss": mean_loss,
                "std_loss": std_loss,
                "ratio": ratio,
                "diffusion_type": diffusion_type
            })

        # Step 2: Build dataframe
        df = pd.DataFrame(rows)
        if df.empty:
            print("No valid data found for loss visualization.")
            return

        # Step 3: Group by horizon → create folder → for each (nobs, naction), make heatmap
        for horizon in sorted(df["horizon"].unique()):
            df_h = df[df["horizon"] == horizon]
            horizon_folder = f"chunk_{horizon}"
            save_dir = os.path.join(self.args_cli.log_dir, horizon_folder)
            os.makedirs(save_dir, exist_ok=True)

            for (nobs, naction), df_sub in df_h.groupby(
                ["n_obs_steps", "n_action_steps"]):
                pivot = df_sub.pivot(index="diffusion_type",
                                     columns="n_inference_steps",
                                     values="mean_loss")

                plt.figure(figsize=(10, 6))
                sns.heatmap(pivot, annot=True, fmt=".3f", cmap="coolwarm")
                plt.title(
                    f"Mean Loss Heatmap\nn_obs={nobs}, n_action={naction}, horizon={horizon}"
                )
                plt.xlabel("Num Inference Steps")
                plt.ylabel("Diffusion Type")
                plt.tight_layout()

                filename = f"loss_heatmap_nobs{nobs}_naction{naction}.png"
                plt.savefig(os.path.join(save_dir, filename),
                            bbox_inches="tight")
                plt.close()

    def plot_act_error_details(self, act_diffusion_error_hist):
        for key, errors in act_diffusion_error_hist.items():
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
        default="logs/data_0604/latent_cfm",
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

    diffusion = EVALdiffusionBENCHMARK(args_cli)
