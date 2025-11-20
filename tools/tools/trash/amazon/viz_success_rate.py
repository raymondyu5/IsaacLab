import numpy as np
import matplotlib.pyplot as plt
import os

file_list = [
    "logs/0515/mappo_bimanual_non_shared_seed100",
    "logs/0515/mappo_bimanual_non_shared_seed300",
    "logs/0515/mappo_bimanual_shared_seed100",
    "logs/0515/mappo_bimanual_shared_seed300", "logs/0515/ppo_right_vae",
    "logs/ppo_right", "logs/ippo_bimanual_seed300", "logs/ippo_pca"
]
class_name = [
    "MAPPO-Separate-1", "MAPPO-Separate-2", "MAPPO-Shared-1", "MAPPO-Shared-2",
    "PPO-VAE-1", "RL-Scratch-1", "RL-Scratch-2", "PPO-PCA-1"
]


def safe_load_npz(file_path):
    if os.path.exists(file_path):
        data = np.load(file_path)
        return data["arr_0"]
    return None


# ------------------- Left Hand Success -------------------
plt.figure(figsize=(8, 4))
for i, file in enumerate(file_list):
    success = safe_load_npz(os.path.join(file, "left_hand_success.npz"))
    if success is not None:
        x = np.arange(len(success)) * 20
        mask = x <= 1000
        plt.plot(x[mask], success[mask], label=class_name[i])
plt.title("Left Hand Success Rate")
plt.xlabel("Evaluation Epoch (×20)")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------- Left Hand Reward -------------------
plt.figure(figsize=(8, 4))
for i, file in enumerate(file_list):
    reward = safe_load_npz(os.path.join(file, "left_hand_rollout.npz"))
    if reward is not None:
        x = np.arange(len(reward)) * 20
        mask = x <= 1000
        reward = reward / np.max(reward)
        plt.plot(x[mask], reward[mask], label=class_name[i])
plt.title("Left Hand Normalized Rollout Reward")
plt.xlabel("Training Iteration (×20)")
plt.ylabel("Normalized Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------- Right Hand Success -------------------
plt.figure(figsize=(8, 4))
for i, file in enumerate(file_list):
    success = safe_load_npz(os.path.join(file, "right_hand_success.npz"))
    if success is not None:
        x = np.arange(len(success)) * 20
        mask = x <= 1000
        plt.plot(x[mask], success[mask], label=class_name[i])
plt.title("Right Hand Success Rate")
plt.xlabel("Evaluation Epoch (×20)")
plt.ylabel("Success Rate")
plt.ylim(0, 1)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ------------------- Right Hand Reward -------------------
plt.figure(figsize=(8, 4))
for i, file in enumerate(file_list):
    reward = safe_load_npz(os.path.join(file, "right_hand_rollout.npz"))
    if reward is not None:
        x = np.arange(len(reward)) * 20
        mask = x <= 1000
        reward = reward / np.max(reward)
        plt.plot(x[mask], reward[mask], label=class_name[i])
plt.title("Right Hand Normalized Rollout Reward")
plt.xlabel("Training Iteration (×20)")
plt.ylabel("Normalized Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
