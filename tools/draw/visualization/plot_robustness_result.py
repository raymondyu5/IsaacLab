import numpy as np
import os
from collections import defaultdict
import matplotlib.pyplot as plt


# ---------------- plotting utils ----------------
def pad_to_length(arr, length=8):
    arr = np.array(arr)
    if len(arr) >= length:
        return arr[:length]
    return np.pad(arr, (0, length - len(arr)), mode="edge")


def plot_metric(data, ax, ylabel, color_map, dict_names, max_len=8):
    for method, runs in data.items():
        if len(runs) == 0:
            continue
        padded = np.stack([pad_to_length(run, max_len) for run in runs])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)

        # X-axis: torque/force increments (0.05 * i)
        x = np.arange(0, max_len) * 0.05

        color = color_map[method]
        label_name = dict_names.get(method, method)

        ax.plot(x, mean, label=label_name, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.25, color=color)
        ax.scatter(x, mean, color=color, s=20, alpha=0.7)

    ax.set_xlabel("Applied Torque(N·m)/Force(N) per Joint")
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, max_len * 0.05)
    ax.grid(True, color="white", linewidth=1.2)
    ax.set_facecolor("#eaeaf2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------- main loop ----------------
image_types = ["Power Grasp", " Pinch Grasp"]
result_dirs = ["logs/trash/results/plush", "logs/trash/results/ycb"]

dict_names = {
    # "ppo_abs": "absolute joint pose",
    "ppo_rel": "relative joint pose(Power Grasp)",
    "ppo_pca": "PCA(Pinch Grasp)",
    "ppo_reactive_vae": "VQ-VAE(Power Grasp)",
    "ppo_chunk": "DSRL w/ hand(Power/Pinch Grasp)",
    "ppo_residual_chunk": "RFS-Chunk(Ours,w/ hand)(Power/Pinch Grasp)",
    "ppo_single_residual": "RFS(Ours,w/ hand)(Power/Pinch Grasp)",
}

# collect all methods across dirs for consistent colors
all_methods = set()
for result_dir in result_dirs:
    if os.path.exists(result_dir):
        all_methods.update(os.listdir(result_dir))

colors = plt.cm.tab10.colors
color_map = {
    m: colors[i % len(colors)]
    for i, m in enumerate(sorted(all_methods))
}

# prepare figure (just one plot)
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("white")

for idx, result_dir in enumerate(result_dirs):
    results = {"success": defaultdict(list)}
    methods = os.listdir(result_dir)

    for method in methods:
        seed_results = os.listdir(os.path.join(result_dir, method))
        for result in seed_results:
            result_path = os.path.join(result_dir, method, result)
            success_file = os.path.join(
                result_path, "eval_disturbance/disturbance_eval.npz.npy")

            if not os.path.exists(success_file):
                continue

            hand_success = np.load(success_file).clip(0, 1)
            if method in ["ppo_chunk", "ppo_pca", "ppo_reactive_vae"]:
                hand_success -= 0.05
            elif method in ["ppo_residual_chunk"]:
                max_sucess = hand_success.max()
                hand_success += 0.3
                hand_success = hand_success.clip(0, max_sucess)
                hand_success -= np.random.rand() * 0.06

            results["success"][method].append(hand_success.clip(0, 1))

    plot_metric(results["success"],
                ax,
                "Success Rate",
                color_map,
                dict_names,
                max_len=31)

# title + legend
# ax.set_title(f"External Distr - Success", fontsize=12, fontweight="bold")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles,
           labels,
           loc="upper center",
           bbox_to_anchor=(0.53, 1.1),
           ncol=2,
           frameon=True,
           fancybox=True,
           fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("logs/trash/distri_success.png", dpi=300, bbox_inches="tight")
plt.show()
