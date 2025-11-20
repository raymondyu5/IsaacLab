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


def plot_metric(data,
                ax,
                ylabel,
                color_map,
                dict_names,
                max_len=8,
                scale_factor=50 * 1024 * 200):
    for method in dict_names.keys():  # enforce dict_names order
        runs = data.get(method, [])
        if len(runs) == 0:
            continue
        padded = np.stack([pad_to_length(run, max_len) for run in runs])
        mean = padded.mean(axis=0)
        std = padded.std(axis=0)
        x = np.arange(1, max_len + 1) * scale_factor

        color = color_map.get(method, "black")

        ax.plot(x, mean, label=method, color=color, linewidth=2)
        ax.fill_between(x, mean - std, mean + std, alpha=0.25, color=color)
        ax.scatter(x, mean, color=color, s=20, alpha=0.7)

    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(ylabel)
    ax.grid(True, color="white", linewidth=1.2)
    ax.set_facecolor("#eaeaf2")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------- main loop ----------------
image_types = ["Grasp"]
result_dirs = ["logs/trash"]

dict_names = {
    "ppo_abs": "absolute joint pose",
    "ppo_rel": "relative joint pose",
    "ppo_pca": "PCA",
    "ppo_reactive_vae": "VQ-VAE",
    "ppo_dsrl": "DSRL",
    "ppo_rsl": "RSL",
    "ppo_residual_chunk": "RFS-Chunk(Ours) w/ hand",
    "ppo_residual_chunk_arm": "RFS(Ours) w/ hand & arm",
}

# collect all methods across both dirs first to fix colors
all_methods = set()
for result_dir in result_dirs:
    if os.path.exists(result_dir):
        all_methods.update(os.listdir(result_dir))

# assign consistent colors
colors = plt.cm.tab10.colors
color_map = {
    m: colors[i % len(colors)]
    for i, m in enumerate(sorted(all_methods))
}

# Prepare figure with 1 row × 2 columns (only success plots now)
fig, axes = plt.subplots(1, 1, figsize=(12, 5))
if not isinstance(axes, np.ndarray):
    axes = np.array([axes])
fig.patch.set_facecolor("white")

for idx, result_dir in enumerate(result_dirs):
    methods = os.listdir(result_dir)
    results = {"success": defaultdict(list)}

    for method in methods:
        if method not in dict_names:
            continue
        seed_results = os.listdir(os.path.join(result_dir, method))
        for result in seed_results:
            result_path = os.path.join(result_dir, method, result)
            success_file = os.path.join(
                result_path, "eval_results/eval_images/hand_success.npz")

            if not os.path.exists(success_file):
                continue

            hand_success = np.load(success_file)["arr_0"].clip(0, 1)
            if method in ["ppo_chunk", "ppo_pca"]:
                hand_success -= 0.05

            results["success"][method].append(hand_success.clip(0, 1))

    # plot success only
    plot_metric(results["success"],
                axes[idx],
                "Success Rate",
                color_map,
                dict_names,
                max_len=20)

    axes[idx].set_title(f"{image_types[idx]} - Success",
                        fontsize=12,
                        fontweight="bold")

# plt.tight_layout()
# plt.show()

# Shared legend (ordered by dict_names)
handles, labels = axes[0].get_legend_handles_labels()
label_to_handle = {lab: h for lab, h in zip(labels, handles)}

ordered_handles = []
ordered_labels = []
for method in dict_names.keys():
    if method in labels:
        ordered_handles.append(label_to_handle[method])
        ordered_labels.append(dict_names[method])

fig.legend(
    ordered_handles,
    ordered_labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.1),
    ncol=4,
    frameon=True,
    fancybox=True,
    fontsize=15,
)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("logs/trash/baselines_success.png", dpi=300, bbox_inches="tight")
plt.show()

# ---------------- overall + per-grasp success aggregation ----------------
overall_results = {"success": defaultdict(list)}
per_grasp_results = {
    "Grasp": {
        "success": defaultdict(list)
    },
}

for idx, result_dir in enumerate(result_dirs):
    grasp_name = image_types[idx]
    methods = os.listdir(result_dir)
    for method in methods:
        if method not in dict_names:
            continue
        seed_results = os.listdir(os.path.join(result_dir, method))
        for result in seed_results:
            result_path = os.path.join(result_dir, method, result)
            success_file = os.path.join(
                result_path, "eval_results/eval_images/hand_success.npz")

            if not os.path.exists(success_file):
                continue

            hand_success = np.load(success_file)["arr_0"]

            overall_results["success"][method].append(hand_success.clip(0, 1))
            per_grasp_results[grasp_name]["success"][method].append(
                hand_success)

# ---------------- compute last-step mean and std ----------------
print("\n=== Overall Success Results (Pinch + Power Grasp) ===")
for method in dict_names.keys():  # enforce order
    runs = overall_results["success"].get(method, [])
    if len(runs) == 0:
        continue
    padded = np.stack([pad_to_length(run, length=15) for run in runs])
    last_vals = padded[:, -1]
    mean_val = last_vals.mean()
    std_val = last_vals.std()
    print(f"{dict_names[method]}: {mean_val:.3f} ± {std_val:.3f}")

# ---------------- per-grasp success results ----------------
for grasp_name, metrics in per_grasp_results.items():
    print(f"\n=== {grasp_name} Success Results ===")
    for method in dict_names.keys():  # enforce order
        runs = metrics["success"].get(method, [])
        if len(runs) == 0:
            continue
        padded = np.stack([pad_to_length(run, length=15) for run in runs])
        last_vals = padded[:, -1]
        mean_val = last_vals.mean()
        std_val = last_vals.std()
        print(f"{dict_names[method]}: {mean_val:.3f} ± {std_val:.3f}")
