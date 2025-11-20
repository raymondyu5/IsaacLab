import numpy as np
import matplotlib.pyplot as plt

# --- All methods ---
all_methods = [
    "Zero-shot Sim2Real", "Co-training", "BC w/ real data(50 demos)",
    r"Residual RL: $Q(\mathrm{a}_r, o)$",
    r"RFS: $Q([\mathrm{a}_r,\mathrm{a}_b], o)$",
    r"Residual RL: $Q(\mathrm{a}_r+\mathrm{a}_b, o)$",
    r"DSRL: $Q(\mathrm{a}_b, o)$",
    r"RFS(Ours): $Q(\mathrm{a}_r+\mathrm{a}_b, o)$"
]

# --- Data arrays (aligned with all_methods) ---
known_means = np.array([43.3, 83.3, 73.3, 0.0, 0.0, 70.0, 65.0, 80.0])
unknown_means = np.array([30.0, 47.5, 35.0, 0.0, 0.0, 48.0, 47.5, 74.0])
known_stds = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5, 4.7])
unknown_stds = np.array([12.0, 28.3, 5, 0.0, 0.0, 14.8, 4.3, 14.1])

# === Select subset of methods to visualize ===
# (indices for "Zero-shot Sim2Real", "Co-training", "BC w/ real data(50 demos)", "RFS(Ours)")
selected_indices = [0]

# Filter arrays and method names
methods = [all_methods[i] for i in selected_indices]
known_means = known_means[selected_indices]
unknown_means = unknown_means[selected_indices]
known_stds = known_stds[selected_indices]
unknown_stds = unknown_stds[selected_indices]


# --- Compute statistics ---
def summarize(values, label, ignore_zeros=True):
    arr = values.copy()
    if ignore_zeros:
        arr = arr[arr > 0]
    return f"{label}: mean = {arr.mean():.2f}, std = {arr.std():.2f}, n = {len(arr)}"


print("\n=== Real-world Evaluation Summary ===")
print(summarize(known_means, "Known Objects"))
print(summarize(unknown_means, "Unknown Objects"))

# --- Plotting ---
x = np.arange(len(methods))
height = 0.1

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

# Known objects subplot
bars = axes[0].barh(x,
                    known_means,
                    height,
                    xerr=known_stds,
                    capsize=5,
                    color="steelblue",
                    label="Known Objects")
for rect, mean, std in zip(bars, known_means, known_stds):
    axes[0].text(mean + std + 2,
                 rect.get_y() + rect.get_height() / 2,
                 f"{mean:.1f}",
                 va="center",
                 fontsize=14,
                 fontweight="bold",
                 color="red")

# Unknown objects subplot
bars = axes[1].barh(x,
                    unknown_means,
                    height,
                    xerr=unknown_stds,
                    capsize=5,
                    color="darkorange",
                    label="Unknown Objects")

for rect, mean, std in zip(bars, unknown_means, unknown_stds):
    axes[1].text(mean + std + 2,
                 rect.get_y() + rect.get_height() / 2,
                 f"{mean:.1f}",
                 va="center",
                 fontsize=14,
                 fontweight="bold",
                 color="red")

# Axis settings
axes[0].set_title("Known Objects (Mean ± Std)")
axes[0].set_xlabel("Success Rate (%)")
# axes[0].set_yticks(x)
# axes[0].set_yticklabels(methods, fontsize=14)
axes[0].yaxis.set_visible(False)
axes[1].set_title("Unknown Objects (Mean ± Std)")
axes[1].set_xlabel("Success Rate (%)")
# axes[1].set_yticks(x)
# axes[1].set_yticklabels([])
axes[1].yaxis.set_visible(False)

for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_xticks(np.linspace(0, 100, 11))
    ax.set_facecolor("#eaeaf2")
    ax.grid(True, axis="x", color="white", linewidth=0.3)

# # --- Add separation line after first 3 methods ---
# sep_idx = 3  # separator after first three methods
# for ax in axes:
#     ax.axhline(y=sep_idx - 0.5, color="black", linewidth=2.5, linestyle="--")

plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.savefig("logs/real_error_bar_subset.png", dpi=300)
plt.show()
