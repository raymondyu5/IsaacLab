import numpy as np
import matplotlib.pyplot as plt

# --- Baseline methods only ---
methods = ["BC w/ real data(50 demos)", "Co-training", "Zero-shot Sim2Real"]

# --- Unknown objects data (baselines only) ---
unknown_means = np.array([35.0, 47.5, 30.0])
unknown_stds = np.array([5.0, 28.3, 12.0])

# --- Bar positions ---
x = np.arange(len(methods))
height = 0.6

# --- Create figure ---
fig, ax = plt.subplots(figsize=(8, 4))

# --- Plot bars ---
bars = ax.barh(x,
               unknown_means,
               height,
               xerr=unknown_stds,
               capsize=5,
               color="darkorange",
               label="Unknown Objects")

# --- Add value labels ---
for rect, mean, std in zip(bars, unknown_means, unknown_stds):
    ax.text(mean + std + 1,
            rect.get_y() + rect.get_height() / 2,
            f"{mean:.1f}",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="red")

# --- Axis labels and formatting ---
ax.set_title("Unknown Objects (Mean ± Std)", fontsize=14)
ax.set_xlabel("Success Rate (%)", fontsize=12)
ax.set_yticks(x)
ax.set_yticklabels(methods, fontsize=13)

# Make it look like left-side labels (no right-side duplication)
ax.yaxis.set_tick_params(pad=10)
ax.invert_yaxis()  # match same top-to-bottom order as full figure

ax.set_xlim(0, 100)
ax.set_xticks(np.linspace(0, 100, 11))
ax.set_facecolor("#eaeaf2")
ax.grid(True, axis="x", color="white", linewidth=0.3)

plt.tight_layout()
plt.savefig("logs/real_error_bar_unknown_baselines.png", dpi=300)
plt.show()
