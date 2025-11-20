import numpy as np
import matplotlib.pyplot as plt

# Methods (7 entries)
methods = [
    "Zero-shot Sim2Real",
    # "Co-training",
    # "BC w/ real data(50 demos)",
    # "BC w/ real data(200 demos)",
    # r"TD3+BC: $Q(\mathrm{a}_r, o)$ w/ $\pi(\mathrm{a}_r|o)$",
    # r"TD3+BC: $Q([\mathrm{a}_r,\mathrm{a}_b], o)$ w/ $\pi(\mathrm{a}_r|o)$",
    # r"TD3+BC: $Q(\mathrm{a}_r+\mathrm{a}_b, o)$ w/ $\pi(\mathrm{a}_r|o)$",
    # r"TD3+BC: $Q(\mathrm{a}_r+\mathrm{a}_b, o)$ w/ $\pi(z|o)$",
    # r"TD3+BC: Residual Q  w/ $\pi(\mathrm{a}_r,z|o)$"
    r"Residual RL: $Q(\mathrm{a}_r, o)$",
    r"Residual RL: $Q([\mathrm{a}_r,\mathrm{a}_b], o)$",
    r"Residual RL: $Q(\mathrm{a}_r+\mathrm{a}_b, o)$",
    r"DSRL: $Q(\mathrm{a}_r+\mathrm{a}_b, o)$",
    r"RFS(Ours): $Q(\mathrm{a}_r+\mathrm{a}_b, o)$"
]

# Data arrays
known_means = np.array([43.3, 83.3, 73.3, 0.0, 0.0, 70.0, 65.0, 80.0])
unknown_means = np.array([30.0, 47.5, 35.0, 0.0, 0.0, 48.0, 47.5, 74.0])
known_stds = np.array([10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5, 4.7])
unknown_stds = np.array([12.0, 28.3, 5, 0.0, 0.0, 14.8, 4.3, 14.1])


# --- Compute statistics ---
def summarize(values, label, ignore_zeros=True):
    arr = values.copy()
    if ignore_zeros:
        arr = arr[arr > 0]
    return f"{label}: mean = {arr.mean():.2f}, std = {arr.std():.2f}, n = {len(arr)}"


print("\n=== Real-world Evaluation Summary ===")
print(summarize(known_means, "Known Objects"))
print(summarize(unknown_means, "Unknown Objects"))
# --- Plotting into two subplots ---
x = np.arange(len(methods))
height = 0.6  # single bar height per subplot

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True)

# Known objects subplot (horizontal)
bars = axes[0].barh(x,
                    known_means,
                    height,
                    xerr=known_stds,
                    capsize=5,
                    color="steelblue",
                    label="Known Objects")
for rect, mean, std in zip(bars, known_means, known_stds):
    axes[0].text(
        mean + std + 2,  # shift to the right
        rect.get_y() + rect.get_height() / 2,
        f"{mean:.1f}",
        va="center",
        fontsize=15,
        fontweight="bold",
        color="red")

# Unknown objects subplot (horizontal)
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
                 fontsize=15,
                 fontweight="bold",
                 color="red")

# Axis settings
axes[0].set_title("Known Objects (Mean ± Std)")
axes[0].set_xlabel("Success Rate (%)")
axes[0].set_yticks(x)
axes[0].set_yticklabels(methods, fontsize=20)

axes[1].set_title("Unknown Objects (Mean ± Std)")
axes[1].set_xlabel("Success Rate (%)")
axes[1].set_yticks(x)
axes[1].set_yticklabels([])  # hide duplicate labels

for ax in axes:
    ax.set_xlim(0, 100)
    ax.set_xticks(np.linspace(0, 100, 11))
    ax.set_facecolor("#eaeaf2")
    ax.grid(True, axis="x", color="white", linewidth=0.3)

# --- After plotting bars, add separation line ---
sep_idx = 3  # after first 3 methods

for ax in axes:
    ax.axhline(
        y=sep_idx - 0.5,  # horizontal line between groups
        color="black",
        linewidth=3.0,
        linestyle="--")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("logs/real_error_bar.png", dpi=300)
plt.show()
