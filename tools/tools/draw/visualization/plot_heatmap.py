import matplotlib.pyplot as plt
import numpy as np

# First heatmap data (original)
data1_orig = np.array([[9 / 10, 10 / 10], [8 / 10, 7 / 10], [10 / 10,
                                                             10 / 10]])

# Second heatmap data (original)
data2_orig = np.array([[2 / 10, 3 / 10], [5 / 10, 6 / 10], [1 / 2, 1 / 3]])

# Scale down intensity for plotting (but keep original for text)
data1 = (data1_orig * 0.5).T  # transpose to 2x3
data2 = (data2_orig * 0.5).T  # transpose to 2x3

fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # taller figure

# Shared color scale
vmin, vmax = 0, 1
cmap = "YlOrRd"

# Coordinate extents
xmin, xmax = -0.25, 0.25
ymin, ymax = -0.1, 0.1
extent = [xmin, xmax, ymin, ymax]

# Compute exact cell centers
nx, ny = data1.shape[1], data1.shape[0]  # (3 cols, 2 rows)
x_edges = np.linspace(xmin, xmax, nx + 1)
y_edges = np.linspace(ymin, ymax, ny + 1)
x_centers = (x_edges[:-1] + x_edges[1:]) / 2
y_centers = (y_edges[:-1] + y_edges[1:]) / 2

# Desired ticks
x_ticks = [-0.2, 0.0, 0.2]
y_ticks = [-0.1, 0.0, 0.1]

# First heatmap
im1 = axes[0].imshow(data1,
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     extent=extent,
                     origin="lower",
                     aspect="auto")
axes[0].set_title("Ours (RFS)")
axes[0].set_xlabel("x [m]")
axes[0].set_ylabel("y [m]")
axes[0].set_xticks(x_ticks)
axes[0].set_yticks(y_ticks)

for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        axes[0].text(x,
                     y,
                     f"{data1_orig[j, i]:.2f}",
                     ha="center",
                     va="center",
                     color="black")

# Second heatmap
im2 = axes[1].imshow(data2,
                     cmap=cmap,
                     vmin=vmin,
                     vmax=vmax,
                     extent=extent,
                     origin="lower",
                     aspect="auto")
axes[1].set_title("Zero-Shot")
axes[1].set_xlabel("x [m]")
axes[1].set_ylabel("y [m]")
axes[1].set_xticks(x_ticks)
axes[1].set_yticks(y_ticks)

for i, y in enumerate(y_centers):
    for j, x in enumerate(x_centers):
        axes[1].text(x,
                     y,
                     f"{data2_orig[j, i]:.2f}",
                     ha="center",
                     va="center",
                     color="black")

# Add a dedicated colorbar axis below all subplots
cbar_ax = fig.add_axes([0.25, 0.08, 0.5,
                        0.03])  # [left, bottom, width, height]

# ✅ Colorbar uses capped normalization (max color at 0.5)
import matplotlib.colors as mcolors

cbar = fig.colorbar(im1,
                    cax=cbar_ax,
                    orientation="horizontal",
                    norm=mcolors.Normalize(vmin=0, vmax=0.5))
# cbar.set_label("Value (normalized)")

plt.tight_layout(rect=[0, 0.1, 1, 1])  # leave space at bottom for colorbar
plt.savefig("logs/trash/heatmap_comparison.png", dpi=300)
plt.show()
