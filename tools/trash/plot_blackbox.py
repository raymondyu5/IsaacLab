import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append(".")
from scripts.workflows.sysID.ASID.tool.utilis import *

# Initialize lists to store data
target_properties = []
rollout_properties = []
rollout_properties_std = []

# Load data
for i in range(10):
    data = np.load(f"logs/cem/rabbit/result/{i}.npy")
    target_properties.append(data[:2])
    rollout_properties.append(data[2:4])
    rollout_properties_std.append(data[4:6])

# Convert lists to numpy arrays
target_properties = np.array(target_properties)
rollout_properties = np.array(rollout_properties)
rollout_properties_std = np.array(rollout_properties_std)

# Plot target properties (blue line)
plt.plot(target_properties[:, 0],
         target_properties[:, 1],
         label='Target Properties',
         color='blue')

plt.plot(rollout_properties[:, 0],
         rollout_properties[:, 1],
         label='Rollout Properties',
         color='red')

# Plot predicted properties with error bars (red line with error bars)
plt.errorbar(
    rollout_properties[:, 0],
    rollout_properties[:, 1],
    xerr=rollout_properties_std[:, 0],
    yerr=abs(rollout_properties_std[:, 1]),
    label='Predicted Properties',
    color='red',
    fmt='o',  # Marker for points
    capsize=5)  # Add caps to the error bars

# Add labels and legend
plt.xlabel('Property 1')
plt.ylabel('Property 2')
plt.legend()

# Show the plot
plt.show()
