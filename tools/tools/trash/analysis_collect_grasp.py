import matplotlib.pyplot as plt
import numpy as np
import h5py

normalized_grasp = h5py.File(f"logs/1107_place/grasp.hdf5", 'r+')
data = []
for i in range(len(normalized_grasp["data"])):
    demo = normalized_grasp["data"][f"demo_{i}"]

    data.append(np.array(demo["obs"]["mug_pose"][0][:2]))

data = np.array(data)
# Extract x and y coordinates
x = data[:, 0]
y = data[:, 1]

# Plot
plt.figure(figsize=(8, 6))
plt.scatter(x, y, c='blue', s=1000, marker='o', label='Object locations')
plt.xlabel('X coordinate')
plt.ylabel('Y coordinate')
plt.title('2D Visualization of Object Locations')
plt.legend()
plt.grid(True)
plt.show()
