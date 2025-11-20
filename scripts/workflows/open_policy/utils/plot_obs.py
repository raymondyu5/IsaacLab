import h5py
import numpy as np
import matplotlib.pyplot as plt

data = h5py.File(
    "/home/ensu/Documents/weird/IsaacLab/logs/droid/planner_data.hdf5",
    "r")["data"]
obs_keys = ["ee_pose"]
all_obs = []
for demo_key in data.keys():
    obs_list = []
    demo = data[demo_key]

    obs = demo["obs"]
    actions = demo["actions"]
    for obs_key in obs_keys:
        obs_list.append(np.array(obs[obs_key]))
    all_obs.append(np.concatenate(obs_list, axis=1))

all_obs = np.concatenate(all_obs, axis=0)

plt.figure(figsize=(12, 6))
for i in range(all_obs.shape[1]):
    plt.plot(all_obs[:, i], label=f'Dim {i+1}',
             alpha=0.6)  # Add transparency for clarity

plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.title("15-Dimensional Observations Over Time")
plt.legend(loc="upper right", ncol=3)
plt.show()
