import h5py
import numpy as np
from scripts.workflows.utils.client.openvla_client import resize_image

all_raw_actions = []
with h5py.File("logs/replay_rl_data.hdf5", 'r') as f:
    raw_data = f["data"]
    for key in raw_data:
        obs = raw_data[key]["obs"]
        gs_image = np.array(obs["gs_image"])[:, 0]
        action = np.array(obs["actions"])
        all_raw_actions.append(action)

        resize_images = [resize_image(img, (224, 224)) for img in gs_image]
        np.savez(f"logs/data/{key}.npz", gs_image=resize_images, action=action)

all_raw_actions = np.concatenate(all_raw_actions, axis=0)
dataset_statistics = {
    "dummy_dataset": {
        "action": {
            "q01": np.min(all_raw_actions, 0),
            "q99": np.max(all_raw_actions, 0)
        }
    }
}
np.save("logs/data/statistics", dataset_statistics)
