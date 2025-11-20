import os
import shutil
import numpy as np
import h5py


def normalize_action(log_dir, h5py_file_path, copied_file_path):

    if os.path.exists(copied_file_path):
        os.remove(copied_file_path)
    shutil.copy(h5py_file_path, copied_file_path)

    # Open the copied file for modification
    with h5py.File(copied_file_path, 'r+') as h5py_file:
        actions_buffer = []

        # Concatenate all actions from all demos
        for demo_id in range(len(h5py_file["data"].keys())):

            h5py_file["data"][f"demo_{demo_id}"]["actions"][:, -1] = np.sign(
                h5py_file["data"][f"demo_{demo_id}"]["actions"][:, -1] + 0.2)
            actions = h5py_file["data"][f"demo_{demo_id}"]["actions"]
            actions_buffer.append(actions)

        all_actions = np.concatenate(actions_buffer, axis=0)

        # Calculate min and max for normalization
        stats = {
            "action": {
                "min": all_actions.min(axis=0),
                "max": all_actions.max(axis=0),
            }
        }

        # Save stats to a separate file
        np.save(log_dir + "/stats.npy", stats)

        # Normalize actions for each demo and save them to the copied HDF5 file
        for demo_id in range(len(h5py_file["data"].keys())):
            actions = h5py_file["data"][f"demo_{demo_id}"]["actions"]
            actions[..., :-1] *= scale

            # Normalize the actions using the calculated stats
            actions_buffer = normalize(actions, stats["action"])

            # Delete the existing dataset and replace with the normalized actions
            del h5py_file["data"][f"demo_{demo_id}"]["actions"]
            h5py_file["data"][f"demo_{demo_id}"].create_dataset(
                "actions", data=actions_buffer)

    return copied_file_path


def normalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 2 * (arr - min_val) / (max_val - min_val) - 1


normalize_action("log/1030", "log/1030/grasp.hdf5",
                 "log/1030/grasp_normalize.hdf5")
