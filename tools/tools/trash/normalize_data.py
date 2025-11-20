import os
import shutil
import numpy as np
import h5py


def sample_train_test(h5_file):
    if "mask" in h5_file.keys():
        del h5_file["mask"]
    grp_mask = h5_file.create_group("mask")

    data = h5_file["data"]
    num_demo = len(data)
    training_set_size = int(0.9 * num_demo)

    # Generate randomized indices for training and testing
    all_indices = np.arange(num_demo)
    training_indices = np.random.choice(all_indices,
                                        size=training_set_size,
                                        replace=False)
    test_indices = np.setdiff1d(all_indices, training_indices)

    # Create the demo keys based on randomized indices
    demo_key = {
        "train": ["demo_" + str(i) for i in training_indices],
        "test": ["demo_" + str(i) for i in test_indices]
    }

    # Save the train and test sets in the HDF5 file
    grp_mask.create_dataset("train",
                            data=np.array(demo_key["train"], dtype='S'))
    grp_mask.create_dataset("test", data=np.array(demo_key["test"], dtype='S'))


def unnormalize(arr, stats):

    min_val, max_val = stats["min"], stats["max"]
    arr = np.array(arr)
    result = 0.5 * (arr + 1) * (max_val - min_val) + min_val

    return result


def normalize_action(log_dir, h5py_file_path, copied_file_path):

    if os.path.exists(copied_file_path):
        os.remove(copied_file_path)
    shutil.copy(h5py_file_path, copied_file_path)

    # Open the copied file for modification
    with h5py.File(copied_file_path, 'r+') as h5py_file:
        actions_buffer = []

        # # Concatenate all actions from all demos
        for demo_id in range(len(h5py_file["data"].keys())):

            # h5py_file["data"][f"demo_{demo_id}"]["actions"][:, -1] = np.sign(
            #     h5py_file["data"][f"demo_{demo_id}"]["actions"][:, -1] + 0.2)
            actions = np.array(h5py_file["data"][f"demo_{demo_id}"]["actions"])

            actions_buffer.append(actions.reshape(-1, actions.shape[-1]))

        all_actions = np.concatenate(actions_buffer, axis=0)

        # stats = np.load(log_dir + "/grasp_stats.npy", allow_pickle=True).item()
        # all_actions[:, :3] = unnormalize(all_actions[:, :3], stats["action"])

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

            import copy
            actions_buffer = copy.deepcopy(np.array(actions))

            # Normalize the actions using the calculated stats
            actions_buffer = normalize(actions, stats["action"])

            # Delete the existing dataset and replace with the normalized actions
            del h5py_file["data"][f"demo_{demo_id}"]["actions"]

            h5py_file["data"][f"demo_{demo_id}"].create_dataset(
                "actions", data=actions_buffer.reshape(-1, actions.shape[-1]))
    sample_train_test(h5py.File(copied_file_path, 'r+'))
    return copied_file_path


def normalize(arr, stats):
    min_val, max_val = stats["min"], stats["max"]
    return 2 * (arr - min_val) / (max_val - min_val) - 1


normalize_action("logs/0227_planner_data",
                 "logs/0227_planner_data/planner_data.hdf5",
                 "logs/0227_planner_data/normalized_planner_data.hdf5")
