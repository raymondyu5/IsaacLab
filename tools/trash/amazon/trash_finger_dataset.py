import argparse
import h5py
import os
import numpy as np
import glob
import torch
import pickle

parser = argparse.ArgumentParser(description="Merge a set of HDF5 datasets.")
parser.add_argument(
    "--input_files",
    type=str,
    nargs="+",
    default=["/media/ensu/data/datasets/teleop_data"],
    help="A list of paths to HDF5 files to merge.",
)
parser.add_argument("--output_file",
                    type=str,
                    default="logs/data_0630",
                    help="File path to merged output.")

args_cli = parser.parse_args()

joint_limits = np.array(
    [[-0.314, 2.23], [-0.349, 2.094], [-0.314, 2.23], [-0.314, 2.23],
     [-1.047, 1.047], [-0.46999997, 2.4429998], [-1.047, 1.047],
     [-1.047, 1.047], [-0.5059999, 1.8849999], [-1.2, 1.8999999],
     [-0.5059999, 1.8849999], [-0.5059999, 1.8849999], [-0.366, 2.0419998],
     [-1.34, 1.8799999], [-0.366, 2.0419998], [-0.366, 2.0419998]],
    dtype=np.float32)


def extension_actions(actions):
    init_actions = actions[0]
    last_actions = actions[-1]

    num_init_frames = np.random.randint(int(len(actions) * 0.1),
                                        int(len(actions) * 0.2))
    num_final_frames = np.random.randint(int(len(actions) * 0.2),
                                         int(len(actions) * 0.4))
    # num_init_frames = np.random.randint(int(len(actions) * 0.1),
    #                                     int(len(actions) * 0.2))
    # num_final_frames = np.random.randint(int(len(actions) * 0.1),
    #                                      int(len(actions) * 0.2))

    # Frozen frames at the beginning
    n_frozen_init = np.random.randint(6, 10)
    frozen_init_extension = init_actions[np.newaxis, :].repeat(n_frozen_init,
                                                               axis=0) * 0.0

    # Interpolate init actions
    init_factors = np.arange(0, num_init_frames).reshape(-1,
                                                         1) / num_init_frames
    exteneded_init_actions = init_factors * init_actions  # shape (num_init_frames, action_dim)

    # Final actions
    final_factors = np.arange(0, num_final_frames).reshape(-1, 1)
    exteneded_final_actions = np.zeros((num_final_frames, actions.shape[1]),
                                       dtype=np.float32)

    exteneded_final_actions[:, -16:] = final_factors / num_final_frames * (
        joint_limits[:, 1] * 0.6 - last_actions[-16:]) + last_actions[-16:]

    # Frozen final frames
    n_frozen_final = np.random.randint(16, 30)
    final_action_to_freeze = exteneded_final_actions[-1:, :]
    frozen_final_extension = final_action_to_freeze.repeat(n_frozen_final,
                                                           axis=0)

    # Combine everything
    extended_actions = np.concatenate([
        frozen_init_extension, exteneded_init_actions, actions,
        exteneded_final_actions, frozen_final_extension
    ],
                                      axis=0)

    return extended_actions


def merge_datasets():
    for filepath in args_cli.input_files:
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"The dataset file {filepath} does not exist.")

    all_npz_files = []
    for file in args_cli.input_files:
        all_npz_files.extend(glob.glob(file + "/**/*.npz", recursive=True))
    os.makedirs(args_cli.output_file, exist_ok=True)

    with h5py.File(args_cli.output_file + f"/raw_right_data.hdf5",
                   "w") as output:
        output_data_group = output.create_group("data")
        episode_idx = 0
        copy_attributes = True
        num_demo = 0

        for filepath in all_npz_files:
            # with h5py.File(filepath, "r") as input:
            #     input_data_group = input["data"]
            #     for episode, group in input_data_group.items():

            data = torch.load(filepath, pickle_module=pickle)
            actions = torch.cat(data["actions"],
                                dim=0).cpu().numpy()  # read into memory
            extended_actions = extension_actions(actions)

            extended_actions[:, -16:] = (
                (extended_actions[:, -16:] - joint_limits[:, 0]) /
                (joint_limits[:, 1] - joint_limits[:, 0])) * 2 - 1
            reverted_actions = extended_actions[::-1, :]
            all_actions = np.concatenate([extended_actions, reverted_actions],
                                         axis=0)

            # all_actions[:, [11]] = 0.0  # remove wrist actions

            demo_group = output_data_group.create_group(f"demo_{episode_idx}")
            demo_group.create_dataset("actions", data=all_actions[::2])
            num_demo += len(all_actions[::2])

            # demo_group = output_data_group.create_group(
            #     f"demo_{episode_idx+1}")
            # demo_group.create_dataset("actions", data=extended_actions[::-1])

            episode_idx += 1
    print(f"Processed {len(all_npz_files)} files, total demos: {num_demo}")

    print(f"Merged dataset saved to {args_cli.output_file}")


if __name__ == "__main__":
    merge_datasets()
