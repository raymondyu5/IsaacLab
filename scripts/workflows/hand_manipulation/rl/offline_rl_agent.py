import sys

sys.path.append("submodule/d3rlpy")

import d3rlpy
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper

from scripts.workflows.utils.multi_datawrapper import list_zarr_files
from scripts.sb3.rl_algo_wrapper import rl_parser
import zarr
import numpy as np
from d3rlpy.dataset.compat import MDPDataset
if __name__ == "__main__":

    args_cli, hydra_args = rl_parser.parse_known_args()
    data_file = list_zarr_files(args_cli.load_path)
    target_obs_key = [
        'right_ee_pose', 'right_hand_joint_pos',
        'right_manipulated_object_pose', 'right_object_in_tip',
        'right_target_object_pose', 'right_contact_obs'
    ]
    observations = []
    actions = []
    rewards = []
    terminals = []

    # Open the HDF5 file in read mode
    # with h5py.File("logs/data_1007/teleop_data/grasp/grasp_data.hdf5",
    #                "r") as f:
    #     # List all groups

    #     data = f["data"]

    #     for demo_key in data.keys():
    #         demo_data = data[demo_key]
    #         per_obs = []
    #         import pdb
    #         pdb.set_trace()
    #         for obs_key in target_obs_key:
    #             per_obs.append(demo_data["obs"][obs_key][:])
    #         per_obs = np.concatenate(per_obs, axis=-1)
    #         observations.append(per_obs)
    #         actions.append(np.array(demo_data["actions"][:]))
    #         rewards.append(np.array(demo_data["rewards"][:]))
    #         termin = np.array(demo_data["dones"][:])
    #         termin[-1] = 1.0  # ensure last step is terminal
    #         terminals.append(termin)

    for file in data_file:
        data = zarr.open(file, mode="r")  # to fix zarr file permission issue
        obs_key = list(data["data"].keys())

        per_obs = []

        for obs_key in target_obs_key:
            per_obs.append(data["data"][obs_key][:])
        per_obs = np.concatenate(per_obs, axis=-1)
        observations.append(per_obs)
        actions.append(np.array(data["data"]["actions"][:]))
        rewards.append(np.array(data["data"]["rewards"][:]))
        termin = np.array(data["data"]["dones"][:])
        termin[-1] = 1.0  # ensure last step is terminal
        terminals.append(termin)

    observations = np.concatenate(observations, axis=0)
    actions = np.concatenate(actions, axis=0)
    rewards = np.concatenate(rewards, axis=0)
    terminals = np.concatenate(terminals, axis=0)

    max_action = np.max(actions, axis=0)
    min_action = np.min(actions, axis=0)
    actions = (actions - min_action) / (max_action - min_action) * 2 - 1

    dataset = MDPDataset(
        observations=np.array(observations),
        actions=np.array(actions),
        rewards=np.array(rewards) / 10,
        terminals=np.array(terminals),
    )
    device = "cuda:0"
    # import torch

    # dataset = MDPDataset(
    #     observations=torch.as_tensor(observations).to(device=device),
    #     actions=torch.as_tensor(np.array(actions)).to(device=device),
    #     rewards=torch.as_tensor(np.array(rewards) / 10).to(device=device),
    #     terminals=torch.as_tensor(np.array(terminals)).to(device=device),
    # )
    # agent = d3rlpy.load_learnable(
    #     "logs/d3rlpy_logs/TD3PlusBC_20251112141643/model_20000.d3")

    # pred_actions = agent.predict(observations)
    # import pdb
    # pdb.set_trace()

    agent = d3rlpy.algos.DDPGConfig(critic_learning_rate=1e-4).create(
        device=device)

    # train offline
    agent.fit(dataset, n_steps=1000000)
