from scripts.workflows.utils.multi_datawrapper import list_zarr_files
import zarr
import numpy as np
import torch
import sys

sys.path.append("submodule/d3rlpy")
import d3rlpy
from scripts.workflows.utils.multi_datawrapper import MultiDatawrapper

from scripts.workflows.utils.multi_datawrapper import list_zarr_files
from scripts.sb3.rl_algo_wrapper import rl_parser
import zarr
import numpy as np
from d3rlpy.dataset.compat import MDPDataset
from scripts.workflows.hand_manipulation.env.rl_env.d3rlpy_online_env import D3RLpyOnlineEnv

from d3rlpy.dataset.replay_buffer import ReplayBuffer, create_fifo_replay_buffer
from d3rlpy.dataset.buffers import BufferProtocol

from scripts.workflows.hand_manipulation.env.rl_env.d3rlpy_scratch_wrapper import D3RLpyScratchWrapper


class Offline2OnlineEnvWrapper:

    def __init__(self, env, agent, args_cli, save_config, target_obs_key=[]):
        self.env = env
        self.args_cli = args_cli
        self.env_cfg = save_config
        self.target_obs_key = target_obs_key
        self.agent = agent

        if self.args_cli.load_path is not None:
            self.init_data()
            # self.online_env = D3RLpyOnlineEnv(
            #     env,
            #     args_cli,
            #     save_config,
            #     target_obs_key,
            #     self.max_action,
            #     self.min_action,
            # )

        self.online_env = D3RLpyScratchWrapper(env, args_cli, save_config,
                                               target_obs_key)

    def init_data(self):

        data_file = list_zarr_files(self.args_cli.load_path)

        observations = []
        actions = []
        rewards = []
        terminals = []

        for file in data_file:
            data = zarr.open(file,
                             mode="r")  # to fix zarr file permission issue
            obs_key = list(data["data"].keys())

            per_obs = []

            for obs_key in self.target_obs_key:
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

        max_action = np.max(actions, axis=0)
        min_action = np.min(actions, axis=0)
        actions = (actions - min_action) / (max_action - min_action) * 2 - 1

        # actions = agent.predict(observations)
        # actions = (actions + 1) / 2 * (max_action - min_action) + min_action
        self.max_action = torch.as_tensor(max_action).to(
            device=self.env.unwrapped.device)
        self.min_action = torch.as_tensor(min_action).to(
            device=self.env.unwrapped.device)

        self.offline_dataset = MDPDataset(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards).reshape(-1) / 10,
            terminals=np.array(terminals).reshape(-1),
        )

    def fit_offline(self):

        # train offline
        self.agent.fit(self.offline_dataset, n_steps=5)

    def fit_online(self):

        # replay_buffer = ReplayBuffer(buffer=BufferProtocol,
        #                              env=self.online_env)
        replay_buffer = create_fifo_replay_buffer(
            env=self.online_env,
            limit=1000000,
            cache_size=200 * self.env.unwrapped.num_envs,
        )

        self.agent.fit_online(
            self.online_env,
            buffer=replay_buffer,
            n_steps=1000000,
            random_steps=300,
        )
