from scripts.workflows.hand_manipulation.env.bc_env.zarr_data_wrapper import ZarrDatawrapper

from scripts.workflows.open_policy.task.BCPPOBuffer import OnlineBCBuffer

import numpy as np


class BCBufferLoader(ZarrDatawrapper):

    def __init__(
        self,
        env,
        args_cli,
        env_config,
        filter_keys=[],
    ):
        super().__init__(
            args_cli,
            env_config,
            filter_keys,
            zarr_cfg=None,
        )
        self.env = env

        self.rollout_buffer = OnlineBCBuffer(
            1,
            1,
            env.observation_space["policy"],
            env.action_space,
            "cuda:0",
            1,
        )
        self.num_demos = args_cli.num_demos
        self.load_demo_data()

    def load_demo_data(self):

        for i in range(self.num_demos):

            data_dict = self.load_data()
            lowdim_obs = []
            for key in self.env.observation_space["policy"].spaces.keys():
                lowdim_obs.append(data_dict[key])
            lowdim_obs = np.concatenate(lowdim_obs, axis=-1)
            actions = data_dict["actions"]
            rewards = data_dict["rewards"]
            num_datas = lowdim_obs.shape[0]
            dones = np.zeros((num_datas, 1)).astype(np.bool_)
            terminates = np.zeros((num_datas, 1))
            dones[-1] = True
            terminates[-1] = 1.0
            for j in range(num_datas - 1):

                self.rollout_buffer.add(lowdim_obs[j].reshape(1, -1),
                                        lowdim_obs[j + 1].reshape(1, -1),
                                        actions[j].reshape(1, -1),
                                        rewards[j].reshape(1, -1),
                                        terminates[j + 1].reshape(1, -1),
                                        dones[j + 1].reshape(1, -1))
