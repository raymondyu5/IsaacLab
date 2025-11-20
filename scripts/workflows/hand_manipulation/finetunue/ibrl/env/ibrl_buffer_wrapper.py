from scripts.workflows.hand_manipulation.finetunue.ibrl.env.replay_buffer import RelayBufferWrapper
from scripts.workflows.hand_manipulation.env.rl_env.rl_wrapper import RLDatawrapperEnv
import numpy as np
import sys

sys.path.append("submodule/ibrl")

import common_utils
from common_utils import ibrl_utils
from rl.q_agent import QAgent, QAgentConfig, update_dataclass
# from rl import replay
from common_utils import ibrl_utils as utils

from scripts.workflows.hand_manipulation.finetunue.ibrl.env.ibrl_agent import MainConfig

import copy
from scripts.workflows.hand_manipulation.finetunue.ibrl.env.ibrl_utils import construct_space
import torch


class IBRLBufferWrapper(RLDatawrapperEnv):

    def __init__(self,
                 env,
                 env_config,
                 args_cli,
                 use_relative_pose=False,
                 use_joint_pose=False,
                 eval_mode=False,
                 replay_mode=False,
                 collect_mode=False,
                 cfg=None,
                 use_visal_obs=False,
                 obs_keys=[]):
        super().__init__(
            env,
            env_config,
            args_cli,
            use_relative_pose=use_relative_pose,
            use_joint_pose=use_joint_pose,
            eval_mode=eval_mode,
            replay_mode=replay_mode,
            collect_mode=collect_mode,
        )
        self.cfg = cfg
        self.obs_keys = obs_keys
        self.use_visal_obs = use_visal_obs

    def _setup_replay(self):

        obs_space, action_space = construct_space(self)

        self.replay_buffer = RelayBufferWrapper(
            rl_buffer_size=self.cfg.replay_buffer_size *
            self.cfg.episode_length,
            bc_buffer_size=self.cfg.num_warm_up_episode *
            (self.cfg.episode_length),
            observation_space=obs_space,
            n_envs=self.env.num_envs,
            action_space=action_space)
        self.concatenate_obs = self.replay_buffer.concatenate_obs

    def process_obs(self, env_id=None, per_obs=[], return_numpy=False):

        if self.use_visal_obs:
            return per_obs

        processed_obs = []

        for key in self.obs_keys:

            if env_id is not None:

                processed_obs.append(per_obs[key][env_id])
            else:

                processed_obs.append(per_obs[key])

        if return_numpy:
            return torch.cat(processed_obs, dim=-1).cpu().numpy()

        return torch.cat(processed_obs, dim=-1)  #.cpu().numpy()

    def push_to_bc_buffer(self, obs, next_obs, actions, rewards, dones,
                          num_success):

        for env_id in range(num_success):

            for env_step in range(self.cfg.episode_length):
                per_obs = obs[env_step]

                next_per_obs = next_obs[env_step]

                last_obs = self.process_obs(env_id, per_obs, return_numpy=True)
                next_observation = self.process_obs(env_id,
                                                    next_per_obs,
                                                    return_numpy=True)

                self.replay_buffer.bc_replay_buffer.add(
                    last_obs, next_observation,
                    actions[env_step][env_id].cpu().numpy(),
                    rewards[env_step][env_id].cpu().numpy(),
                    dones[env_step][env_id].cpu().numpy(), [])

                if self.replay_buffer.bc_replay_buffer.pos == (
                        self.cfg.num_warm_up_episode * self.cfg.episode_length
                        - 1):

                    return True
        return False
