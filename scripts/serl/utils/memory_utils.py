from serl_launcher.data.memory_efficient_replay_buffer import (
    MemoryEfficientReplayBuffer, )

from agentlace.data.data_store import DataStoreBase
import zarr

from collections import defaultdict

import copy

import numpy as np
import os
# ['right_contact_obs', 'right_ee_pose', 'right_hand_joint_pos', 'right_manipulated_object_pose', 'right_object_in_tip', 'right_target_object_pose']
from typing import Optional

import gymnasium as gym
from typing import Union, Iterable
from threading import Lock
from typing import List, Optional, TypeVar
# import oxe_envlogger if it is installed
try:
    from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType
except ImportError:
    print("rlds logger is not installed, install it if required: "
          "https://github.com/rail-berkeley/oxe_envlogger ")
    RLDSLogger = TypeVar("RLDSLogger")

from serl_launcher.data.replay_buffer import ReplayBuffer


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        rlds_logger=None,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION  # to init the state for restart
            self._logger = rlds_logger

    # ensure thread safety
    def insert(self, data):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(data)

            # add data to the rlds logger
            if self._logger:
                if self.step_type in {
                        RLDSStepType.TERMINATION,
                        RLDSStepType.TRUNCATION,
                }:
                    self.step_type = RLDSStepType.RESTART
                elif not data["masks"]:  # 0 is done, 1 is not done
                    self.step_type = RLDSStepType.TERMINATION
                elif data["dones"]:
                    self.step_type = RLDSStepType.TRUNCATION
                else:
                    self.step_type = RLDSStepType.TRANSITION

                self._logger(
                    action=data["actions"],
                    obs=data[
                        "next_observations"],  # TODO: check if this is correct
                    reward=data["rewards"],
                    step_type=self.step_type,
                )

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class MemoryEfficientReplayBufferDataStore(MemoryEfficientReplayBuffer,
                                           DataStoreBase):

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        image_keys=[],
        rlds_logger=None,
    ):
        MemoryEfficientReplayBuffer.__init__(self,
                                             observation_space,
                                             action_space,
                                             capacity,
                                             pixel_keys=image_keys)
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION  # to init the state for restart
            self._logger = rlds_logger

    # ensure thread safety
    def insert(self, data):
        with self._lock:
            super(MemoryEfficientReplayBufferDataStore, self).insert(data)

            if self._logger:
                # handle restart when it was done before
                if self.step_type in {
                        RLDSStepType.TERMINATION,
                        RLDSStepType.TRUNCATION,
                }:
                    self.step_type = RLDSStepType.RESTART
                elif self.step_type == RLDSStepType.TRUNCATION:
                    self.step_type = RLDSStepType.RESTART
                elif not data["masks"]:  # 0 is done, 1 is not done
                    self.step_type = RLDSStepType.TERMINATION
                elif data["dones"]:
                    self.step_type = RLDSStepType.TRUNCATION
                else:
                    self.step_type = RLDSStepType.TRANSITION

                self._logger(
                    action=data["actions"],
                    obs=data[
                        "next_observations"],  # TODO: not obs, but next_obs
                    reward=data["rewards"],
                    step_type=self.step_type,
                )

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(MemoryEfficientReplayBufferDataStore,
                         self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO
