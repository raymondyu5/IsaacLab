import torch
import h5py

import numpy as np


def h5py_group_to_dict(h5_group):
    result = {}

    for key in h5_group:
        item = h5_group[key]
        if isinstance(item, h5py.Group):
            result[key] = h5py_group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]  # Convert dataset to NumPy array or scalar
    return result


def reset_buffer(object):
    setattr(object, "obs_buffer", [])
    setattr(object, "action_buffer", [])
    setattr(object, "rewards_buffer", [])
    setattr(object, "next_obs_buffer", [])
    setattr(object, "does_buffer", [])


from typing import Dict, Callable, List


def dict_apply(
        x: Dict[str, torch.Tensor],
        func: Callable[[torch.Tensor],
                       torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            if not isinstance(value, torch.Tensor):
                result[key] = value
            else:
                result[key] = func(value)
    return result


def update_buffer(object,
                  next_obs,
                  last_obs,
                  actions,
                  rewards,
                  terminated,
                  time_outs,
                  convert_to_cpu=False):

    if convert_to_cpu:

        object.obs_buffer.append(dict_apply(last_obs, lambda x: x.cpu()))

        object.action_buffer.append(actions.cpu())
        object.rewards_buffer.append(rewards.cpu())

        if next_obs is not None:

            object.next_obs_buffer.append(next_obs.cpu())
        # object.next_obs_buffer.append(next_obs)

        object.does_buffer.append((terminated | time_outs).cpu())
    else:
        object.obs_buffer.append(last_obs)
        object.action_buffer.append(actions)
        object.rewards_buffer.append(rewards)
        if next_obs is not None:
            object.next_obs_buffer.append(next_obs)

        object.does_buffer.append(terminated | time_outs)


from collections import defaultdict


def filter_out_obs_data(
    index,
    per_obs,
):

    per_obs_dict = defaultdict(list)

    for obs_key in list(per_obs["policy"].keys()):
        if obs_key in ["id2lables"]:
            continue

        if len(per_obs["policy"][obs_key]) == 0:
            continue

        if isinstance(per_obs["policy"][obs_key], torch.Tensor):

            per_obs_dict[obs_key] = per_obs["policy"][obs_key][index]
        elif isinstance(per_obs["policy"][obs_key], list):

            if len(per_obs["policy"][obs_key]) == 1:
                per_obs_dict[obs_key] = per_obs["policy"][obs_key]

            else:

                per_obs_dict[obs_key] = [
                    per_obs["policy"][obs_key][i] for i in index
                ]
    return per_obs_dict


def filter_out_data(self, index, save_data=True):

    obs_buffer = []
    action_buffer = []
    rewards_buffer = []
    does_buffer = []
    next_obs_buffer = []
    for i in range(len(self.obs_buffer)):
        per_obs = self.obs_buffer[i]
        # per_obs_dict = defaultdict(list)
        # for obs_key in list(per_obs["policy"].keys()):
        #     if obs_key in ["id2lables"]:
        #         continue

        #     if len(per_obs["policy"][obs_key]) == 0:
        #         continue

        #     if isinstance(per_obs["policy"][obs_key], torch.Tensor):

        #         per_obs_dict[obs_key] = per_obs["policy"][obs_key][index]
        #     elif isinstance(per_obs["policy"][obs_key], list):

        #         if len(per_obs["policy"][obs_key]) == 1:
        #             per_obs_dict[obs_key] = per_obs["policy"][obs_key]

        #         else:

        #             per_obs_dict[obs_key] = [
        #                 per_obs["policy"][obs_key][i] for i in index
        #             ]
        per_obs_dict = filter_out_obs_data(
            index,
            per_obs,
        )

        if len(self.next_obs_buffer) > 0:
            next_per_obs = self.next_obs_buffer[i]
            next_per_obs_dict = filter_out_obs_data(
                index,
                next_per_obs,
            )
            next_obs_buffer.append(next_per_obs_dict)

        obs_buffer.append(per_obs_dict)

        action_buffer.append(self.action_buffer[i][index])
        rewards_buffer.append(self.rewards_buffer[i][index])
        does_buffer.append(self.does_buffer[i][index])

    if save_data:

        self.collector_interface.add_demonstraions_to_buffer(
            obs_buffer,
            action_buffer,
            rewards_buffer,
            does_buffer,
        )
    return obs_buffer, next_obs_buffer, action_buffer, rewards_buffer, does_buffer
