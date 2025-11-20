import sys
import einops

sys.path.append("submodule/benchmark_VAE/src")

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace

import hydra
import os
import torch

import h5py
import numpy as np
import copy


def load_reactive_vae_model(vae_path):

    checkpoint = os.path.join(f"{vae_path}/checkpoints", "latest.ckpt")

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

    cfg = payload['cfg']

    cls = hydra.utils.get_class(cfg._target_)

    workspace = cls(cfg, args_cli=None)
    workspace: BaseWorkspace
    workspace.load_payload(payload,
                           exclude_keys=["normalizer"],
                           include_keys=None)
    reactive_vae_model = workspace.model

    return reactive_vae_model


def h5py2zarr(path):

    root = {'meta': {}, 'data': {}}
    action_dim = 0
    with h5py.File(path, "r") as f:

        demo_buffer = f['data']
        episode_ends = []
        episode_count = 0
        action_buffer = []
        state_buffer = []
        obs_buffer = []
        for key in demo_buffer.keys():

            if "obs" in demo_buffer[key].keys():
                import pdb
                pdb.set_trace()

                action = np.array(demo_buffer[key]['actions'])
                state = np.array(demo_buffer[key]['obs']["state"])

            else:
                action = demo_buffer[key]['actions'][1:, -16:]
                state = demo_buffer[key]['actions'][:-1, -16:]
            num_steps = action.shape[0]

            episode_ends.append(copy.deepcopy(num_steps + episode_count))
            episode_count += num_steps
            action_buffer.append(copy.deepcopy(action))

            state_buffer.append(copy.deepcopy(state))
            obs_buffer.append(copy.deepcopy(state))
        action_dim = action.shape[-1]

        root['meta']['episode_ends'] = np.array(episode_ends, dtype=np.int64)
        action_buffer = np.concatenate(action_buffer, axis=0)
        state_buffer = np.concatenate(state_buffer, axis=0)
        obs_buffer = np.concatenate(obs_buffer, axis=0)
        root['data']['action'] = action_buffer
        root['data']['state'] = state_buffer
        root['data']['hand_joints'] = obs_buffer

    return root, action_dim


model = load_reactive_vae_model(
    vae_path=
    "logs/data_0705/retarget_visionpro_data/reactive_vae/horizon_2_nobs_1_nlatent_4"
)

root, action_dim = h5py2zarr(
    path="logs/data_0705/retarget_visionpro_data/retarget_visionpro_data.hdf5")

state = root['data']['state']
hand_joints = root['data']['hand_joints']

device = "cuda:0"
with torch.no_grad():
    latent = model.get_latent_state(
        batch={
            'state':
            torch.as_tensor(state).float().unsqueeze(1).to(torch.float32).to(
                device).clip(-1, 1),
            'obs':
            torch.as_tensor(hand_joints).float().unsqueeze(1).to(
                torch.float32).to(device).clip(-1, 1),
            'action':
            torch.as_tensor(root['data']['action']).float().unsqueeze(1).to(
                torch.float32).to(device).clip(-1, 1),
        })

max_latent_value = latent.max(0).values
min_latent_value = latent.min(0).values
hand_noise = torch.randn_like(latent).clip(-1, 1)
hand_noise = (hand_noise + 1) / 2 * (
    (max_latent_value - min_latent_value)) + min_latent_value
print(latent.max(0).values, latent.min(0).values)

with torch.no_grad():

    action = model.get_action_from_latent(
        hand_noise.to(torch.float32),
        torch.zeros((hand_noise.shape[0], 1, 16),
                    device=device).to(torch.float32))

    # reconstruct_action = einops.rearrange(action,
    #                                       "N (T A) -> N T A",
    #                                       A=self.num_hand_joints)
    # reconstructed_hand_actions = self.reactive_vae_model.normalizer[
    #     "action"].unnormalize(reconstruct_action).to(self.device).clip(-1, 1)
