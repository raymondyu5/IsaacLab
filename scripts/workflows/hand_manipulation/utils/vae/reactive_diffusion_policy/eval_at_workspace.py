import numpy as np

import torch
import einops
import sys
import os
from sklearn.decomposition import PCA

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra
from scripts.workflows.hand_manipulation.utils.diffusion.hand_dataset import HandLowdimDataset
from model import VAE
from torch.utils.data import DataLoader
import tqdm
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
import matplotlib.pyplot as plt

checkpoint = os.path.join(
    "logs/data_0618/reactive_vae_state/horizon_4_nobs_1_nlatent_4/checkpoints",
    "latest.ckpt")

payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

cfg = payload['cfg']

cls = hydra.utils.get_class(cfg._target_)

workspace = cls(cfg, args_cli=None)
workspace: BaseWorkspace
workspace.load_payload(payload, exclude_keys=["normalizer"], include_keys=None)
policy = workspace.model

dataset = HandLowdimDataset(
    h5py_path="logs/data_0618/raw_right_data.hdf5",
    horizon=cfg.horizon,
    pad_before=cfg.n_obs_steps - 1,
    pad_after=cfg.n_obs_steps - 1,
    obs_key="hand_joints",
    state_key="state",
    action_key="action",
    seed=cfg.training.seed,
)

train_dataloader = DataLoader(dataset, **cfg.dataloader)

policy.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
latent_means = []
state_vq = []
with torch.no_grad():
    with tqdm.tqdm(train_dataloader,
                   desc="Collecting latent means",
                   leave=False,
                   mininterval=cfg.training.tqdm_interval_sec) as tepoch:
        for batch_idx, batch in enumerate(tepoch):
            batch = dict_apply(batch,
                               lambda x: x.to(device, non_blocking=True))

            result = policy.eval_step(batch)

            if 'latent_mean' in result:
                latent_mean = result['latent_mean']  # shape: [B, D, T]
                latent_means.append(latent_mean.cpu())
                state_vq.append(result["state_vq"].cpu())

            latent_dim = policy.n_latent_dims
            T_down = policy.downsampled_input_h  # usually 1 unless using conv encoder

            # Sample from standard normal
            B = batch["action"].shape[0]

            max_latent_value = policy.max_latent_value
            min_latent_value = policy.min_latent_value

            sample = torch.rand((B, latent_dim * T_down)).to(policy.device) * (
                max_latent_value - min_latent_value) + min_latent_value
            action = policy.get_action_from_latent(sample.to(torch.float32),
                                                   batch["obs"][:, 0])

            # reconstruct_action = einops.rearrange(result["dec_out"],    "N (T A) -> N T A",  A=16)
            # torch.nn.MSELoss()(dd, reconstruct_action.cpu())
            # torch.linalg.norm(dd - reconstruct_action.cpu(), dim=-1).mean()

            # dd = policy.normalizer["action"].normalize( batch["action"])  # unnormalize the action to get the original scale

# Stack collected VQ states
state_vq = torch.cat(state_vq, dim=0).numpy()  # shape: [N, D]
N, D = state_vq.shape

# Plot each dimension
ncols = 4
nrows = int(np.ceil(D / ncols))

plt.figure(figsize=(4 * ncols, 3 * nrows))

for i in range(D):
    plt.subplot(nrows, ncols, i + 1)
    plt.hist(state_vq[:, i], bins=100, density=True, alpha=0.75)
    plt.title(f"Dim {i}")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.grid(True)

plt.tight_layout()
plt.suptitle("Histograms of All state_vq Dimensions", y=1.02, fontsize=16)
plt.show()

# latent_means = torch.cat(latent_means, dim=0)  # [num_samples, T, D]

# latent_means = latent_means.reshape(
#     -1, latent_means.shape[-1])  # [num_samples * T, D]
# import matplotlib.pyplot as plt

# num_dims = latent_means.shape[1]
# ncols = 2
# nrows = 1
# fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3))

# for i in range(num_dims):
#     ax = axes[i]
#     ax.hist(latent_means[:, i].numpy(), bins=50, density=True, alpha=0.7)
#     ax.set_title(f'Latent Dim {i}')
#     ax.grid(True)

# fig.tight_layout()
# plt.show()
