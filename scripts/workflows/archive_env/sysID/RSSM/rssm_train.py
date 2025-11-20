# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to an environment with random action agent."""
"""Launch Isaac Sim Simulator first."""

import argparse
import sys

sys.path.append(".")

from tools.visualization_utils import *
from tqdm import trange
import cv2
import matplotlib.pyplot as plt
import numpy as np
# import open3d as o3d
import sys
from torch.distributions.kl import kl_divergence

from scripts.workflows.sysID.bayes_sim.utils.data_utils import collect_trajectories
from scripts.workflows.sysID.bayes_sim.utils.plot import plot_trajectories
from scripts.workflows.sysID.RSSM.utils.rssm_dataset import RssmDataset
from scripts.workflows.sysID.RSSM.model.rssm_model import *
from scripts.workflows.sysID.bayes_sim.models import mdnn
# add argparse arguments
parser = argparse.ArgumentParser(
    description="Random agent for Isaac Lab environments.")

parser.add_argument("--logdir",
                    type=str,
                    default=None,
                    help="Name of the task.")
parser.add_argument("--data_dir",
                    type=str,
                    default=None,
                    help="Name of the task.")

args_cli = parser.parse_args()
from scripts.workflows.sysID.RSSM.utils.utils import *
import matplotlib.pyplot as plt


def save_images(batch_size,
                original_tensor,
                reconstructed_tensor,
                prms,
                save_path,
                bayes_result=None):
    batch_size, time_horizon, height, width, channels = original_tensor.shape
    reconstructed_images = reconstructed_tensor.permute(0, 2, 1, 3, 4).reshape(
        batch_size, height, width * time_horizon, channels) + 0.5
    fig, axs = plt.subplots(batch_size * 2, 1, figsize=(batch_size * 5, 5))
    raw_images = original_tensor.permute(0, 2, 1, 3, 4).reshape(
        batch_size, height, width * time_horizon, channels) + 0.5
    reconstructed_images = reconstructed_tensor.permute(0, 2, 1, 3, 4).reshape(
        batch_size, height, width * time_horizon, channels) + 0.5

    fig, axs = plt.subplots(batch_size, 2)
    num_id = int(batch_size / 2)
    for i in range(batch_size):
        # Get the image for the i-th batch
        img = raw_images[i].cpu().numpy(
        )  # Convert to numpy array for plotting

        axs[(i % num_id) * 2][i // num_id].imshow(
            (img * 255).astype(np.uint16))

        img = reconstructed_images[i].cpu().numpy(
        )  # Convert to numpy array for plotting
        axs[(i % num_id) * 2 + 1][i // num_id].imshow(
            np.clip((img * 255).astype(np.uint16), 0, 255))

        # img = reconstructed_images[i].cpu().numpy(
        # )  # Convert to numpy array for plotting
        # axs[i].imshow(img)
        axs[(i % num_id) * 2][i // num_id].axis('off')  # Turn off axis labels
        axs[(i % num_id) * 2 + 1][i // num_id].axis('off')

        gt_prm = np.round(prms[1][i].cpu().numpy()[0], 2).copy()
        pd_prm = np.round(prms[0][i].detach().cpu().numpy(), 2)

        axs[(i % num_id) * 2][i // num_id].set_title("gt prms:" + str(gt_prm))

        if bayes_result is None:
            axs[(i % num_id) * 2 +
                1][i // num_id].set_title("predicted prms:" + str(pd_prm))
        else:

            m, S = bayes_result[i].calc_mean_and_cov()

            m = np.round(m[0][0], 2)
            S = np.round(S[0][0], 2)
            axs[(i % num_id) * 2 +
                1][i // num_id].set_title("predicted prms mean/std:" + str(m) +
                                          " ," + str(S))

    plt.tight_layout()

    # plt.axis("off")
    # plt.show()
    plt.savefig(save_path)
    plt.cla()


def visualize_images(original_tensor,
                     reconstructed_tensor,
                     prms,
                     save_path,
                     bayes_result=None):
    """
    Visualize specific images from the original and reconstructed tensors.
    
    Args:
        original_tensor (torch.Tensor): The original images tensor of shape (batch_size, time_horizon, channels, height, width).
        reconstructed_tensor (torch.Tensor): The reconstructed images tensor of the same shape as original_tensor.
        indices (list of int): The indices of images to visualize.
    """

    # Create a plot for each index in the original and reconstructed sets

    original_tensor = original_tensor.permute(1, 0, 3, 4, 2)
    reconstructed_tensor = reconstructed_tensor.permute(1, 0, 3, 4, 2)
    batch_size, time_horizon, height, width, channels = original_tensor.shape

    save_images(batch_size, original_tensor, reconstructed_tensor, prms,
                save_path, bayes_result)

    cat_images = np.clip(
        ((torch.cat([original_tensor, reconstructed_tensor], dim=3).permute(
            1, 0, 2, 3, 4).reshape(time_horizon, batch_size * height,
                                   width * 2, channels) + 0.5) *
         255).cpu().numpy().astype(np.int32), 0, 255)
    batch_images_to_video(cat_images,
                          save_path.replace("png", "mp4"),
                          1,
                          image_save_folder=None)


def train(dataset,
          rssm,
          device,
          optimizer,
          beta=1.0,
          grads=False,
          bayes_model=None):
    free_nats = torch.ones(1, device=device) * 3.0
    sample_rbg, sample_prms, sample_u = dataset.sample(200)
    #preprocess_img(sample_rbg, depth=5)

    e_t = bottle(rssm.encoder, sample_rbg)
    states, priors, posteriors, posterior_samples = [], [], [], []
    h_t, s_t = rssm.get_init_state(e_t[0])

    for i, a_t in enumerate(torch.unbind(sample_u, dim=0)):

        h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
        states.append(h_t)
        priors.append(rssm.state_prior(h_t))
        posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
        posterior_samples.append(Normal(*posteriors[-1]).rsample())
        s_t = posterior_samples[-1]
    prior_dist = Normal(*map(torch.stack, zip(*priors)))
    posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
    states, posterior_samples = map(torch.stack, (states, posterior_samples))

    rew_loss = F.mse_loss(rssm.pred_reward(h_t, s_t), sample_prms[0]) * 500

    rec_loss = F.mse_loss(bottle(rssm.decoder, states, posterior_samples),
                          sample_rbg[1:],
                          reduction='none').sum((2, 3, 4)).mean()
    kld_loss = torch.max(
        kl_divergence(posterior_dist, prior_dist).sum(-1), free_nats).mean()
    optimizer.zero_grad()
    nn.utils.clip_grad_norm_(rssm.parameters(), 1000., norm_type=2)

    (beta * kld_loss + rec_loss + rew_loss).backward()
    bayes_metric = None
    if bayes_model is not None:
        bayes_metric = bayes_model.run_training(x_data=torch.cat(
            [h_t, s_t], dim=1).detach(),
                                                y_data=sample_prms.to(device),
                                                n_updates=100,
                                                batch_size=64)
    optimizer.step()
    metrics = {
        'losses': {
            'kl': kld_loss.item(),
            'reconstruction': rec_loss.item(),
            'reward_pred': rew_loss.item()
        },
    }
    if grads:
        metrics['grad_norms'] = {
            k: 0 if v.grad is None else v.grad.norm().item()
            for k, v in rssm.named_parameters()
        }

    return metrics, bayes_metric


def eval(dataset, rssm, save_path, bayes_model):
    with torch.no_grad():
        eval_rbg, eval_prms, eval_u = dataset.sample(6, for_eval=True)

        #(eval_rbg, depth=5)

        e_t = bottle(rssm.encoder, eval_rbg)
        states, priors, posteriors, posterior_samples = [], [], [], []
        h_t, s_t = rssm.get_init_state(e_t[0])

        for i, a_t in enumerate(torch.unbind(eval_u, dim=0)):

            h_t = rssm.deterministic_state_fwd(h_t, s_t, a_t)
            states.append(h_t)
            priors.append(rssm.state_prior(h_t))
            posteriors.append(rssm.state_posterior(h_t, e_t[i + 1]))
            posterior_samples.append(Normal(*posteriors[-1]).rsample())
            s_t = posterior_samples[-1]
        prior_dist = Normal(*map(torch.stack, zip(*priors)))
        posterior_dist = Normal(*map(torch.stack, zip(*posteriors)))
        states, posterior_samples = map(torch.stack,
                                        (states, posterior_samples))
        deco_image = bottle(rssm.decoder, states, posterior_samples)
        rec_loss = F.mse_loss(bottle(rssm.decoder, states, posterior_samples),
                              eval_rbg[1:],
                              reduction='none').sum((2, 3, 4)).mean()
        pred_prms = rssm.pred_reward(h_t, s_t)

        bayes_result = None
        if bayes_model is not None:
            bayes_result = bayes_model.predict_MoGs(
                torch.cat([h_t, s_t], dim=1).detach())

            # ms = [x.m for x in result[0].xs]
            # m = np.dot(result[0].a, np.array(ms)[np.newaxis, :])
            # Ss = [x.C[0] for x in result[0].xs]
            # S = np.dot(result[0].a, np.array(Ss)[np.newaxis, :])

        visualize_images(eval_rbg[1:].detach(), deco_image.detach(),
                         [pred_prms, eval_prms], save_path, bayes_result)


def main():
    device = "cuda:0"
    start_index = 5
    end_index = 13
    interval = 1
    dataset = RssmDataset(args_cli.data_dir,
                          device=device,
                          start_index=start_index,
                          end_index=end_index,
                          interval=interval,
                          resize=(64, 64))
    bayes_model = mdnn.MDNN(input_dim=230,
                            output_dim=1,
                            output_lows=np.zeros(1),
                            output_highs=np.ones(1),
                            n_gaussians=2,
                            full_covariance=False,
                            hidden_layers=[128, 64, 32, 16],
                            activation=torch.nn.Tanh,
                            lr=1.e-4,
                            device=device)
    rssm_model = RecurrentStateSpaceModel(1).to(device)
    optimizer = torch.optim.Adam(rssm_model.parameters(), lr=1e-3, eps=1e-4)
    res_dir = args_cli.logdir
    os.makedirs(f'{res_dir}/checkpoints', exist_ok=True)
    os.makedirs(f'{res_dir}/eval', exist_ok=True)
    summary = TensorBoardMetrics(f'{res_dir}/checkpoints')

    for i in trange(150, desc='Epoch', leave=False):
        print('\nEPOCH: %d' % i)
        metrics = {}
        for _ in trange(30, desc='Iter ', leave=False):

            train_metrics, bayes_metrix = train(
                dataset,
                rssm_model,
                device,
                optimizer,
                bayes_model=bayes_model if i > 10 else None)
            for k, v in flatten_dict(train_metrics).items():
                if k not in metrics.keys():
                    metrics[k] = []
                metrics[k].append(v)
                metrics[f'{k}_mean'] = np.array(v).mean()
        summary.update(metrics)

        if bayes_metrix is not None:
            print('============')
            print(bayes_metrix)
        print(train_metrics)
        eval(dataset,
             rssm_model,
             save_path=f'{res_dir}/eval/eval_{i}.png',
             bayes_model=bayes_model if i > 10 else None)

        if (i + 1) % 25 == 0:
            torch.save(rssm_model.state_dict(),
                       f'{res_dir}/checkpoints/rssm_ckpt_{i+1}.pth')
            torch.save(bayes_model.state_dict(),
                       f'{res_dir}/checkpoints/bayes_ckpt_{i+1}.pth')

    print('DONE')
    exit()


if __name__ == "__main__":
    # run the main function
    main()
