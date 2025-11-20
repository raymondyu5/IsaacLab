import torch
import os

import math
from matplotlib import pyplot as plt
import numpy as np


def viz_object_success_rate(self, title="Success Rate per Object"):
    image_path = f"{self.save_result_path}/eval_images"
    os.makedirs(image_path, exist_ok=True)

    success_rate = torch.stack(self.rigid_object_success_rate,
                               dim=0).reshape(-1,
                                              self.env.num_envs).cpu().numpy()
    object_rewards = torch.stack(self.rigid_object_reward, dim=0).reshape(
        -1, self.env.num_envs).cpu().numpy()
    target_dev = torch.stack(self.rigid_object_dev,
                             dim=0).reshape(-1,
                                            self.env.num_envs).cpu().numpy()

    env_target_ids = self.env_target_ids.cpu().numpy()

    num_objects = len(self.rigid_object_list)

    # Square layout
    ncols = math.ceil(math.sqrt(num_objects))
    nrows = math.ceil(num_objects / ncols)

    # visualize object success rate

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i in range(num_objects):
        ax = axes[i]
        mask = env_target_ids == i

        obj_success = success_rate[:, mask].sum(
            axis=1) / success_rate[:, mask].shape[1]

        ax.plot(obj_success, color='orange')
        ax.set_title(self.rigid_object_list[i])
        ax.set_ylim(0, 1.1)
        ax.set_xlim(0, success_rate.shape[0] + 1)

        ax.set_xlabel("Env idx")
        ax.set_ylabel("Success")
        ax.grid(True)
        self.object_success_rate[self.rigid_object_list[i]] = obj_success

    fig.suptitle(title, fontsize=16)
    plt.tight_layout()

    plt.savefig(f"{image_path}/object_success.png")  # Save to file

    plt.cla()
    plt.close()
    # plt.show()

    # visualize object reward

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i in range(num_objects):
        ax = axes[i]
        mask = env_target_ids == i

        obj_reward = object_rewards[:, mask].sum(axis=1) / len(mask)

        ax.plot(obj_reward, color='orange')
        ax.set_title(self.rigid_object_list[i])

        ax.set_xlim(0, object_rewards.shape[0] + 1)

        ax.set_xlabel("Env idx")
        ax.set_ylabel("Success")
        ax.grid(True)
        self.object_rewards[self.rigid_object_list[i]] = obj_reward

    fig.suptitle("reward per Object", fontsize=16)
    plt.tight_layout()

    plt.savefig(f"{image_path}/object_reward.png")  # Save to file

    plt.close()
    plt.cla()

    ### visualize object dev

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for i in range(num_objects):
        ax = axes[i]
        mask = env_target_ids == i

        obj_dev = target_dev[:, mask].mean(axis=1)

        ax.plot(obj_dev, color='orange')
        ax.set_title(self.rigid_object_list[i])

        ax.set_xlim(0, target_dev.shape[0] + 1)

        ax.set_xlabel("Env idx")
        ax.set_ylabel("Success")
        ax.grid(True)
        self.object_dev[self.rigid_object_list[i]] = obj_dev

    fig.suptitle("dev per Object", fontsize=16)
    plt.tight_layout()

    plt.savefig(f"{image_path}/object_dev.png")  # Save to file

    plt.close()
    plt.cla()

    np.savez(
        f"{image_path}/object_success",
        **self.object_success_rate,
    )
    np.savez(f"{image_path}/object_rollout", **self.object_rewards)
    np.savez(f"{image_path}/object_dev", **self.object_dev)


def viz_result(self, ):

    image_path = f"{self.save_result_path}/eval_images"
    os.makedirs(image_path, exist_ok=True)

    plt.figure()

    plt.plot(self.eval_success, label="Eval Success")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Success Rate")
    plt.title("Evaluation Success over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{image_path}/eval_success_plot.png")  # Save to file
    plt.close()

    # Plot rollout reward
    plt.figure()
    plt.plot(self.rollout_reward, label="Rollout Reward", color='orange')
    plt.xlabel("Evaluation Step")
    plt.ylabel("Reward")
    plt.title("Rollout Reward over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{image_path}/rollout_reward_plot.png")  # Save to file
    plt.close()

    # Plot rollout reward
    plt.figure()
    plt.plot(self.eval_dev, label="Target dev", color='orange')
    plt.xlabel("Evaluation Step")
    plt.ylabel("Reward")
    plt.title("Rollout Reward over Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{image_path}/rollout_dev_plot.png")  # Save to file
    plt.close()

    np.savez(
        f"{image_path}/hand_success",
        np.array(self.eval_success),
    )
    np.savez(
        f"{image_path}/hand_rollout",
        np.array(self.rollout_reward),
    )
    np.savez(
        f"{image_path}/hand_dev",
        np.array(self.eval_dev),
    )
