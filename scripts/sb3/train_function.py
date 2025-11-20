from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

import torch

try:
    from scripts.rsrl.agent.td3bc_policy import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy
    from scripts.rsrl.utils.robot_rl_utils import *
except:
    from isaac_scripts.rsrl.agent.td3bc_policy import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy
    from isaac_scripts.rsrl.utils.robot_rl_utils import *
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
import time


def infer_diffusion(self,
                    noise,
                    obs,
                    cartesian_actions,
                    use_latent_noise_only=False):

    with th.no_grad():

        # get next action
        with torch.no_grad():

            indices = torch.randperm(10240)[:1024].to("cuda")  # shape: [256]
            diffusion_dict = {
                "agent_pos":
                obs["state"][:, None][..., :23],
                "seg_pc":
                obs["seg_pc"][:, indices].unsqueeze(1).permute(0, 1, 3, 2),
            }

            delta_actions = self.latent_model.policy.predict_action(
                diffusion_dict, noise.unsqueeze(1))["action_pred"][:, 0]
    target_pos, target_rot = apply_delta_pose(cartesian_actions[..., :3],
                                              cartesian_actions[..., 3:7],
                                              delta_actions[..., :6])
    target_finger_pose = cartesian_actions[..., 7:] + delta_actions[..., 6:]

    actions = th.cat([target_pos, target_rot, target_finger_pose], dim=-1)
    if not use_latent_noise_only:
        return actions
    else:
        return infer_whole_actions(self, actions,
                                   torch.zeros((len(actions), 22)).cuda())


def train_latent_noise_only(self,
                            gradient_steps: int,
                            batch_size: int = 100,
                            callback=None) -> None:
    # Switch to train mode (this affects batch norm / dropout)

    self.policy.set_training_mode(True)

    # Update learning rate according to lr schedule
    self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

    actor_losses, critic_losses = [], []
    for _ in range(gradient_steps):
        self._n_updates += 1
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(
            batch_size,
            env=self._vec_normalize_env)  # type: ignore[union-attr]
        # For n-step replay, discount factor is gamma**n_steps (when no early termination)
        discounts = self.gamma

        with th.no_grad():

            noise = replay_data.actions.clone().data.normal_(
                0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip,
                                self.target_noise_clip)
            next_actions_noise = (
                self.actor_target(replay_data.next_observations) +
                noise).clamp(-1, 1)

            # get next action
            next_actions = infer_diffusion(self,
                                           next_actions_noise,
                                           replay_data.next_observations,
                                           replay_data.cartesian_actions,
                                           use_latent_noise_only=True)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(
                replay_data.next_observations, next_actions),
                                   dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (
                1 - replay_data.dones) * discounts * next_q_values

        # Get current Q-values estimates for each critic network
        current_actions = infer_diffusion(
            self,
            replay_data.actions,
            replay_data.observations,
            replay_data.cartesian_actions,
        )

        current_q_values = self.critic(replay_data.observations,
                                       current_actions)

        # Compute critic loss
        critic_loss = sum(
            F.mse_loss(current_q, target_q_values)
            for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            current_actions = infer_diffusion(
                self,
                self.actor(replay_data.observations),
                replay_data.observations,
                replay_data.cartesian_actions,
            )

            actor_loss = -self.critic.q1_forward(replay_data.observations,
                                                 current_actions).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(),
                          self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(),
                          self.actor_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.critic_batch_norm_stats,
                          self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats,
                          self.actor_batch_norm_stats_target, 1.0)

    self.logger.record("train/n_updates",
                       self._n_updates,
                       exclude="tensorboard")

    if len(actor_losses) > 0:
        self.logger.record("train/actor_loss", np.mean(actor_losses))

    self.logger.record("train/critic_loss", np.mean(critic_losses))
    self.dump_logs()
    callback.on_rollout_end()


def train_only(self,
               gradient_steps: int,
               batch_size: int = 100,
               callback=None) -> None:
    # Switch to train mode (this affects batch norm / dropout)

    self.policy.set_training_mode(True)

    # Update learning rate according to lr schedule
    self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

    actor_losses, critic_losses, bc_losses, q_values = [], [], [], []
    for _ in range(gradient_steps):
        start_time = time.time()
        self._n_updates += 1
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(
            batch_size,
            env=self._vec_normalize_env)  # type: ignore[union-attr]
        # For n-step replay, discount factor is gamma**n_steps (when no early termination)
        discounts = self.gamma

        with th.no_grad():
            # Select action according to policy and add clipped noise
            noise = replay_data.actions.clone().data.normal_(
                0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip,
                                self.target_noise_clip)

            if self.policy.q_type == "res":

                next_actions = (
                    self.actor_target(replay_data.next_observations) +
                    noise).clamp(-1, 1)
            elif self.policy.q_type == "sum":

                delta_actions = (
                    self.actor_target(replay_data.next_observations) +
                    noise).clamp(-1, 1)
                # append a constant 1 to the action
                base_action = replay_data.base_actions
                residual_action = (delta_actions + 1) / 2 * (
                    self.residual_action_range[:, 1] -
                    self.residual_action_range[:, 0]
                ) + self.residual_action_range[:, 0]
                target_pos, target_rot = apply_delta_pose(
                    base_action[..., :3], base_action[..., 3:7],
                    residual_action[..., :6])
                target_finger_pose = base_action[...,
                                                 7:] + residual_action[..., 6:]

                next_actions = th.cat(
                    [target_pos, target_rot, target_finger_pose], dim=-1)
            elif self.policy.q_type == "concat":
                delta_actions = (
                    self.actor_target(replay_data.next_observations) +
                    noise).clamp(-1, 1)
                next_actions = th.cat(
                    [replay_data.base_actions, delta_actions], dim=-1)

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(
                replay_data.next_observations, next_actions),
                                   dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (
                1 - replay_data.dones) * discounts * next_q_values

        # Get current Q-values estimates for each critic network

        if self.policy.q_type == "res":
            current_q_values = self.critic(replay_data.observations,
                                           replay_data.actions)
        elif self.policy.q_type == "sum":

            current_q_values = self.critic(replay_data.observations,
                                           replay_data.robot_actions)
        elif self.policy.q_type == "concat":
            current_q_values = self.critic(
                replay_data.observations,
                th.cat([replay_data.base_actions, replay_data.actions],
                       dim=-1))

        # Compute critic loss
        critic_loss = sum(
            F.mse_loss(current_q, target_q_values)
            for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)
        critic_losses.append(critic_loss.item())

        q_values.append(
            torch.cat(current_q_values, dim=1).detach().mean().item())

        # Optimize the critics
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:

            # # Compute actor loss
            # actor_loss = -self.critic.q1_forward(
            #     replay_data.observations,
            #     self.actor(replay_data.observations)).mean()

            policy_actions = self.actor(replay_data.observations)

            # Q term (same as TD3)

            if self.policy.q_type == "res":
                policy_actions = policy_actions
                # Behavior cloning term
                bc_loss = F.mse_loss(policy_actions,
                                     replay_data.actions,
                                     reduction="mean")

                q_loss = -self.critic.q1_forward(replay_data.observations,
                                                 policy_actions).mean()
            elif self.policy.q_type == "sum":
                # append a constant 1 to the action
                base_action = replay_data.base_actions
                residual_action = (policy_actions + 1) / 2 * (
                    self.residual_action_range[:, 1] -
                    self.residual_action_range[:, 0]
                ) + self.residual_action_range[:, 0]
                target_pos, target_rot = apply_delta_pose(
                    base_action[..., :3], base_action[..., 3:7],
                    residual_action[..., :6])
                target_finger_pose = base_action[...,
                                                 7:] + residual_action[..., 6:]
                combined_policy_actions = th.cat(
                    [target_pos, target_rot, target_finger_pose], dim=-1)

                q_loss = -self.critic.q1_forward(
                    replay_data.observations, combined_policy_actions).mean()
                # Behavior cloning term
                bc_loss = F.mse_loss(policy_actions,
                                     replay_data.actions,
                                     reduction="mean")
            elif self.policy.q_type == "concat":
                combined_policy_actions = th.cat(
                    [replay_data.base_actions, policy_actions], dim=-1)
                q_loss = -self.critic.q1_forward(
                    replay_data.observations, combined_policy_actions).mean()
                # Behavior cloning term
                bc_loss = F.mse_loss(policy_actions,
                                     replay_data.actions,
                                     reduction="mean")
            bc_losses.append(bc_loss.item())

            bc_loss = self.lambda_bc * bc_loss

            if th.isnan(q_loss).any():
                import pdb

                pdb.set_trace()
            th.isnan(replay_data.actions).any()
            # for name, p in self.actor.named_parameters():
            #     if p.grad is not None and not torch.isfinite(p.grad).all():
            #         print("NaN grad in", name)

            actor_loss = q_loss + bc_loss
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(),
                          self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(),
                          self.actor_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.critic_batch_norm_stats,
                          self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats,
                          self.actor_batch_norm_stats_target, 1.0)

    self.logger.record("train/n_updates",
                       self._n_updates,
                       exclude="tensorboard")
    if len(actor_losses) > 0:
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
        self.logger.record("train/q_values", np.mean(q_values))
    self.logger.record("train/critic_loss", np.mean(critic_losses))
    import wandb

    self.dump_logs()
    self.iteractions += 1

    wandb.log({
        "train/actor_loss": np.mean(actor_losses),
        "train/bc_loss": np.mean(bc_losses),
        "train/q_values": np.mean(q_values),
        "train/critic_loss": np.mean(critic_losses),
        "n_updates": self._n_updates,
    })

    callback.on_rollout_end()
    return callback


def infer_whole_actions(self, base_action, delta_actions):

    # append a constant 1 to the action

    residual_action = (delta_actions + 1) / 2 * (
        self.residual_action_range[:, 1] -
        self.residual_action_range[:, 0]) + self.residual_action_range[:, 0]
    target_pos, target_rot = apply_delta_pose(base_action[..., :3],
                                              base_action[..., 3:7],
                                              residual_action[..., :6])
    target_finger_pose = base_action[..., 7:] + residual_action[..., 6:]

    next_actions = th.cat([target_pos, target_rot, target_finger_pose], dim=-1)
    return next_actions


def train_latent_noise(self,
                       gradient_steps: int,
                       batch_size: int = 100,
                       callback=None) -> None:
    # Switch to train mode (this affects batch norm / dropout)

    self.policy.set_training_mode(True)

    # Update learning rate according to lr schedule
    self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

    actor_losses, critic_losses, bc_losses = [], [], []
    for _ in range(gradient_steps):
        self._n_updates += 1
        # Sample replay buffer
        replay_data = self.replay_buffer.sample(
            batch_size,
            env=self._vec_normalize_env)  # type: ignore[union-attr]
        # For n-step replay, discount factor is gamma**n_steps (when no early termination)
        discounts = self.gamma

        with th.no_grad():

            noise = replay_data.actions.clone().data.normal_(
                0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip,
                                self.target_noise_clip)
            next_actions_noise = (
                self.actor_target(replay_data.next_observations) +
                noise).clamp(-1, 1)  #[...,22:]

            # get next action
            next_diffusion_actions = infer_diffusion(
                self, next_actions_noise[..., 22:],
                replay_data.next_observations, replay_data.cartesian_actions)
            next_actions = infer_whole_actions(self, next_diffusion_actions,
                                               next_actions_noise[..., :22])

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(
                replay_data.next_observations, next_actions),
                                   dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (
                1 - replay_data.dones) * discounts * next_q_values

        # Get current Q-values estimates for each critic network
        current_diffusion_actions = infer_diffusion(
            self,
            replay_data.actions[..., 22:],
            replay_data.observations,
            replay_data.cartesian_actions,
        )

        current_actions = infer_whole_actions(self, current_diffusion_actions,
                                              replay_data.actions[..., :22])

        current_q_values = self.critic(replay_data.observations,
                                       current_actions)

        # Compute critic loss
        critic_loss = sum(
            F.mse_loss(current_q, target_q_values)
            for current_q in current_q_values)
        assert isinstance(critic_loss, th.Tensor)
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:
            policy_actions = self.actor(replay_data.observations)
            policy_diffusion_actions = infer_diffusion(
                self,
                policy_actions[..., 22:],
                replay_data.observations,
                replay_data.cartesian_actions,
            )
            final_policy_actions = infer_whole_actions(
                self, policy_diffusion_actions, policy_actions[..., 22:])

            actor_loss = -self.critic.q1_forward(replay_data.observations,
                                                 final_policy_actions).mean()
            actor_losses.append(actor_loss.item())

            bc_loss = F.mse_loss(policy_actions[..., :22],
                                 replay_data.actions[..., :22],
                                 reduction="mean")
            actor_loss += self.lambda_bc * bc_loss
            bc_losses.append(bc_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(),
                          self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(),
                          self.actor_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.critic_batch_norm_stats,
                          self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats,
                          self.actor_batch_norm_stats_target, 1.0)

    self.logger.record("train/n_updates",
                       self._n_updates,
                       exclude="tensorboard")

    if len(actor_losses) > 0:
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))

    self.logger.record("train/critic_loss", np.mean(critic_losses))
    self.dump_logs()
    callback.on_rollout_end()


def co_train_latent(self,
                    gradient_steps: int,
                    batch_size: int = 100,
                    callback=None) -> None:
    # Switch to train mode (this affects batch norm / dropout)

    self.policy.set_training_mode(True)

    # Update learning rate according to lr schedule
    self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])

    actor_losses, critic_losses, bc_losses = [], [], []
    import pdb

    for _ in range(gradient_steps):

        self._n_updates += 1
        # Sample replay buffer
        online_buffer_num = int(batch_size * (1 - self.bc_ratio))
        offline_buffer_num = int(batch_size - online_buffer_num)
        online_replay_data = self.replay_buffer.sample(
            online_buffer_num, env=self._vec_normalize_env)

        offline_replay_data = self.bc_buffer.sample(
            offline_buffer_num, env=self._vec_normalize_env)
        discounts = self.gamma

        with th.no_grad():

            # rollout online data
            noise = online_replay_data.actions.clone().data.normal_(
                0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip,
                                self.target_noise_clip)
            next_actions_noise = (
                self.actor_target(online_replay_data.next_observations) +
                noise).clamp(-1, 1)

            # get next action
            next_online_actions = infer_diffusion(
                self, next_actions_noise, online_replay_data.next_observations,
                online_replay_data.cartesian_actions)

            next_online_actions = infer_whole_actions(
                self, next_online_actions, next_actions_noise[..., :22])

            # Compute the next Q-values: min over all critics targets
            next_q_values = th.cat(self.critic_target(
                online_replay_data.next_observations, next_online_actions),
                                   dim=1)
            next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
            target_online_q_values = online_replay_data.rewards + (
                1 - online_replay_data.dones) * discounts * next_q_values

            # rollout offline data
            noise = offline_replay_data.actions.clone().data.normal_(
                0, self.target_policy_noise)
            noise = noise.clamp(-self.target_noise_clip,
                                self.target_noise_clip)

            delta_actions = (
                self.actor_target(offline_replay_data.next_observations) +
                noise).clamp(-1, 1)[..., :22]
            # append a constant 1 to the action
            base_action = offline_replay_data.base_actions
            residual_action = (delta_actions + 1) / 2 * (
                self.residual_action_range[:, 1] - self.
                residual_action_range[:, 0]) + self.residual_action_range[:, 0]
            target_pos, target_rot = apply_delta_pose(base_action[..., :3],
                                                      base_action[..., 3:7],
                                                      residual_action[..., :6])
            target_finger_pose = base_action[..., 7:] + residual_action[...,
                                                                        6:]

            next_offline_actions = th.cat(
                [target_pos, target_rot, target_finger_pose], dim=-1)
            # Compute the next Q-values: min over all critics targets
            next_offline_q_values = th.cat(self.critic_target(
                offline_replay_data.next_observations, next_offline_actions),
                                           dim=1)
            next_offline_q_values, _ = th.min(next_offline_q_values,
                                              dim=1,
                                              keepdim=True)
            target_offline_q_values = offline_replay_data.rewards + (
                1 -
                offline_replay_data.dones) * discounts * next_offline_q_values

        current_diffusion_actions = infer_diffusion(
            self,
            online_replay_data.actions[..., 22:],
            online_replay_data.observations,
            online_replay_data.cartesian_actions,
        )
        current_diffusion_actions = infer_whole_actions(
            self, current_diffusion_actions,
            online_replay_data.actions[..., :22])

        current_online_q_values = self.critic(online_replay_data.observations,
                                              current_diffusion_actions)

        current_offline_actions = infer_whole_actions(
            self, offline_replay_data.base_actions,
            offline_replay_data.actions[..., :22])
        current_offline_q_values = self.critic(
            offline_replay_data.observations, current_offline_actions)

        # Compute critic loss
        critic_loss = sum(
            F.mse_loss(current_q, target_offline_q_values)
            for current_q in current_offline_q_values) + sum(
                F.mse_loss(current_q, target_online_q_values)
                for current_q in current_online_q_values)
        assert isinstance(critic_loss, th.Tensor)
        critic_losses.append(critic_loss.item())

        # Optimize the critics
        self.critic.optimizer.zero_grad()
        critic_loss.backward()
        self.critic.optimizer.step()

        # Delayed policy updates
        if self._n_updates % self.policy_delay == 0:

            # # Compute actor loss
            # actor_loss = -self.critic.q1_forward(
            #     replay_data.observations,
            #     self.actor(replay_data.observations)).mean()

            policy_actions = self.actor(online_replay_data.observations)
            bc_actions = self.actor(offline_replay_data.observations)

            online_actions = infer_diffusion(
                self, policy_actions[...,
                                     22:], online_replay_data.observations,
                online_replay_data.cartesian_actions)

            online_actions = infer_whole_actions(self, online_actions,
                                                 policy_actions[..., :22])

            actor_loss = -self.critic.q1_forward(
                online_replay_data.observations, online_actions).mean()
            # Behavior cloning term
            bc_loss = F.mse_loss(bc_actions,
                                 offline_replay_data.actions,
                                 reduction="mean")
            bc_losses.append(bc_loss.item())

            bc_loss = self.lambda_bc * bc_loss

            actor_loss = actor_loss + bc_loss
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            polyak_update(self.critic.parameters(),
                          self.critic_target.parameters(), self.tau)
            polyak_update(self.actor.parameters(),
                          self.actor_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.critic_batch_norm_stats,
                          self.critic_batch_norm_stats_target, 1.0)
            polyak_update(self.actor_batch_norm_stats,
                          self.actor_batch_norm_stats_target, 1.0)

    self.logger.record("train/n_updates",
                       self._n_updates,
                       exclude="tensorboard")
    if len(actor_losses) > 0:
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/bc_loss", np.mean(bc_losses))
    self.logger.record("train/critic_loss", np.mean(critic_losses))
    self.dump_logs()
    callback.on_rollout_end()
