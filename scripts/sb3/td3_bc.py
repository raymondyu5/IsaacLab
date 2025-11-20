from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from scripts.rsrl.agent.residual_offline_wrapper import OffPolicyAlgorithm

from stable_baselines3.common.policies import BasePolicy, ContinuousCritic
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
try:
    from scripts.rsrl.agent.td3bc_policy import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy
    from scripts.rsrl.utils.robot_rl_utils import *
except:
    from isaac_scripts.rsrl.agent.td3bc_policy import Actor, CnnPolicy, MlpPolicy, MultiInputPolicy, TD3Policy
    from isaac_scripts.rsrl.utils.robot_rl_utils import *

SelfTD3 = TypeVar("SelfTD3", bound="TD3")

try:
    from scripts.rsrl.agent.algo import BaseAlgorithm

    from scripts.rsrl.utils.residual_sb3_buffer import ReplayBuffer, DictReplayBufferSamples
except:
    from isaac_scripts.rsrl.agent.algo import BaseAlgorithm

try:
    from scripts.sb3.train_function import train_latent_noise_only, train_only, train_latent_noise, co_train_latent
except:
    from isaac_scripts.sb3.train_function import train_latent_noise_only, train_only, train_latent_noise, co_train_latent


class TD3BC(OffPolicyAlgorithm):
    """
    Twin Delayed DDPG (TD3)
    Addressing Function Approximation Error in Actor-Critic Methods.

    Original implementation: https://github.com/sfujim/TD3
    Paper: https://arxiv.org/abs/1802.09477
    Introduction to TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param n_steps: When n_step > 1, uses n-step return (with the NStepReplayBuffer) when updating the Q-value network.
    :param policy_delay: Policy and target networks will only be updated once every policy_delay steps
        per training steps. The Q values will be updated policy_delay more often (update every training step).
    :param target_policy_noise: Standard deviation of Gaussian noise added to target policy
        (smoothing noise)
    :param target_noise_clip: Limit for absolute value of target policy smoothing noise.
    :param stats_window_size: Window size for the rollout logging, specifying the number of episodes to average
        the reported success rate, mean episode length, and mean reward over
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation. See :ref:`td3_policies`
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: TD3Policy
    actor: Actor
    actor_target: Actor
    critic: ContinuousCritic
    critic_target: ContinuousCritic

    def __init__(
        self,
        policy: Union[str, type[TD3Policy]],
        env=None,
        observation_space: spaces.Space = None,
        action_space: spaces.Space = None,
        learning_rate: Union[float, Schedule] = 1e-3,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, tuple[int, str]] = 1,
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        initial_buffer: bool = True,
        lambda_bc=0.2,
        residual_action_range=[0.015, 0.02, 0.1],
        bc_buffer=None,
        bc_ratio=0.5,
        use_latent_noise=False,
        latent_model=None,
        use_latent_noise_only=False,
        save_buffer=False,
    ):
        self.bc_ratio = bc_ratio
        self.device = device
        self.latent_model = latent_model
        self.use_latent_noise = use_latent_noise
        self.use_latent_noise_only = use_latent_noise_only

        self.lambda_bc = lambda_bc
        self.observation_space = observation_space
        self.action_space = action_space
        self.residual_action_range = torch.as_tensor(
            [float(residual_action_range[0])] * 3 +
            [float(residual_action_range[1])] * 3 +
            [float(residual_action_range[2])] * 16)
        self.residual_action_range = torch.stack(
            [-self.residual_action_range, self.residual_action_range],
            dim=1).cuda()
        self.bc_buffer = bc_buffer

        if self.use_latent_noise_only:
            self.train = train_latent_noise_only
        elif latent_model is not None:
            self.train = train_latent_noise
        elif self.bc_buffer is None:
            self.train = train_only

        else:

            if self.latent_model is not None:
                self.train = co_train_latent
            else:
                self.train = self.co_train

        super().__init__(policy,
                         observation_space,
                         action_space,
                         learning_rate,
                         buffer_size,
                         learning_starts,
                         batch_size,
                         tau,
                         gamma,
                         train_freq,
                         gradient_steps,
                         action_noise=action_noise,
                         replay_buffer_class=replay_buffer_class,
                         replay_buffer_kwargs=replay_buffer_kwargs,
                         optimize_memory_usage=optimize_memory_usage,
                         n_steps=n_steps,
                         policy_kwargs=policy_kwargs,
                         stats_window_size=stats_window_size,
                         tensorboard_log=tensorboard_log,
                         verbose=verbose,
                         device=device,
                         seed=seed,
                         sde_support=False,
                         supported_action_spaces=(spaces.Box, ),
                         support_multi_env=True,
                         initial_buffer=initial_buffer,
                         save_buffer=save_buffer)
        self.policy_delay = policy_delay
        self.target_noise_clip = target_noise_clip
        self.target_policy_noise = target_policy_noise

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Running mean and running var
        self.actor_batch_norm_stats = get_parameters_by_name(
            self.actor, ["running_"])
        self.critic_batch_norm_stats = get_parameters_by_name(
            self.critic, ["running_"])
        self.actor_batch_norm_stats_target = get_parameters_by_name(
            self.actor_target, ["running_"])
        self.critic_batch_norm_stats_target = get_parameters_by_name(
            self.critic_target, ["running_"])

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.actor_target = self.policy.actor_target
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def learn(
        self: SelfTD3,
        total_timesteps: int,
        iteration: Optional[int] = None,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfTD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + [
            "actor", "critic", "actor_target", "critic_target"
        ]  # noqa: RUF005

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        return state_dicts, []

    def sample_data(self, batch_size):
        # Sample from each buffer
        online_buffer_num = int(batch_size * (1 - self.bc_ratio))
        offline_buffer_num = int(batch_size - online_buffer_num)
        online_replay_data = self.replay_buffer.sample(
            online_buffer_num, env=self._vec_normalize_env)

        offline_replay_data = self.bc_buffer.sample(
            offline_buffer_num, env=self._vec_normalize_env)

        # ---- Shuffle order ----

        perm = th.randperm(batch_size, device=self.device)

        # Concatenate
        actions = th.cat(
            [online_replay_data.actions, offline_replay_data.actions], dim=0)
        rewards = th.cat(
            [online_replay_data.rewards, offline_replay_data.rewards], dim=0)
        dones = th.cat([online_replay_data.dones, offline_replay_data.dones],
                       dim=0)
        robot_actions = th.cat([
            online_replay_data.robot_actions, offline_replay_data.robot_actions
        ],
                               dim=0)
        base_actions = th.cat([
            online_replay_data.base_actions, offline_replay_data.base_actions
        ],
                              dim=0)
        cartesian_actions = th.cat([
            online_replay_data.cartesian_actions,
            offline_replay_data.cartesian_actions
        ],
                                   dim=0)

        observations = self.construct_combined_obs(
            online_replay_data.observations, offline_replay_data.observations,
            perm)
        next_observations = self.construct_combined_obs(
            online_replay_data.next_observations,
            offline_replay_data.next_observations, perm)

        actions = actions[perm]
        rewards = rewards[perm]
        dones = dones[perm]
        robot_actions = robot_actions[perm]
        base_actions = base_actions[perm]
        cartesian_actions = cartesian_actions[perm]

        return DictReplayBufferSamples(
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            dones=dones,
            rewards=rewards,
            robot_actions=robot_actions,
            base_actions=base_actions,
            cartesian_actions=cartesian_actions,
        ), online_replay_data, offline_replay_data

    def construct_combined_obs(self, obs01, obs02, perm):
        combined = {}
        for k in obs01.keys():

            v1, v2 = obs01[k], obs02[k]

            combined[k] = th.cat([v1, v2], dim=0)[perm]
        return combined

    def co_train(self,
                 gradient_steps: int,
                 batch_size: int = 100,
                 callback=None) -> None:
        # Switch to train mode (this affects batch norm / dropout)

        self.policy.set_training_mode(True)

        # Update learning rate according to lr schedule
        self._update_learning_rate(
            [self.actor.optimizer, self.critic.optimizer])

        actor_losses, critic_losses, bc_losses = [], [], []
        import pdb

        for _ in range(gradient_steps):

            self._n_updates += 1
            # Sample replay buffer
            replay_data, _, offline_replay_data = self.sample_data(batch_size)
            # offline_replay_data = self.bc_buffer.sample(
            #     batch_size, env=self._vec_normalize_env)
            # replay_data = offline_replay_data

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
                        noise).clamp(-1, 1)[..., :22]
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
                                                     7:] + residual_action[...,
                                                                           6:]

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

                bc_actions = self.actor(offline_replay_data.observations)

                # Q term (same as TD3)

                if self.policy.q_type == "res":
                    policy_actions = policy_actions

                    q_loss = -self.critic.q1_forward(replay_data.observations,
                                                     policy_actions).mean()
                elif self.policy.q_type == "sum":
                    # append a constant 1 to the action

                    if not self.use_latent_noise:
                        base_action = replay_data.base_actions
                        residual_action = (policy_actions + 1) / 2 * (
                            self.residual_action_range[:, 1] -
                            self.residual_action_range[:, 0]
                        ) + self.residual_action_range[:, 0]
                        target_pos, target_rot = apply_delta_pose(
                            base_action[..., :3], base_action[..., 3:7],
                            residual_action[..., :6])
                        target_finger_pose = base_action[
                            ..., 7:] + residual_action[..., 6:]
                        combined_policy_actions = th.cat(
                            [target_pos, target_rot, target_finger_pose],
                            dim=-1)
                        q_loss = -self.critic.q1_forward(
                            replay_data.observations,
                            combined_policy_actions).mean()

                elif self.policy.q_type == "concat":
                    combined_policy_actions = th.cat(
                        [replay_data.base_actions, policy_actions], dim=-1)
                    q_loss = -self.critic.q1_forward(
                        replay_data.observations,
                        combined_policy_actions).mean()

                # Behavior cloning term
                bc_loss = F.mse_loss(bc_actions,
                                     offline_replay_data.actions,
                                     reduction="mean")
                bc_losses.append(bc_loss.item())

                bc_loss = self.lambda_bc * bc_loss

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
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.dump_logs()

        callback.on_rollout_end()
