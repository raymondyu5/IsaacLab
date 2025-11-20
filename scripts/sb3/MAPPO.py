import warnings
from typing import Any, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch.nn import functional as F

from scripts.sb3.buffers import RolloutBuffer
from scripts.sb3.multiagent_on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn
from scripts.sb3.bimanual.bistd_on_policy_algo import BiStdActorCriticPolicy
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

from stable_baselines3.common.base_class import BaseAlgorithm
from scripts.sb3.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
import sys

SelfPPO = TypeVar("SelfPPO", bound="PPO")

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm",
                                bound="OnPolicyAlgorithm")
import time


class IPPO:

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "cuda",
        _init_setup_model: bool = True,
        gpu_buffer: bool = False,
        args_cli: Optional[Any] = None,
    ):

        policy_aliases = {
            "MlpPolicy": ActorCriticPolicy,
            "CnnPolicy": ActorCriticCnnPolicy,
            "MultiInputPolicy": MultiInputActorCriticPolicy,
            "BiStdPolicy": BiStdActorCriticPolicy,
        }
        if rollout_buffer_kwargs is None:
            rollout_buffer_kwargs = {"gpu_buffer": gpu_buffer}
        else:
            rollout_buffer_kwargs["gpu_buffer"] = gpu_buffer
        self.env = env

        self.use_sde = use_sde
        self.sde_sample_freq = sde_sample_freq

        self.action_dim = env.action_space.shape[0]
        self.observation_dim = env.observation_space.shape[0]
        self.device = device

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.verbose = verbose

        self.policies = []
        self.args_cli = args_cli

        if args_cli.add_left_hand:

            self.left_hand_policy = OnPolicyAlgorithm(
                policy_aliases[policy],
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                rollout_buffer_class=rollout_buffer_class,
                rollout_buffer_kwargs=rollout_buffer_kwargs,
                stats_window_size=stats_window_size,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                device=device,
                seed=seed,
                _init_setup_model=False,
                supported_action_spaces=(
                    spaces.Box,
                    spaces.Discrete,
                    spaces.MultiDiscrete,
                    spaces.MultiBinary,
                ),
            )
            self.policies.append(self.left_hand_policy)

        if args_cli.add_right_hand:

            self.right_hand_policy = OnPolicyAlgorithm(
                policy_aliases[policy],
                env,
                learning_rate=learning_rate,
                n_steps=n_steps,
                gamma=gamma,
                gae_lambda=gae_lambda,
                ent_coef=ent_coef,
                vf_coef=vf_coef,
                max_grad_norm=max_grad_norm,
                use_sde=use_sde,
                sde_sample_freq=sde_sample_freq,
                rollout_buffer_class=rollout_buffer_class,
                rollout_buffer_kwargs=rollout_buffer_kwargs,
                stats_window_size=stats_window_size,
                tensorboard_log=tensorboard_log,
                policy_kwargs=policy_kwargs,
                verbose=verbose,
                device=device,
                seed=seed,
                _init_setup_model=False,
                supported_action_spaces=(
                    spaces.Box,
                    spaces.Discrete,
                    spaces.MultiDiscrete,
                    spaces.MultiBinary,
                ))

            self.policies.append(self.right_hand_policy)

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()
        self._last_obs = None

    def _setup_model(self) -> None:
        for hand_policy in self.policies:
            hand_policy._setup_model()

        if self.args_cli.add_left_hand:
            from copy import deepcopy
            init_state_dict = deepcopy(
                self.left_hand_policy.policy.state_dict())
            self.policies[-1].policy.load_state_dict(init_state_dict)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        for agent_id, hand_policy in reversed(list(enumerate(self.policies))):

            hand_policy.policy.set_training_mode(True)

            # Update optimizer learning rate

            hand_policy._update_learning_rate(hand_policy.policy.optimizer)

            # Compute current clip range
            clip_range = self.clip_range(
                hand_policy._current_progress_remaining
            )  # type: ignore[operator]
            # Optional: clip range for the value function
            if self.clip_range_vf is not None:
                clip_range_vf = self.clip_range_vf(
                    hand_policy._current_progress_remaining
                )  # type: ignore[operator]

            entropy_losses = []
            pg_losses, value_losses = [], []
            clip_fractions = []

            continue_training = True
            # train for n_epochs epochs

            for epoch in range(self.n_epochs):
                approx_kl_divs = []
                # Do a complete pass on the rollout buffer
                for rollout_data in hand_policy.rollout_buffer.get(
                        self.batch_size):
                    actions = rollout_data.actions
                    # if isinstance(self.action_space, spaces.Discrete):
                    #     # Convert discrete action from float to long
                    #     actions = rollout_data.actions.long().flatten()

                    values, log_prob, entropy = hand_policy.policy.evaluate_actions(
                        rollout_data.observations, actions)

                    values = values.flatten()
                    # Normalize advantage
                    advantages = rollout_data.advantages
                    # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                    if self.normalize_advantage and len(advantages) > 1:
                        advantages = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8)

                    # ratio between old and new policy, should be one at the first iteration
                    ratio = th.exp(log_prob - rollout_data.old_log_prob)

                    # clipped surrogate loss
                    policy_loss_1 = advantages * ratio
                    policy_loss_2 = advantages * th.clamp(
                        ratio, 1 - clip_range, 1 + clip_range)
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                    # Logging
                    pg_losses.append(policy_loss.item())
                    clip_fraction = th.mean((th.abs(ratio - 1)
                                             > clip_range).float()).item()
                    clip_fractions.append(clip_fraction)

                    if self.clip_range_vf is None:
                        # No clipping
                        values_pred = values
                    else:
                        # Clip the difference between old and new value
                        # NOTE: this depends on the reward scaling
                        values_pred = rollout_data.old_values + th.clamp(
                            values - rollout_data.old_values, -clip_range_vf,
                            clip_range_vf)
                    # Value loss using the TD(gae_lambda) target
                    value_loss = F.mse_loss(rollout_data.returns, values_pred)
                    value_losses.append(value_loss.item())

                    # Entropy loss favor exploration
                    if entropy is None:
                        # Approximate entropy when no analytical form
                        entropy_loss = -th.mean(-log_prob)
                    else:
                        entropy_loss = -th.mean(entropy)

                    entropy_losses.append(entropy_loss.item())

                    loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                    # Calculate approximate form of reverse KL Divergence for early stopping
                    # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                    # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                    # and Schulman blog: http://joschu.net/blog/kl-approx.html
                    with th.no_grad():
                        log_ratio = log_prob - rollout_data.old_log_prob
                        approx_kl_div = th.mean((th.exp(log_ratio) - 1) -
                                                log_ratio).cpu().numpy()
                        approx_kl_divs.append(approx_kl_div)

                    if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                        continue_training = False
                        if self.verbose >= 1:
                            print(
                                f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                            )
                        break

                    # Optimization step
                    hand_policy.policy.optimizer.zero_grad()
                    loss.backward()

                    # Clip grad norm

                    th.nn.utils.clip_grad_norm_(
                        hand_policy.policy.parameters(), self.max_grad_norm)
                    hand_policy.policy.optimizer.step()

                hand_policy._n_updates += 1
                if not continue_training:
                    break

            # Logs

            print('======================')
            hand_policy.logger.record(
                f"rollout/agent_{agent_id}_ep_rew_mean",
                safe_mean([
                    ep_info[f"agent_{agent_id}_r"]
                    for ep_info in hand_policy.ep_info_buffer
                ]))
            hand_policy.logger.record(
                f"rollout/agent_{agent_id}_ep_len_mean",
                safe_mean(
                    [ep_info["l"] for ep_info in hand_policy.ep_info_buffer]))
            hand_policy.logger.record(f"train/agent_{agent_id}_entropy_loss",
                                      np.mean(entropy_losses))
            hand_policy.logger.record(
                f"train/agent_{agent_id}_policy_gradient_loss",
                np.mean(pg_losses))
            hand_policy.logger.record(f"train/agent_{agent_id}_value_loss",
                                      np.mean(value_losses))
            hand_policy.logger.record(f"train/agent_{agent_id}_approx_kl",
                                      np.mean(approx_kl_divs))
            hand_policy.logger.record(f"train/agent_{agent_id}_clip_fraction",
                                      np.mean(clip_fractions))
            hand_policy.logger.record(f"train/agent_{agent_id}_loss",
                                      loss.item())
            # self.logger.record("train/explained_variance", explained_var)
            if hasattr(hand_policy.policy, "log_std"):
                hand_policy.logger.record(
                    f"train/agent_{agent_id}_std",
                    th.exp(hand_policy.policy.log_std).mean().item())

            hand_policy.logger.record(f"train/agent_{agent_id}_n_updates",
                                      hand_policy._n_updates,
                                      exclude="tensorboard")
            hand_policy.logger.record(f"train/agent_{agent_id}_clip_range",
                                      clip_range)
            if self.clip_range_vf is not None:
                hand_policy.record(f"train/agent_{agent_id}_clip_range_vf",
                                   clip_range_vf)
            del hand_policy

    def _dump_logs(self, iteration: int) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        for _, hand_policy in enumerate(self.policies):
            time_elapsed = max((time.time_ns() - hand_policy.start_time) / 1e9,
                               sys.float_info.epsilon)
            fps = int((hand_policy.num_timesteps -
                       hand_policy._num_timesteps_at_start) / time_elapsed)
            hand_policy.logger.record("time/iterations",
                                      iteration,
                                      exclude="tensorboard")

            hand_policy.logger.record("time/fps", fps)

            hand_policy.logger.record("time/total_timesteps",
                                      hand_policy.num_timesteps,
                                      exclude="tensorboard")

            hand_policy.logger.record("rollout/rollout_rew_mean",
                                      self.last_rollout_reward.tolist())

            hand_policy.logger.dump(step=hand_policy.num_timesteps)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        if self._last_obs is None:
            self._last_obs = env.reset()

        # Switch to eval mode (this affects batch norm / dropout)

        for hand_policy in self.policies:

            hand_policy.policy.set_training_mode(False)
            hand_policy.rollout_buffer.reset()

        n_steps = 0

        # Sample new weights for the state dependent exploration
        if self.use_sde:
            for hand_policy in self.policies:
                hand_policy.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        self.last_rollout_reward = 0
        num_rollouts = 0

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                for hand_policy in self.policies:
                    hand_policy.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict

                actions = []
                values = []
                log_probs = []
                for agent_id, hand_policy in reversed(
                        list(enumerate(self.policies))):

                    obs_tensor = obs_as_tensor(self._last_obs[:, agent_id],
                                               self.device)

                    action, value, log_prob = hand_policy.policy(obs_tensor)
                    actions.append(action.unsqueeze(1).clone())
                    values.append(value.reshape(-1, 1).clone())
                    log_probs.append(log_prob.reshape(-1, 1).clone())

                actions = th.cat(actions, dim=1)
                values = th.cat(values, dim=1)
                log_probs = th.cat(log_probs, dim=1)

            actions = actions.cpu().numpy()
            # Rescale and perform action
            clipped_actions = actions

            clipped_actions = np.stack([
                np.clip(actions[:, i], self.policies[i].action_space.low,
                        self.policies[i].action_space.high)
                for i in range(len(self.policies))
            ],
                                       axis=1)

            new_obs, shared_obs, rewards, dones, infos = env.step(
                clipped_actions)

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            for hand_policy in self.policies:
                hand_policy._update_info_buffer(infos, dones)
                hand_policy.num_timesteps += env.num_envs
            n_steps += 1

            # Handle timeout by bootstrapping with value function
            # see GitHub issue #633
            for idx in range(len(dones)):  # loop over envs
                if infos[idx].get("terminal_observation") is None:
                    continue
                if not infos[idx].get("TimeLimit.truncated", False):
                    continue

                # Get terminal observation for this environment
                terminal_obs = obs_as_tensor(
                    infos[idx]["terminal_observation"], self.device)

                num_rollouts += 1
                with th.no_grad():
                    for agent_id, hand_policy in enumerate(self.policies):
                        # Check if the agent is done
                        if not dones[idx][agent_id]:
                            continue  # skip if this agent did not terminate

                        # Predict value for bootstrapping
                        agent_terminal_obs = terminal_obs[agent_id].unsqueeze(
                            0)
                        terminal_value = hand_policy.policy.predict_values(
                            agent_terminal_obs)[0].clone()

                        rewards[idx][agent_id] += self.gamma * terminal_value

            for agent_id, hand_policy in reversed(
                    list(enumerate(self.policies))):

                hand_policy.rollout_buffer.add(
                    self._last_obs[:, agent_id],  # type: ignore[arg-type]
                    actions[:, agent_id],  # type: ignore[arg-type]
                    rewards[:, agent_id],
                    hand_policy._last_episode_starts,  # type: ignore[arg-type]
                    values[:, agent_id],
                    log_probs[:, agent_id],
                )
                hand_policy._last_episode_starts = dones[:,
                                                         agent_id]  # type: ignore[arg-type]

            self._last_obs = new_obs  # type: ignore[assignment]

            self.last_rollout_reward += rewards.sum()
        self.last_rollout_reward /= num_rollouts
        with th.no_grad():

            # Compute value for the last timestep

            for agent_id, hand_policy in reversed(
                    list(enumerate(self.policies))):
                new_obs_tensor = obs_as_tensor(new_obs[:, agent_id],
                                               self.device).clone()

                # Compute value for the last timestep
                values = hand_policy.policy.predict_values(new_obs_tensor)

                hand_policy.rollout_buffer.compute_returns_and_advantage(
                    last_values=values.clone(), dones=dones[:, agent_id])

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True

    def learn(
        self: SelfOnPolicyAlgorithm,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfOnPolicyAlgorithm:
        iteration = 0

        for hand_policy in self.policies:
            total_timesteps, callback = hand_policy._setup_learn(
                total_timesteps,
                callback,
                reset_num_timesteps,
                tb_log_name,
                progress_bar,
            )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while hand_policy.num_timesteps < total_timesteps:
            start = time.time()
            continue_training = self.collect_rollouts(
                self.env, callback, n_rollout_steps=self.n_steps)
            print("collect_rollouts time", time.time() - start)

            if not continue_training:
                break

            iteration += 1
            for hand_policy in self.policies:
                hand_policy._update_current_progress_remaining(
                    hand_policy.num_timesteps, total_timesteps)

            self.train()
            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:

                self._dump_logs(iteration)

            print("train time", time.time() - start)

        callback.on_training_end()

        return self
