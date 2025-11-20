import copy

import sys
import time
import warnings
from typing import Any, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from scripts.sb3.multiagent.multiagent_buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from scripts.sb3.multiagent.multiagent_policy_wrapper import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

SelfOnPolicyAlgorithm = TypeVar("SelfOnPolicyAlgorithm",
                                bound="OnPolicyAlgorithm")
import time


class OnPolicyAlgorithm(BaseAlgorithm):

    rollout_buffer: RolloutBuffer
    policy: ActorCriticPolicy

    def __init__(self,
                 policy: Union[str, type[ActorCriticPolicy]],
                 env: Union[GymEnv, str],
                 learning_rate: Union[float, Schedule],
                 n_steps: int,
                 gamma: float,
                 gae_lambda: float,
                 ent_coef: float,
                 vf_coef: float,
                 max_grad_norm: float,
                 use_sde: bool,
                 sde_sample_freq: int,
                 rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
                 rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
                 stats_window_size: int = 100,
                 tensorboard_log: Optional[str] = None,
                 monitor_wrapper: bool = True,
                 policy_kwargs: Optional[dict[str, Any]] = None,
                 verbose: int = 0,
                 seed: Optional[int] = None,
                 device: Union[th.device, str] = "auto",
                 _init_setup_model: bool = True,
                 supported_action_spaces: Optional[tuple[type[spaces.Space],
                                                         ...]] = None,
                 share_policy=False,
                 use_multi_agent=True):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer_class = rollout_buffer_class
        self.rollout_buffer_kwargs = rollout_buffer_kwargs or {}
        self.shared_observation_space = env.shared_observation_space
        self.shared_action_space = env.shared_action_space
        self.share_policy = share_policy
        self.use_multi_agent = use_multi_agent

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.rollout_buffer_class is None:

            self.rollout_buffer_class = RolloutBuffer

        self.rollout_buffer = self.rollout_buffer_class(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.shared_observation_space,
            self.action_space,
            self.shared_action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
            **self.rollout_buffer_kwargs,
        )

        if not self.share_policy:
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.shared_observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                share_policy=self.share_policy,
                use_multi_agent=self.use_multi_agent,
                **self.policy_kwargs)
        else:
            self.policy = self.policy_class(  # type: ignore[assignment]
                self.observation_space,
                self.shared_observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                share_policy=self.share_policy,
                use_multi_agent=self.use_multi_agent,
                **self.policy_kwargs)
        self.policy = self.policy.to(self.device)

        # Warn when not using CPU with MlpPolicy
        self._maybe_recommend_cpu()

    def _maybe_recommend_cpu(self,
                             mlp_class_name: str = "ActorCriticPolicy"
                             ) -> None:
        """
        Recommend to use CPU only when using A2C/PPO with MlpPolicy.

        :param: The name of the class for the default MlpPolicy.
        """
        policy_class_name = self.policy_class.__name__
        if self.device != th.device(
                "cpu") and policy_class_name == mlp_class_name:
            warnings.warn(
                f"You are trying to run {self.__class__.__name__} on the GPU, "
                "but it is primarily intended to run on the CPU when not using a CNN policy "
                f"(you are using {policy_class_name} which should be a MlpPolicy). "
                "See https://github.com/DLR-RM/stable-baselines3/issues/1245 "
                "for more info. "
                "You can pass `device='cpu'` or `export CUDA_VISIBLE_DEVICES=` to force using the CPU."
                "Note: The model will train, but the GPU utilization will be poor and "
                "the training might take longer than on CPU.",
                UserWarning,
            )

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

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            start = time.time()
            continue_training = self.collect_rollouts(
                self.env,
                callback,
                self.rollout_buffer,
                n_rollout_steps=self.n_steps)
            print("collect_rollouts time", time.time() - start)

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps,
                                                    total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                assert self.ep_info_buffer is not None
                self._dump_logs(iteration)

            self.train()
            print("train time", time.time() - start)

        callback.on_training_end()

        return self
