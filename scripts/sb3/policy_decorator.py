ALGO_NAME = 'PolicyDecorator-DiffusionPolicy-rgbd'

import os
import argparse
import random
from distutils.util import strtobool

os.environ["OMP_NUM_THREADS"] = "1"

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter

import datetime
from collections import defaultdict, deque


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):

    def __init__(self, observation_space, action_space):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(
                np.array(observation_space.shape).prod() +
                np.prod(action_space.shape), 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            layer_init(nn.Linear(256, 1), std=0.01),
        )

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):

    def __init__(self, env, args):
        super().__init__()
        obs_dim = np.array(env.single_observation_space.shape).prod()
        input_dim = obs_dim if args.actor_input == 'obs' else obs_dim + np.prod(
            env.single_action_space.shape)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.fc_mean = layer_init(nn.Linear(
            256, np.prod(env.single_action_space.shape)),
                                  std=0.01)
        self.fc_logstd = layer_init(nn.Linear(
            256, np.prod(env.single_action_space.shape)),
                                    std=0.01)

        # action rescaling
        h, l = env.single_action_space.high, env.single_action_space.low
        self.register_buffer("action_scale",
                             torch.tensor((h - l) / 2.0, dtype=torch.float32))
        self.register_buffer("action_bias",
                             torch.tensor((h + l) / 2.0, dtype=torch.float32))
        # will be saved in the state_dict

    def forward(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_eval_action(self, x):
        x = self.backbone(x)
        mean = self.fc_mean(x)
        action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample(
        )  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        return super().to(device)


def to_tensor(x, device='cuda'):
    if isinstance(x, dict):
        return {k: to_tensor(v, device) for k, v in x.items()}
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)


import wandb


class PolicyDecotrator:

    def __init__(self, base_policy, observation_space, action_space):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.base_policy = base_policy

        self.qf1 = SoftQNetwork(observation_space,
                                action_space).to(self.device)

        self.qf2 = SoftQNetwork(observation_space,
                                action_space).to(self.device)
        if is_ms1_env(args.env_id):
            for m in list(res_actor.modules()) + list(qf1.modules()) + list(
                    qf2.modules()):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight, gain=1)
                    torch.nn.init.zeros_(m.bias)
        qf1_target = SoftQNetwork(dummy_env).to(device)
        qf2_target = SoftQNetwork(dummy_env).to(device)
        qf1_target.load_state_dict(qf1.state_dict())
        qf2_target.load_state_dict(qf2.state_dict())
        q_optimizer = optim.Adam(list(qf1.parameters()) +
                                 list(qf2.parameters()),
                                 lr=args.q_lr)
        actor_optimizer = optim.Adam(list(res_actor.parameters()),
                                     lr=args.policy_lr)

        # Automatic entropy tuning
        if args.autotune:
            target_entropy = -torch.prod(
                torch.Tensor(
                    dummy_env.single_action_space.shape).to(device)).item()
            log_sac_alpha = torch.zeros(1, requires_grad=True, device=device)
            sac_alpha = log_sac_alpha.exp().item()
            a_optimizer = optim.Adam([log_sac_alpha], lr=args.q_lr)
        else:
            sac_alpha = args.sac_alpha

        dummy_env.single_observation_space.dtype = np.float32
        rb = ReplayBuffer(
            args.buffer_size,
            dummy_env.single_observation_space,
            dummy_env.single_action_space if args.critic_input == 'res'
            and args.actor_input == 'obs' else gym.spaces.Box(
                low=-np.inf, high=np.inf, shape=(total_act_dim * 3, )),
            device,
            n_envs=args.num_envs,
            handle_timeout_termination=
            False,  # stable-baselines3 has not fully supported Gymnasium's termination signal
        )

        # TRY NOT TO MODIFY: start the game
        obs_seq, info = envs.reset(
            seed=args.seed
        )  # in Gymnasium, seed is given to reset() instead of seed()
        global_step = 0
        global_update = 0
        learning_has_started = False
        num_updates_per_training = int(args.training_freq * args.utd)
        result = defaultdict(list)
        step_in_episodes = np.zeros((args.num_envs, 1, 1), dtype=np.int32)
        timer = NonOverlappingTimeProfiler()

        while global_step < args.total_timesteps:

            # Collect samples from environemnts
            for local_step in range(args.training_freq // args.num_envs):
                global_step += 1 * args.num_envs

                obs_seq_tensor = to_tensor(obs_seq, device)
                base_act_seq_tensor, obs_seq_embedding_tensor = base_policy.get_eval_action(
                    obs_seq_tensor, return_obs_embedding=True)
                obs_embedding_tensor = obs_seq_embedding_tensor[:,
                                                                -base_policy.
                                                                obs_embedding_dim:].detach(
                                                                )  # most recent obs
                base_act_seq = base_act_seq_tensor.cpu().numpy(
                )  # (B, act_horizon, act_dim)
                base_actions = base_act_seq.reshape(-1, total_act_dim)
                res_ratio = min(global_step / args.prog_explore, 1)
                enable_res_masks = np.random.rand(args.num_envs) < res_ratio

                # ALGO LOGIC: put action logic here
                if not learning_has_started:
                    res_actions = np.array([
                        dummy_env.single_action_space.sample()
                        for _ in range(envs.num_envs)
                    ])  # (B, act_horizon*act_dim)
                    res_actions[np.logical_not(enable_res_masks)] = 0.0
                else:
                    actor_input = obs_embedding_tensor if args.actor_input == 'obs' else torch.cat(
                        [
                            obs_embedding_tensor,
                            base_act_seq_tensor.flatten(start_dim=1)
                        ],
                        dim=1)
                    res_actions, _, _ = res_actor.get_action(actor_input)
                    res_actions = res_actions.detach().cpu().numpy(
                    )  # (B, act_horizon*act_dim)
                    res_actions[np.logical_not(enable_res_masks)] = 0.0

                res_act_seq = res_actions.reshape(-1, args.act_horizon,
                                                  act_dim)
                scaled_res_seq = args.res_scale * res_act_seq  # (B, act_horizon, act_dim)
                final_act_seq = base_act_seq + scaled_res_seq

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs_seq, rewards, terminations, truncations, infos = envs.step(
                    final_act_seq)
                rewards = rewards - 1.0  # negative reward + bootstrap at truncated yields best results

                # TRY NOT TO MODIFY: record rewards for plotting purposes
                result = collect_episode_info(infos, result)

                # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
                real_next_obs_seq = {
                    k: v.copy()
                    for k, v in next_obs_seq.items()
                }
                if args.bootstrap_at_done == 'never':
                    stop_bootstrap = truncations | terminations  # always stop bootstrap when episode ends
                else:
                    if args.bootstrap_at_done == 'always':
                        need_final_obs = truncations | terminations  # always need final obs when episode ends
                        stop_bootstrap = np.zeros_like(
                            terminations, dtype=bool)  # never stop bootstrap
                    else:  # bootstrap at truncated
                        need_final_obs = truncations & (
                            ~terminations
                        )  # only need final obs when truncated and not terminated
                        stop_bootstrap = terminations  # only stop bootstrap when terminated, don't stop when truncated
                    for idx, _need_final_obs in enumerate(need_final_obs):
                        if _need_final_obs:
                            for k in next_obs_seq.keys():
                                real_next_obs_seq[k][idx] = infos[
                                    "final_observation"][idx][
                                        k]  # info saves np object

                if args.critic_input == 'res' and args.actor_input == 'obs':
                    raise NotImplementedError(
                        'need to get obs embedding for real_next_obs here')
                else:  # sum or concat both need base actions for s and s'
                    base_next_act_seq_tensor, real_next_obs_seq_embedding_tensor = base_policy.get_eval_action(
                        to_tensor(real_next_obs_seq, device),
                        return_obs_embedding=True)
                    base_next_act_seq = base_next_act_seq_tensor.cpu().numpy()
                    real_next_obs_embedding = real_next_obs_seq_embedding_tensor[:,
                                                                                 -base_policy
                                                                                 .
                                                                                 obs_embedding_dim:].detach(
                                                                                 ).cpu(
                                                                                 ).numpy(
                                                                                 )  # most recent obs
                    base_next_actions = base_next_act_seq.reshape(
                        -1, total_act_dim)
                    actions_to_save = np.concatenate(
                        [res_actions, base_actions, base_next_actions], axis=1)

                obs_embedding = obs_embedding_tensor.cpu().numpy()
                rb.add(obs_embedding, real_next_obs_embedding, actions_to_save,
                       rewards, stop_bootstrap, infos)

                step_in_episodes += args.act_horizon
                step_in_episodes[terminations | truncations] = 0

                # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
                obs_seq = next_obs_seq

            timer.end('collect')

            # ALGO LOGIC: training.
            if rb.size() * rb.n_envs < args.learning_starts:
                continue

            learning_has_started = True
            for local_update in range(num_updates_per_training):
                global_update += 1
                data = rb.sample(args.batch_size)

                if args.critic_input != 'res' or args.actor_input == 'obs_base_action':
                    res_actions = data.actions[:, :total_act_dim]
                    base_actions = data.actions[:,
                                                total_act_dim:total_act_dim *
                                                2]
                    base_next_actions = data.actions[:, -total_act_dim:]
                else:
                    res_actions = data.actions

                #############################################
                # Train agent
                #############################################

                # update the value networks
                with torch.no_grad():
                    actor_input = data.next_observations if args.actor_input == 'obs' else torch.cat(
                        [data.next_observations, base_next_actions], dim=1)
                    next_state_res_actions, next_state_log_pi, _ = res_actor.get_action(
                        actor_input)
                    if args.critic_input == 'res':
                        next_state_actions = next_state_res_actions
                    elif args.critic_input == 'sum':
                        scaled_res_actions = args.res_scale * next_state_res_actions
                        next_state_actions = base_next_actions + scaled_res_actions
                    else:  # concat
                        next_state_actions = torch.cat(
                            [next_state_res_actions, base_next_actions], dim=1)
                    qf1_next_target = qf1_target(data.next_observations,
                                                 next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations,
                                                 next_state_actions)
                    min_qf_next_target = torch.min(
                        qf1_next_target,
                        qf2_next_target) - sac_alpha * next_state_log_pi
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()) * args.gamma * (
                            min_qf_next_target).view(-1)
                    # data.dones is "stop_bootstrap", which is computed earlier according to args.bootstrap_at_done

                if args.critic_input == 'res':
                    current_actions = res_actions
                elif args.critic_input == 'sum':
                    scaled_res_actions = args.res_scale * res_actions
                    current_actions = base_actions + scaled_res_actions
                else:  # concat
                    current_actions = torch.cat([res_actions, base_actions],
                                                dim=1)
                qf1_a_values = qf1(data.observations, current_actions).view(-1)
                qf2_a_values = qf2(data.observations, current_actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad()
                qf_loss.backward()
                qf1_grad_norm = nn.utils.clip_grad_norm_(
                    qf1.parameters(), args.max_grad_norm)
                qf2_grad_norm = nn.utils.clip_grad_norm_(
                    qf2.parameters(), args.max_grad_norm)
                q_optimizer.step()

                # update the policy network
                if global_update % args.policy_frequency == 0:  # TD 3 Delayed update support
                    actor_input = data.observations if args.actor_input == 'obs' else torch.cat(
                        [data.observations, base_actions], dim=1)
                    res_pi, log_pi, _ = res_actor.get_action(actor_input)
                    if args.critic_input == 'res':
                        pi = res_pi
                    elif args.critic_input == 'sum':
                        scaled_res_actions = args.res_scale * res_pi
                        pi = base_actions + scaled_res_actions
                    else:  # concat
                        pi = torch.cat([res_pi, base_actions], dim=1)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((sac_alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_grad_norm = nn.utils.clip_grad_norm_(
                        res_actor.parameters(), args.max_grad_norm)
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = res_actor.get_action(actor_input)
                        sac_alpha_loss = (-log_sac_alpha *
                                          (log_pi + target_entropy)).mean()
                        # log_sac_alpha has a legacy reason: https://github.com/rail-berkeley/softlearning/issues/136#issuecomment-619535356

                        a_optimizer.zero_grad()
                        sac_alpha_loss.backward()
                        a_optimizer.step()
                        sac_alpha = log_sac_alpha.exp().item()

                # update the target networks
                if global_update % args.target_network_frequency == 0:
                    for param, target_param in zip(qf1.parameters(),
                                                   qf1_target.parameters()):
                        target_param.data.copy_(args.tau * param.data +
                                                (1 - args.tau) *
                                                target_param.data)
                    for param, target_param in zip(qf2.parameters(),
                                                   qf2_target.parameters()):
                        target_param.data.copy_(args.tau * param.data +
                                                (1 - args.tau) *
                                                target_param.data)

            timer.end('train')

            # Log training-related data
            if (global_step - args.training_freq
                ) // args.log_freq < global_step // args.log_freq:
                if len(result['return']) > 0:
                    for k, v in result.items():
                        writer.add_scalar(f"train/{k}", np.mean(v),
                                          global_step)
                    result = defaultdict(list)
                writer.add_scalar("losses/qf1_values",
                                  qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values",
                                  qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(),
                                  global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(),
                                  global_step)
                writer.add_scalar("losses/qf_loss",
                                  qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(),
                                  global_step)
                writer.add_scalar("losses/sac_alpha", sac_alpha, global_step)
                writer.add_scalar("losses/qf1_grad_norm", qf1_grad_norm.item(),
                                  global_step)
                writer.add_scalar("losses/qf2_grad_norm", qf2_grad_norm.item(),
                                  global_step)
                writer.add_scalar("losses/actor_grad_norm",
                                  actor_grad_norm.item(), global_step)
                # print("SPS:", int(global_step / (time.time() - start_time)))
                timer.dump_to_writer(writer, global_step)
                if args.autotune:
                    writer.add_scalar("losses/sac_alpha_loss",
                                      sac_alpha_loss.item(), global_step)

            # Evaluation
            if (global_step - args.training_freq
                ) // args.eval_freq < global_step // args.eval_freq:
                result = evaluate(args.num_eval_episodes, base_policy,
                                  res_actor, eval_envs, device)
                for k, v in result.items():
                    writer.add_scalar(f"eval/{k}", np.mean(v), global_step)
                timer.end('eval')

            # Checkpoint
            if args.save_freq and ( global_step >= args.total_timesteps or \
                    (global_step - args.training_freq) // args.save_freq < global_step // args.save_freq):
                os.makedirs(f'{log_path}/checkpoints', exist_ok=True)
                torch.save(
                    {
                        'res_actor':
                        res_actor.state_dict(),
                        'qf1':
                        qf1_target.state_dict(),
                        'qf2':
                        qf2_target.state_dict(),
                        'log_sac_alpha':
                        log_sac_alpha
                        if args.autotune else np.log(args.sac_alpha),
                    }, f'{log_path}/checkpoints/{global_step}.pt')

        envs.close()
        writer.close()
