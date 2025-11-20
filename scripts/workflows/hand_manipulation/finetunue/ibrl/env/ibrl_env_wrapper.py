import sys
import os
import sys
from dataclasses import dataclass, field
import yaml
import copy
from typing import Optional
import pyrallis
import torch
import numpy as np

from scripts.workflows.open_policy.utils.buffer_utils import reset_buffer, update_buffer, filter_out_data
import sys

sys.path.append("submodule/stable-baselines3")

sys.path.append("submodule/ibrl")

import common_utils
from common_utils import ibrl_utils
from rl.q_agent import QAgent, QAgentConfig, update_dataclass
# from rl import replay
from common_utils import ibrl_utils as utils

from scripts.workflows.hand_manipulation.finetunue.ibrl.env.ibrl_agent import MainConfig
from scripts.workflows.hand_manipulation.finetunue.ibrl.env.ibrl_buffer_wrapper import IBRLBufferWrapper
import copy
import time


class IBRLEnvWrapper(IBRLBufferWrapper):

    def __init__(self,
                 env,
                 env_config,
                 args_cli,
                 use_relative_pose=False,
                 use_joint_pose=False,
                 eval_mode=False,
                 replay_mode=False,
                 collect_mode=False,
                 use_visal_obs=False,
                 obs_keys=[
                     'right_hand_joint_pos', 'right_ee_pose',
                     'right_target_object_pose', 'right_contact_obs',
                     'right_object_in_tip', 'right_manipulated_object_pose'
                 ]):

        MainConfig.save_dir = args_cli.diffusion_path + "ibrl"
        self.use_visal_obs = use_visal_obs

        self.cfg_dict = yaml.safe_load(
            open("source/config/task/hand_env/ibrl/rlpd_ycb.yaml", "r"))
        cfg = MainConfig(**self.cfg_dict)

        cfg.q_agent = QAgentConfig()
        self.train_step = 0
        self.global_episode = 0

        self.cfg = cfg
        super().__init__(env,
                         env_config,
                         args_cli,
                         use_relative_pose=use_relative_pose,
                         use_joint_pose=use_joint_pose,
                         eval_mode=eval_mode,
                         replay_mode=replay_mode,
                         collect_mode=collect_mode,
                         cfg=cfg,
                         obs_keys=obs_keys)

        self._setup_replay()

        update_dataclass(cfg.q_agent, self.cfg_dict.get("q_agent", {}))

        self.work_dir = cfg.save_dir
        print(f"workspace: {self.work_dir}")
        self.init_agent()

    def init_agent(self):

        if self.cfg.use_state == 1:
            obs_dim = 0
            for key, value in self.env.unwrapped.observation_space.spaces[
                    "policy"].items():

                obs_dim += value.shape[-1]

        self.agent = QAgent(
            self.cfg.use_state,
            [obs_dim],
            [0],
            self.env.action_space.shape[-1],
            None,
            self.cfg.q_agent,
        )

        if self.cfg.load_pretrained_agent and self.cfg.load_pretrained_agent != "None":
            print(
                f"loading loading pretrained agent from {self.cfg.load_pretrained_agent}"
            )
            critic_states = copy.deepcopy(self.agent.critic.state_dict())
            self.agent.load_state_dict(
                torch.load(self.cfg.load_pretrained_agent))
            if self.cfg.load_policy_only:
                # avoid overwriting critic
                self.agent.critic.load_state_dict(critic_states)
                self.agent.critic_target.load_state_dict(critic_states)

        self.ref_agent = copy.deepcopy(self.agent)
        # override to always use RL even when self.agent is ibrl
        self.ref_agent.cfg.act_method = "rl"

    def pretrain_policy(self):
        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        for epoch in range(self.cfg.pretrain_num_epoch):
            for _ in range(self.cfg.pretrain_epoch_len):
                batch = self.replay_buffer.sample_bc(self.cfg.batch_size,
                                                     "cuda")
                metrics = self.agent.pretrain_actor_with_bc(batch)

                for k, v in metrics.items():
                    stat[k].append(v)

            eval_seed = epoch * self.cfg.pretrain_epoch_len
            score = self.eval(eval_seed, policy=self.agent)
            stat["pretrain/score"].append(score)

            stat.summary(epoch, reset=True)
            saved = saver.save(self.agent.state_dict(),
                               score,
                               save_latest=True)
            print(f"saved?: {saved}")
            print(common_utils.get_mem_usage())

    def rl_train(self, stat: common_utils.MultiCounter):
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for i in range(self.cfg.num_critic_update):

            self.replay_buffer
            if self.cfg.mix_rl_rate < 1:
                rl_bsize = int(self.cfg.batch_size * self.cfg.mix_rl_rate)
                bc_bsize = self.cfg.batch_size - rl_bsize
                batch = self.replay_buffer.sample(rl_bsize, bc_bsize,
                                                  self.env.device)
            else:
                batch = self.replay_buffer.sample(self.cfg.batch_size, None,
                                                  self.env.device)

            # in RED-Q, only update actor once
            update_actor = i == self.cfg.num_critic_update - 1

            bc_batch = None
            if update_actor and self.cfg.add_bc_loss:
                bc_batch = self.replay_buffer.sample(None, self.cfg.batch_size,
                                                     "cuda:0")

            metrics = self.agent.update(batch, stddev, update_actor, bc_batch,
                                        self.ref_agent)

            stat.append(metrics)
            stat["data/discount"].append(batch.bootstrap.mean().item())

    def log_and_save(
        self,
        stopwatch: common_utils.Stopwatch,
        stat: common_utils.MultiCounter,
        saver: common_utils.TopkSaver,
    ):
        elapsed_time = stopwatch.elapsed_time_since_reset
        stat["other/speed"].append(self.cfg.log_per_step / elapsed_time)
        stat["other/elapsed_time"].append(elapsed_time)
        stat["other/episode"].append(self.global_episode)
        stat["other/step"].append(self.global_step)
        stat["other/train_step"].append(self.train_step)
        # stat["other/replay"].append(self.replay.size())
        # stat["score/num_success"].append(self.replay.num_success)

        # if self.replay.bc_replay is not None:
        #     stat["data/bc_replay_size"].append(self.replay.size(bc=True))

        # with stopwatch.time("eval"):
        #     eval_seed = (self.global_step //
        #                  self.cfg.log_per_step) * self.cfg.num_eval_episode
        #     stat["eval/seed"].append(eval_seed)
        #     eval_score = self.eval(seed=eval_seed, policy=self.agent)
        #     stat["score/score"].append(eval_score)

        #     original_act_method = self.agent.cfg.act_method
        #     # if self.agent.cfg.act_method != "rl":
        #     #     with self.agent.override_act_method("rl"):
        #     #         rl_score = self.eval(seed=eval_seed, policy=self.agent)
        #     #         stat["score/score_rl"].append(rl_score)
        #     #         stat["score_diff/hybrid-rl"].append(eval_score - rl_score)

        #     if self.agent.cfg.act_method == "ibrl_soft":
        #         with self.agent.override_act_method("ibrl"):
        #             greedy_score = self.eval(seed=eval_seed, policy=self.agent)
        #             stat["score/greedy_score"].append(greedy_score)
        #             stat["score_diff/greedy-soft"].append(greedy_score -
        #                                                   eval_score)
        #     assert self.agent.cfg.act_method == original_act_method

        # saved = saver.save(self.agent.state_dict(),
        #                    eval_score,
        #                    save_latest=True)
        stat.summary(self.global_step, reset=True)
        # print(f"saved?: {saved}")
        stopwatch.summary(reset=True)
        print("total time:", common_utils.sec2str(stopwatch.total_time))
        print(common_utils.get_mem_usage())

    def train_agent(self):
        stat = common_utils.MultiCounter(
            self.work_dir,
            bool(self.cfg.use_wb),
            wb_exp_name=self.cfg.wb_exp,
            wb_run_name=self.cfg.wb_run,
            wb_group_name=self.cfg.wb_group,
            config=self.cfg_dict,
        )
        self.agent.set_stats(stat)
        saver = common_utils.TopkSaver(save_dir=self.work_dir, topk=1)

        stopwatch = common_utils.Stopwatch()
        self.global_step = 0
        last_obs, _ = self.reset()

        while self.global_step < self.cfg.num_train_step:
            ### act ###
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(
                    self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule,
                                        self.global_step)
                last_state = self.process_obs(env_id=None,
                                              per_obs=last_obs["policy"])

                action = self.agent.act({"state": last_state},
                                        eval_mode=False,
                                        stddev=stddev)
                stat["data/stddev"].append(stddev)

            ### env.step ###
            with stopwatch.time("env step"):

                next_obs, rewards, terminated, time_outs, extras, diffused_actions, diffusion_obs = self.step(
                    torch.as_tensor(action).to(self.env.device))

            with stopwatch.time("add"):
                next_state = self.process_obs(env_id=None,
                                              per_obs=next_obs["policy"])

                done = terminated | time_outs
                self.replay_buffer.rl_replay_buffer.add(
                    obs=last_state.cpu().numpy(),
                    next_obs=next_state.cpu().numpy(),
                    action=action.cpu().numpy(),
                    reward=rewards.cpu().numpy(),
                    done=done.cpu().numpy(),
                    infos=[[] for i in range(action.shape[0])])

                self.global_step += 1
                last_obs = copy.deepcopy(next_obs)

            if done[0]:
                with stopwatch.time("reset"):
                    self.global_episode += 1

                    stat["score/train_score"].append(
                        float(
                            self.eval_success(next_obs).sum().item() /
                            self.env.num_envs))
                    stat["data/episode_len"].append(
                        self.env.unwrapped.max_episode_length)

                    # reset env
                    last_obs, _ = self.reset()

            ### logging ###
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, stat, saver)

            ### train ###
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self.rl_train(stat)
                    self.train_step += 1

    def warm_up(self):
        reset_buffer(self)
        last_obs, _ = self.reset()
        start_time = time.time()

        for _ in range(self.cfg.episode_length):
            action = torch.randn(
                (self.env.action_space.shape),
                device=self.env.device,
            )

            next_obs, rewards, terminated, truncated, extras, diffused_action, diffusion_obs = self.step(
                action)

            done = terminated | truncated
            update_buffer(
                self,
                next_obs,
                last_obs,
                action,
                rewards,
                terminated,
                done,
            )
            last_obs = copy.deepcopy(next_obs)

        success_index = self.eval_success(next_obs)
        print(
            f"warm up success rate: {success_index.sum().item()/self.env.num_envs }",
            "time:",
            time.time() - start_time)

        obs_buffer, next_obs_buffer, action_buffer, rewards_buffer, does_buffer = filter_out_data(
            self,
            torch.nonzero(success_index, as_tuple=False),
            save_data=False)
        return self.push_to_bc_buffer(
            obs_buffer,
            next_obs_buffer,
            action_buffer,
            rewards_buffer,
            does_buffer,
            success_index.sum(),
        )

    def train(self):
        stop = False
        while not stop:
            stop = self.warm_up()

        self.train_agent()
