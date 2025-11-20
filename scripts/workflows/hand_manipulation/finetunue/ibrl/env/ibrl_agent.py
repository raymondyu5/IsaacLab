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

sys.path.append("submodule/ibrl")

import common_utils
from common_utils import ibrl_utils as utils

from rl.q_agent import QAgent, QAgentConfig
from rl import replay


@dataclass
class MainConfig(common_utils.RunConfig):
    seed: int = 1
    # env
    episode_length: int = 200

    # agent
    use_state: int = 0
    q_agent: QAgentConfig = field(default_factory=lambda: QAgentConfig())
    stddev_max: float = 1.0
    stddev_min: float = 0.1
    stddev_step: int = 500000
    nstep: int = 3
    discount: float = 0.99
    replay_buffer_size: int = 500
    batch_size: int = 256
    num_critic_update: int = 1
    update_freq: int = 2
    bc_policy: str = ""
    # rl with preload data
    mix_rl_rate: float = 1  # 1: only use rl, <1, mix in some bc data
    preload_num_data: int = 0
    preload_datapath: str = ""
    freeze_bc_replay: int = 1
    # pretrain rl policy with bc and finetune
    pretrain_only: int = 1
    pretrain_num_epoch: int = 0
    pretrain_epoch_len: int = 10000
    load_pretrained_agent: str = ""
    load_policy_only: int = 1
    add_bc_loss: int = 0
    # others
    env_reward_scale: float = 1
    num_warm_up_episode: int = 50
    num_eval_episode: int = 10
    save_per_success: int = -1
    mp_eval: int = 0  # eval with multiprocess
    num_train_step: int = 200000
    log_per_step: int = 5000
    # log
    save_dir: str = "logs/rl/run1"
    use_wb: int = 0

    def __post_init__(self):

        if self.pretrain_num_epoch > 0:
            assert self.preload_num_data > 0

        self.stddev_min = min(self.stddev_max, self.stddev_min)

        if self.preload_datapath:
            self.num_warm_up_episode += self.preload_num_data

    @property
    def stddev_schedule(self):
        return f"linear({self.stddev_max},{self.stddev_min},{self.stddev_step})"


class IBRLWorkspace:

    def __init__(self, cfg: MainConfig, from_main=True, bc_policy=None):
        self.work_dir = cfg.save_dir
        print(f"workspace: {self.work_dir}")

        if from_main:
            common_utils.set_all_seeds(cfg.seed)
            sys.stdout = common_utils.Logger(cfg.log_path,
                                             print_to_stdout=True)

            pyrallis.dump(cfg, open(cfg.cfg_path, "w"))  # type: ignore
            print(common_utils.wrap_ruler("config"))
            with open(cfg.cfg_path, "r") as f:
                print(f.read(), end="")
            print(common_utils.wrap_ruler(""))

        self.cfg = cfg
        self.cfg_dict = yaml.safe_load(open(cfg.cfg_path, "r"))

        self.global_step = 0
        self.global_episode = 0
        self.train_step = 0

        print(self.train_env.observation_shape)
        self.agent = QAgent(
            self.cfg.use_state,
            self.train_env.observation_shape,
            self.train_env.prop_shape,
            self.train_env.action_dim,
            self.cfg.rl_camera,
            cfg.q_agent,
        )

        if not from_main:
            return

        if cfg.load_pretrained_agent and cfg.load_pretrained_agent != "None":
            print(
                f"loading loading pretrained agent from {cfg.load_pretrained_agent}"
            )
            critic_states = copy.deepcopy(self.agent.critic.state_dict())
            self.agent.load_state_dict(torch.load(cfg.load_pretrained_agent))
            if cfg.load_policy_only:
                # avoid overwriting critic
                self.agent.critic.load_state_dict(critic_states)
                self.agent.critic_target.load_state_dict(critic_states)

        self.ref_agent = copy.deepcopy(self.agent)
        # override to always use RL even when self.agent is ibrl
        self.ref_agent.cfg.act_method = "rl"

        self._setup_replay()

    def _setup_replay(self):
        use_bc = False
        if self.cfg.mix_rl_rate < 1:
            use_bc = True
        if self.cfg.save_per_success > 0:
            use_bc = True
        if self.cfg.pretrain_num_epoch > 0 or self.cfg.add_bc_loss:
            assert self.cfg.preload_num_data
            use_bc = True

        self.replay = replay.ReplayBuffer(
            self.cfg.nstep,
            self.cfg.discount,
            frame_stack=1,
            max_episode_length=self.cfg.episode_length,
            replay_size=self.cfg.replay_buffer_size,
            use_bc=use_bc,
            save_per_success=self.cfg.save_per_success,
            save_dir=self.cfg.save_dir,
        )

    def eval(self, seed, policy) -> float:
        random_state = np.random.get_state()
        import pdb
        pdb.set_trace()

        np.random.set_state(random_state)
        return float(np.mean(scores))  # type: ignore

    def train(self):
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

        if self.replay.num_episode < self.cfg.num_warm_up_episode:
            self.warm_up()

        stopwatch = common_utils.Stopwatch()
        obs, _ = self.train_env.reset()
        self.replay.new_episode(obs)
        while self.global_step < self.cfg.num_train_step:
            ### act ###
            with stopwatch.time("act"), torch.no_grad(), utils.eval_mode(
                    self.agent):
                stddev = utils.schedule(self.cfg.stddev_schedule,
                                        self.global_step)
                action = self.agent.act(obs, eval_mode=False, stddev=stddev)
                stat["data/stddev"].append(stddev)

            ### env.step ###
            with stopwatch.time("env step"):
                obs, reward, terminal, success, image_obs = self.train_env.step(
                    action)

            with stopwatch.time("add"):
                assert isinstance(terminal, bool)
                reply = {"action": action}
                self.replay.add(obs, reply, reward, terminal, success,
                                image_obs)
                self.global_step += 1

            if terminal:
                with stopwatch.time("reset"):
                    self.global_episode += 1
                    stat["score/train_score"].append(float(success))
                    stat["data/episode_len"].append(self.train_env.time_step)

                    # reset env
                    obs, _ = self.train_env.reset()
                    self.replay.new_episode(obs)

            ### logging ###
            if self.global_step % self.cfg.log_per_step == 0:
                self.log_and_save(stopwatch, stat, saver)

            ### train ###
            if self.global_step % self.cfg.update_freq == 0:
                with stopwatch.time("train"):
                    self.rl_train(stat)
                    self.train_step += 1

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
        stat["other/replay"].append(self.replay.size())
        stat["score/num_success"].append(self.replay.num_success)

        if self.replay.bc_replay is not None:
            stat["data/bc_replay_size"].append(self.replay.size(bc=True))

        with stopwatch.time("eval"):
            eval_seed = (self.global_step //
                         self.cfg.log_per_step) * self.cfg.num_eval_episode
            stat["eval/seed"].append(eval_seed)
            eval_score = self.eval(seed=eval_seed, policy=self.agent)
            stat["score/score"].append(eval_score)

            original_act_method = self.agent.cfg.act_method
            # if self.agent.cfg.act_method != "rl":
            #     with self.agent.override_act_method("rl"):
            #         rl_score = self.eval(seed=eval_seed, policy=self.agent)
            #         stat["score/score_rl"].append(rl_score)
            #         stat["score_diff/hybrid-rl"].append(eval_score - rl_score)

            if self.agent.cfg.act_method == "ibrl_soft":
                with self.agent.override_act_method("ibrl"):
                    greedy_score = self.eval(seed=eval_seed, policy=self.agent)
                    stat["score/greedy_score"].append(greedy_score)
                    stat["score_diff/greedy-soft"].append(greedy_score -
                                                          eval_score)
            assert self.agent.cfg.act_method == original_act_method

        saved = saver.save(self.agent.state_dict(),
                           eval_score,
                           save_latest=True)
        stat.summary(self.global_step, reset=True)
        print(f"saved?: {saved}")
        stopwatch.summary(reset=True)
        print("total time:", common_utils.sec2str(stopwatch.total_time))
        print(common_utils.get_mem_usage())

    def rl_train(self, stat: common_utils.MultiCounter):
        stddev = utils.schedule(self.cfg.stddev_schedule, self.global_step)
        for i in range(self.cfg.num_critic_update):
            if self.cfg.mix_rl_rate < 1:
                rl_bsize = int(self.cfg.batch_size * self.cfg.mix_rl_rate)
                bc_bsize = self.cfg.batch_size - rl_bsize
                batch = self.replay.sample_rl_bc(rl_bsize, bc_bsize, "cuda:0")
            else:
                batch = self.replay.sample(self.cfg.batch_size, "cuda:0")

            # in RED-Q, only update actor once
            update_actor = i == self.cfg.num_critic_update - 1

            bc_batch = None
            if update_actor and self.cfg.add_bc_loss:
                bc_batch = self.replay.sample_bc(self.cfg.batch_size, "cuda:0")

            metrics = self.agent.update(batch, stddev, update_actor, bc_batch,
                                        self.ref_agent)

            stat.append(metrics)
            stat["data/discount"].append(batch.bootstrap.mean().item())

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
                batch = self.replay.sample_bc(self.cfg.batch_size, "cuda")
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
