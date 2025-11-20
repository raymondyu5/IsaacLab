import time
from functools import partial

import gym
import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from serl_launcher.utils.launcher import (
    make_sac_agent,
    make_trainer_config,
    make_wandb_logger,
    make_replay_buffer,
)
from serl_launcher.utils.train_utils import concat_batches

from scripts.serl.utils.record_episode_statistics import RecordEpisodeStatistics
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer

import os
import shutil
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore

from typing import Any, Dict, Optional


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


from scripts.serl.env.serl_wrapper import obs_keys
from scripts.serl.utils.additional import to_python_type


def learner(
    rng,
    agent: SACAgent,
    replay_buffer,
    FLAGS,
    demo_buffer: Optional[MemoryEfficientReplayBufferDataStore] = None,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """
    if os.path.exists(FLAGS.checkpoint_path):
        shutil.rmtree(FLAGS.checkpoint_path)
    # set up wandb and logging
    wandb_logger = make_wandb_logger(
        project="serl_dev",
        description=FLAGS.exp_name or FLAGS.env,
        debug=FLAGS.debug,
    )

    # To track the step in the training loop
    update_steps = 0

    def stats_callback(type: str, payload: dict) -> dict:
        """Callback for when server receives stats request."""
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(payload, step=update_steps)
        return {}  # not expecting a response

    # Create server
    server = TrainerServer(make_trainer_config(),
                           request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.start(threaded=True)

    # Loop to wait until replay_buffer is filled
    pbar = tqdm.tqdm(
        total=FLAGS.training_starts,
        initial=len(replay_buffer),
        desc="Filling up replay buffer",
        position=0,
        leave=True,
    )
    while len(replay_buffer) < FLAGS.training_starts:
        pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
        time.sleep(1)
    pbar.update(len(replay_buffer) - pbar.n)  # Update progress bar
    pbar.close()

    # send the initial network to the actor
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    train_critic_networks_to_update = frozenset({"critic"})
    train_networks_to_update = frozenset({"critic", "actor", "temperature"})

    # 50/50 sampling from RLPD, half from demo and half from online experience if
    # demo_buffer is provided
    if demo_buffer is None:
        single_buffer_batch_size = FLAGS.batch_size
        demo_iterator = None
    else:
        devices = jax.local_devices()

        sharding = jax.sharding.PositionalSharding(devices)
        single_buffer_batch_size = int(FLAGS.batch_size * 0.9)
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": single_buffer_batch_size,
            },
            device=sharding.replicate(),
        )

    # create replay buffer iterator
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
        },
        device=sharding.replicate(),
    )

    # wait till the replay buffer is filled with enough data
    timer = Timer()

    # show replay buffer progress bar during training
    pbar = tqdm.tqdm(
        total=FLAGS.replay_buffer_capacity,
        initial=len(replay_buffer),
        desc="replay buffer",
    )

    for step in tqdm.tqdm(range(FLAGS.max_steps),
                          dynamic_ncols=True,
                          desc="learner"):
        # run n-1 critic updates and 1 critic + actor update.
        # This makes training on GPU faster by reducing the large batch transfer time from CPU to GPU
        # Train the networks
        for critic_step in range(FLAGS.cta_ratio - 1):
            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)

            with timer.context("train_critics"):
                agent, critics_info = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        with timer.context("train"):
            batch = next(replay_iterator)
            demo_batch = next(demo_iterator)
            batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

        if update_steps % FLAGS.log_period == 0 and wandb_logger:
            wandb_logger.log(to_python_type(update_info), step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()},
                             step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(FLAGS.checkpoint_path,
                                        agent.state,
                                        step=update_steps,
                                        keep=20)

        pbar.update(len(replay_buffer) - pbar.n)  # update replay buffer bar
        update_steps += 1


def actor(env, agent, data_store, sampling_rng, FLAGS):
    """
    This is the actor loop, which runs when "--actor" is set to True.
    """

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_store,
        wait_for_server=True,
    )

    # Function to update the agent with new params
    def update_params(params):
        nonlocal agent
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    obs, _ = env.reset()
    done = False
    print(list(env.unwrapped.observation_space["policy"].keys()))

    # if FLAGS.demo_path is not None:

    #     load_demo_data(FLAGS, data_store,
    #                    list(env.unwrapped.observation_space["policy"].keys()))

    # training loop
    timer = Timer()
    running_return = 0.0
    for i in range(50):
        actions = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(actions)
    env.reset()
    print("[INFO] Start actor loop.")
    for step in tqdm.tqdm(range(FLAGS.max_steps), dynamic_ncols=True):
        timer.tick("total")

        with timer.context("sample_actions"):
            if step < FLAGS.random_steps:
                actions = env.action_space.sample()
            else:
                sampling_rng, key = jax.random.split(sampling_rng)
                actions = agent.sample_actions(
                    observations=jax.device_put(obs),
                    seed=key,
                    deterministic=False,
                )
                actions = np.asarray(jax.device_get(actions))

        # Step environment
        with timer.context("step_env"):

            next_obs, reward, done, truncated, info = env.step(actions)
            next_obs = np.asarray(next_obs, dtype=np.float32)
            reward = np.asarray(reward, dtype=np.float32)

            running_return += reward

            data_store.insert(
                dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - done,
                    dones=done or truncated,
                ))

            obs = next_obs
            if done or truncated:

                running_return = 0.0
                obs, _ = env.reset()

        if FLAGS.render:
            env.render()

        if step % FLAGS.steps_per_update == 0:
            client.update()

        # if step % FLAGS.eval_period == 0:
        #     with timer.context("eval"):
        #         eval_env = RecordEpisodeStatistics(env)
        #         evaluate_info = evaluate(
        #             policy_fn=partial(agent.sample_actions, argmax=True),
        #             env=eval_env,
        #             num_episodes=FLAGS.eval_n_trajs,
        #         )
        #     stats = {"eval": evaluate_info}
        #     client.request("send-stats", stats)
        # import pdb
        # pdb.set_trace()

        timer.tock("total")

        if step % FLAGS.log_period == 0:
            stats = {"timer": timer.get_average_times()}
            client.request("send-stats", stats)


class SerlRLPDAgent:

    def __init__(self, env, FLAGS, env_config=None, use_diffusion_model=False):
        devices = jax.local_devices()
        self.replay_bufffer = None

        num_devices = len(devices)
        sharding = jax.sharding.PositionalSharding(devices)
        assert FLAGS.batch_size % num_devices == 0
        rng = jax.random.PRNGKey(FLAGS.seed)

        rng, sampling_rng = jax.random.split(rng)
        agent: SACAgent = make_sac_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
        )
        self.agent: SACAgent = jax.device_put(jax.tree_map(jnp.array, agent),
                                              sharding.replicate())
        self.FLAGS = FLAGS
        if FLAGS.learner:

            self.sampling_rng = jax.device_put(sampling_rng,
                                               device=sharding.replicate())
            self.replay_buffer = make_replay_buffer(
                env,
                capacity=FLAGS.replay_buffer_capacity,
                rlds_logger_path=FLAGS.log_rlds_path,
                type="replay_buffer",
                preload_rlds_path=FLAGS.preload_rlds_path,
            )
            self.replay_iterator = self.replay_buffer.get_iterator(
                sample_args={
                    "batch_size": FLAGS.batch_size * FLAGS.critic_actor_ratio,
                },
                device=sharding.replicate(),
            )
            # learner loop
            print_green("starting learner loop")

            if self.FLAGS.demo_path is not None:
                from scripts.serl.utils.demo_utils import make_zarr_replay_buffer
                self.replay_bufffer = make_zarr_replay_buffer(
                    FLAGS.demo_path,
                    obs_keys,
                    action_range=env_config["params"]["Task"]["action_range"],
                    use_diffusion_model=use_diffusion_model)[0]

            self.step_function = learner

        elif FLAGS.actor:
            self.env = env
            self.sampling_rng = jax.device_put(sampling_rng,
                                               sharding.replicate())
            self.data_store = QueuedDataStore(
                2000)  # the queue size on the actor

            # actor loop
            print_green("starting actor loop")
            self.step_function = actor
        else:
            self.sampling_rng = jax.device_put(sampling_rng,
                                               sharding.replicate())
            self.env = env

    def evaluate(self, ):

        ckpt = checkpoints.restore_checkpoint(
            self.FLAGS.checkpoint_path,
            self.agent.state,
            step=220000,
        )
        self.agent = self.agent.replace(state=ckpt)
        while True:
            last_obs, _ = self.env.reset()
            running_return = 0.0

            for i in range(120):

                actions = self.agent.sample_actions(
                    observations=jax.device_put(last_obs),
                    argmax=True,
                )
                actions = np.asarray(jax.device_get(actions))
                next_obs, reward, done, truncated, info = self.env.step(
                    actions)
                next_obs = np.asarray(next_obs, dtype=np.float32)
                reward = np.asarray(reward, dtype=np.float32)

                running_return += reward

    def main(self, _):
        if self.FLAGS.learner:

            self.step_function(
                self.sampling_rng,
                self.agent,
                self.replay_buffer,
                FLAGS=self.FLAGS,
                demo_buffer=self.replay_bufffer,
            )
        else:
            self.step_function(self.env, self.agent, self.data_store,
                               self.sampling_rng, self.FLAGS)

    def run_agent(self):
        app.run(self.main)
