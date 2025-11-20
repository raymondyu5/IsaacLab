import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tqdm
from absl import app, flags
from flax.training import checkpoints

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer
from serl_launcher.utils.launcher import (
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.utils.train_utils import concat_batches

from serl_launcher.agents.continuous.drq import DrQAgent
from scripts.serl.agent.residual_drq_agent import ResidualDrQAgent
from serl_launcher.common.evaluation import evaluate
from serl_launcher.utils.timer_utils import Timer

import os
import shutil
from scripts.serl.utils.data_buffer import MemoryEfficientReplayBufferDataStore

from typing import Any, Dict, Optional

import copy


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


from scripts.serl.env.serl_wrapper import obs_keys
from scripts.serl.utils.additional import to_python_type

from jax import nn


def make_drq_agent(
    seed,
    sample_obs,
    sample_action,
    image_keys=("image", ),
    encoder_type="small",
    discount=0.96,
    policy_kwargs=None,
    critic_network_kwargs=None,
    policy_network_kwargs=None,
    critic_input="sum",
):
    # update drq

    # ---- default dicts ----
    default_policy_kwargs = {
        "tanh_squash_distribution": True,
        "std_parameterization": "exp",
        "std_min": 1e-5,
        "std_max": 5,
    }
    default_critic_network_kwargs = {
        "activations": nn.tanh,
        "use_layer_norm": True,
        "hidden_dims": [256, 256],
    }
    default_policy_network_kwargs = {
        "activations": nn.tanh,
        "use_layer_norm": True,
        "hidden_dims": [256, 256],
    }

    # ---- merge with user overrides ----
    merged_policy_kwargs = {**default_policy_kwargs, **(policy_kwargs or {})}
    merged_critic_network_kwargs = {
        **default_critic_network_kwargs,
        **(critic_network_kwargs or {})
    }
    merged_policy_network_kwargs = {
        **default_policy_network_kwargs,
        **(policy_network_kwargs or {})
    }

    agent_factory = ResidualDrQAgent

    agent = agent_factory.create_drq(
        jax.random.PRNGKey(seed),
        sample_obs,
        sample_action,
        encoder_type=encoder_type,
        use_proprio=True,
        image_keys=image_keys,
        policy_kwargs=merged_policy_kwargs,
        critic_network_kwargs=merged_critic_network_kwargs,
        policy_network_kwargs=merged_policy_network_kwargs,
        temperature_init=1e-2,
        discount=discount,
        backup_entropy=False,
        critic_ensemble_size=10,
        critic_subsample_size=2,
        critic_input=critic_input,
    )
    return agent


def learner(
    rng,
    agent,
    replay_buffer,
    FLAGS,
    demo_buffer: Optional[MemoryEfficientReplayBufferDataStore] = None,
):
    """
    The learner loop, which runs when "--learner" is set to True.
    """

    if FLAGS.checkpoint_path is not None:

        parent_dir = os.path.dirname(os.path.abspath(__file__))
        # Build the absolute checkpoint path
        checkpoint_path = os.path.join(parent_dir, FLAGS.checkpoint_path)
        if os.path.exists(checkpoint_path):
            shutil.rmtree(checkpoint_path)
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

    # 50/50 sampling from RLPD, half from demo and half from online experience if
    devices = jax.local_devices()
    # demo_buffer is provided
    sharding = jax.sharding.PositionalSharding(devices)
    if demo_buffer is None:
        single_buffer_batch_size = FLAGS.batch_size
        demo_iterator = None

    else:

        single_buffer_batch_size = int(FLAGS.batch_size * 0.5)
        demo_iterator = demo_buffer.get_iterator(
            sample_args={
                "batch_size": single_buffer_batch_size,
                "pack_obs_and_next_obs": True,
            },
            device=sharding.replicate(),
        )

    # create replay buffer iterator
    replay_iterator = replay_buffer.get_iterator(
        sample_args={
            "batch_size": single_buffer_batch_size,
            "pack_obs_and_next_obs": True,
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

        for critic_step in range(FLAGS.critic_actor_ratio - 1):

            with timer.context("sample_replay_buffer"):
                batch = next(replay_iterator)

                # we will concatenate the demo data with the online data
                # if demo_buffer is provided
                if demo_iterator is not None:
                    demo_batch = next(demo_iterator)

                    batch = concat_batches(batch, demo_batch, axis=0)
            with timer.context("train_critics"):
                agent, critics_info = agent.update_critics(batch, )

        with timer.context("train"):
            batch = next(replay_iterator)

            # we will concatenate the demo data with the online data
            # if demo_buffer is provided
            if demo_iterator is not None:
                demo_batch = next(demo_iterator)
                batch = concat_batches(batch, demo_batch, axis=0)
            agent, update_info = agent.update_high_utd(batch, utd_ratio=1)
            print("publishing network")
            server.publish_network(agent.state.params)

        if update_steps % FLAGS.log_period == 0 and wandb_logger:

            wandb_logger.log(to_python_type(update_info), step=update_steps)
            wandb_logger.log({"timer": timer.get_average_times()},
                             step=update_steps)

        if FLAGS.checkpoint_period and update_steps % FLAGS.checkpoint_period == 0:
            assert FLAGS.checkpoint_path is not None
            checkpoints.save_checkpoint(checkpoint_path,
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

        print("updating params")
        agent = agent.replace(state=agent.state.replace(params=params))

    client.recv_network_callback(update_params)

    obs, _ = env.reset()
    done = False
    print(list(env.unwrapped.observation_space["policy"].keys()))

    # training loop
    timer = Timer()
    running_return = 0.0
    for i in range(50):
        actions = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(actions)
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
            base_action = next_obs.pop("base_action", None)
            next_base_action = next_obs.pop("next_base_action", None)
            obs.pop("base_action", None)
            obs.pop("next_base_action", None)

            running_return += reward

            data_store.insert(
                dict(observations=copy.deepcopy(obs),
                     actions=copy.deepcopy(actions),
                     next_observations=copy.deepcopy(next_obs),
                     rewards=reward,
                     masks=1.0 - done,
                     dones=done or truncated,
                     base_actions=base_action,
                     next_base_actions=next_base_action))

            obs = copy.deepcopy(next_obs)
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


class DrqAgent:

    def __init__(self, env, FLAGS, env_config=None, use_diffusion_model=False):
        devices = jax.local_devices()
        self.replay_bufffer = None

        num_devices = len(devices)
        sharding = jax.sharding.PositionalSharding(devices)
        assert FLAGS.batch_size % num_devices == 0
        rng = jax.random.PRNGKey(FLAGS.seed)

        rng, sampling_rng = jax.random.split(rng)
        agent = make_drq_agent(seed=FLAGS.seed,
                               sample_obs=env.observation_space.sample(),
                               sample_action=env.action_space.sample(),
                               image_keys=["rgb"],
                               encoder_type=FLAGS.encoder_type,
                               policy_kwargs={
                                   "std_max": 0.5,
                               },
                               critic_input=FLAGS.critic_input)
        self.agent = jax.device_put(jax.tree_map(jnp.array, agent),
                                    sharding.replicate())
        self.FLAGS = FLAGS
        if FLAGS.learner:

            self.sampling_rng = jax.device_put(sampling_rng,
                                               device=sharding.replicate())
            self.replay_buffer = MemoryEfficientReplayBufferDataStore(
                env.observation_space,
                env.action_space,
                capacity=FLAGS.replay_buffer_capacity,
                image_keys=["rgb"],
                flush_every=FLAGS.flush_every,
                gemini_rewards=FLAGS.gemini_rewards,
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
                2000, latest_seq_id=self.FLAGS.latest_seq_id
            )  # the queue size on the actor

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
            step=self.FLAGS.eval_checkpoint_step,
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
                last_obs = next_obs

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
        if self.FLAGS.actor:
            app.run(self.main)
        else:
            self.main(None)
