from scripts.rsrl.agent.agentlace_trainer import TrainerServer, TrainerClient, TrainerSMInterface, make_trainer_config
import time
from scripts.rsrl.utils.sb3_datastore import QueuedDataStore
from scripts.rsrl.utils.sb3_utils import init_algo_from_dict, serialize_space
import yaml
from box import Box
import copy

import numpy as np


class AsynEnvWrapper:

    def __init__(self, agent_env, args_cli=None, steps_per_update=30):
        self.agent_env = agent_env
        self.args_cli = args_cli

        cfg = make_trainer_config(port_number=5555, broadcast_port=5556)
        if "send_dict" not in cfg.request_types:
            cfg.request_types.append("send_dict")

        self.data_store = QueuedDataStore(
            2000, latest_seq_id=0)  # the queue size on the actor
        self.client = TrainerClient(
            name="actor_env",
            server_ip="127.0.0.1",
            config=cfg,
            data_store=self.data_store,
            wait_for_server=True,
        )

        obs_space_dict = serialize_space(self.agent_env.observation_space)
        act_space_dict = serialize_space(self.agent_env.action_space)

        with open(self.args_cli.rl_config, "r", encoding="utf-8") as file:
            yaml_data = yaml.safe_load(file)
        agent_cfg = Box(yaml_data)

        agent_cfg.seed = agent_cfg["seed"]
        # from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import process_sb3_cfg

        # agent_cfg = process_sb3_cfg(agent_cfg)
        cfg_dict = agent_cfg.to_dict()

        self.cfg_with_spaces = {
            **cfg_dict,
            "save_dir": args_cli.log_dir,
            "model_save_freq": 5,
            "observation_space": obs_space_dict,
            "action_space": act_space_dict,
        }
        while True:
            res = self.client.request("init_rl", self.cfg_with_spaces)
            if res["success"]:
                print("Server response:", res)
                break
        self.algo = init_algo_from_dict(self.cfg_with_spaces)
        self.steps_per_update = steps_per_update
        self.client.recv_network_callback(self.update_params)
        self.max_residual_schedule = self.cfg_with_spaces.get(
            "max_residual_schedule", 100000)
        self.use_residual_schedule = self.cfg_with_spaces.get(
            "use_residual_schedule", False)
        self.num_rollouts = 0

        self.actor()

    def update_residual_schedule(self) -> None:
        """
        Progressive residual exploration with step schedule.
        """
        if self.algo.iteractions > self.max_residual_schedule:
            self.epsilon_residual = 1.0
        else:
            # Compute step index

            self.epsilon_residual = self.num_rollouts / self.max_residual_schedule

        # Clip just in case
        self.epsilon_residual = min(1.0, self.epsilon_residual)

        # Optional: log
        self.algo.logger.record("exploration/epsilon_residual",
                                self.epsilon_residual)

    def update_params(self, params):
        # print("[INFO] Updating policy parameters from the learner.")

        self.algo.policy.load_state_dict(params)

    def before_step(self):
        learning_start = self.cfg_with_spaces.get("learning_starts", 1000)

        self.last_obs = self.agent_env.reset()
        while learning_start > self.data_store.latest_data_id():

            if self.use_residual_schedule:
                self.update_residual_schedule()
            random_action = self.agent_env.action_space.sample()

            if self.use_residual_schedule:
                random_seed = np.random.rand(1)
                disable_res_masks = random_seed > self.epsilon_residual

                random_action[disable_res_masks] = 0

            self.agent_env.step_async(random_action)
            next_obs, reward, done, info = self.agent_env.step_wait()
            self.data_store.insert(
                dict(obs=self.last_obs,
                     action=random_action,
                     next_obs=next_obs,
                     reward=reward,
                     done=done,
                     infos=info,
                     latest_data_id=self.data_store.latest_data_id()))
            self.last_obs = copy.deepcopy(next_obs)
            self.num_rollouts += 1

            self.client.update()

    def after_step(self):
        step_count = 0

        while True:
            if self.use_residual_schedule:
                self.update_residual_schedule()

            unscaled_action, _ = self.algo.predict(self.last_obs,
                                                   deterministic=False)
            scaled_action = self.algo.policy.scale_action(unscaled_action)
            buffer_action = scaled_action
            action = self.algo.policy.unscale_action(scaled_action)

            if self.use_residual_schedule:
                random_seed = np.random.rand(1)
                disable_res_masks = random_seed > self.epsilon_residual

                buffer_action[disable_res_masks] = 0
                action[disable_res_masks] = 0

            self.agent_env.step_async(action)
            next_obs, reward, done, info = self.agent_env.step_wait()
            self.data_store.insert(
                dict(obs=self.last_obs,
                     action=buffer_action,
                     next_obs=next_obs,
                     reward=reward,
                     done=done,
                     infos=info,
                     latest_data_id=self.data_store.latest_data_id()))
            self.client.update()
            self.last_obs = copy.deepcopy(next_obs)
            step_count += 1
            if step_count % self.steps_per_update == 0:
                self.client.update()
            self.num_rollouts += 1
            # print(f"[DEBUG] Step {step_count}, data id {self.data_store.latest_data_id()}"
            #       f", action {action}, reward {reward}, done {done}")

    def actor(self):

        self.before_step()

        print("[INFO] Starting the actor loop.")

        self.after_step()
