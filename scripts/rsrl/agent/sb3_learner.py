#!/usr/bin/env python3
import logging
from typing import Any, Dict, Optional
import os
import threading
import time

from scripts.rsrl.agent.asyn_residual_sac import AsynResidualSAC
from scripts.rsrl.agent.agentlace_trainer import TrainerServer, make_trainer_config
from scripts.rsrl.utils.sb3_utils import init_algo_from_dict
from scripts.sb3.wandb_callback import setup_wandb, WandbCallback
from agentlace.data.data_store import QueuedDataStore
import copy


class Sb3Learner:

    def __init__(self, log_level: int = logging.INFO):
        self.log_level = log_level
        self.server: Optional[TrainerServer] = None
        self.algo: Optional[AsynResidualSAC] = None
        self._last_dict: Optional[Dict[str, Any]] = None
        self._config_ready = threading.Event()  # wait until config arrives

    # -------------------------------------------------------------

    def _request_callback(self, rtype: str,
                          payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle requests from client → server."""
        print(f"Request received: {rtype}")

        if rtype == "init_rl":

            cfg_dict = dict(payload)  # copy so we can pop
            self.rl_cfg = cfg_dict
            self._last_dict = cfg_dict
            self.train_cfg = {
                "n_timesteps": payload.pop("n_timesteps"),
                "rollout_id": payload.pop("rollout_id", 0),
                "save_dir": payload.pop("save_dir", "logs"),
                "model_save_freq": payload.pop("model_save_freq", 10000),
            }

            self.algo = init_algo_from_dict(payload)
            self._config_ready.set()  # unblock main

            return {"success": True, "message": "AsynResidualSAC initialized"}

        return {"success": False, "message": f"Unknown request type: {rtype}"}

    def _start_server(self):
        cfg = make_trainer_config()
        for rt in ["send_dict", "init_rl"]:
            if rt not in cfg.request_types:
                cfg.request_types.append(rt)

        self.server = TrainerServer(
            cfg,
            log_level=logging.INFO,
            request_callback=self._request_callback,
        )

        # run server in background
        # threading.Thread(target=self.server.start,
        #                  kwargs={
        #                      "threaded": True
        #                  },
        #                  daemon=True).start()
        self.server.start(threaded=True)

        # wait until we get the config
        logging.info("Waiting for config (send_dict) from client...")
        self._config_ready.wait()
        logging.info("Config received, starting training.")

        self.server.register_data_store("actor_env", self.algo.replay_buffer)

        setup_wandb(self._last_dict,
                    "real_time_rl",
                    tags=None,
                    project="real_rl")

        total_timesteps = self.train_cfg["n_timesteps"]
        callback = WandbCallback(
            model_save_freq=self.train_cfg["model_save_freq"],
            video_folder=os.path.join(self.train_cfg["save_dir"], "videos",
                                      "train"),
            model_save_path=str(self.train_cfg["save_dir"] + f"/residual_sac"),
            eval_env_fn=None,
            eval_freq=10,
            eval_cam_names=None,
            viz_point_cloud=False,
            viz_pc_env=None,
            rollout_id=self.train_cfg["rollout_id"],
        )

        total_timesteps, callback = self.algo._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps=True,
            tb_log_name="SAC",
            progress_bar=False,
        )

        callback.on_training_start(locals(), globals())

        if self.algo.use_residual_schedule:
            self.algo.update_residual_schedule()

        while self.algo.num_timesteps < total_timesteps:

            if self.algo.replay_buffer.latest_data_id < self.algo.learning_starts:
                last_data_id = self.algo.replay_buffer.latest_data_id

                continue

            gradient_steps = self.algo.gradient_steps

            # if last_data_id < self.algo.replay_buffer.latest_data_id:  #update only if new data
            #     print(self.algo.num_timesteps,
            #           self.algo.replay_buffer.latest_data_id)
            print(last_data_id)

            self.algo.train(batch_size=self.algo.batch_size,
                            gradient_steps=gradient_steps)
            self.server.publish_network(self.algo.policy.state_dict())
            last_data_id = copy.deepcopy(
                self.algo.replay_buffer.latest_data_id)
            # self.algo.num_timesteps + 1

        callback.on_training_end()

        # self.algo.learn(
        #     total_timesteps=self.train_cfg["n_timesteps"],
        #     iteration=self.train_cfg["rollout_id"],
        #     callback=WandbCallback(
        #         model_save_freq=self.train_cfg["model_save_freq"],
        #         video_folder=os.path.join(self.train_cfg["save_dir"], "videos",
        #                                   "train"),
        #         model_save_path=str(self.train_cfg["save_dir"] +
        #                             f"/residual_sac"),
        #         eval_env_fn=None,
        #         eval_freq=1000,
        #         eval_cam_names=None,
        #         viz_point_cloud=False,
        #         viz_pc_env=None,
        #         rollout_id=self.train_cfg["rollout_id"],
        #     ),
        # )

    def main(self):
        self._start_server()


if __name__ == "__main__":
    learner = Sb3Learner(log_level=logging.INFO)
    learner.main()
