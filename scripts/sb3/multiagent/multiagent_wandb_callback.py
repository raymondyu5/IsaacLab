import logging
import os
from typing import Optional, Callable, List

import numpy as np
import wandb
from wandb.sdk.lib import telemetry as wb_telemetry

from collections import deque

logger = logging.getLogger(__name__)
#=====================tactile===========================
import torch

import cv2

import matplotlib.pyplot as plt
import copy
from stable_baselines3.common.callbacks import BaseCallback

from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union


class KVWriter:
    """
    Key Value writer
    """

    def write(self,
              key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
              step: int = 0) -> None:
        """
        Write a dictionary to file

        :param key_values:
        :param key_excluded:
        :param step:
        """
        raise NotImplementedError

    def close(self) -> None:
        """
        Close owned resources
        """
        raise NotImplementedError


class WandbOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.

    :param folder: the folder to write the log to
    """

    def __init__(self):
        pass

    def write(self,
              key_values: Dict[str, Any],
              key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
              step: int = 0) -> None:

        write_dict = {}
        for (key, value), (_, excluded) in zip(sorted(key_values.items()),
                                               sorted(key_excluded.items())):

            if excluded is not None and "wandb" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                write_dict[key] = value

            if isinstance(value, torch.Tensor):
                write_dict[key] = value.item()

        # Flush the output to the file
        # wandb.log(write_dict, step=step)
        # import pdb
        # pdb.set_trace()

    def close(self) -> None:
        """
        closes the file
        """
        pass


import logging

logging.getLogger("wandb").setLevel(logging.ERROR)


class WandbCallback(BaseCallback):
    """Log SB3 experiments to Weights and Biases
        - Added model tracking and uploading
        - Added complete hyperparameters recording
        - Added gradient logging
        - Note that `wandb.init(...)` must be called before the WandbCallback can be used

    Args:
        verbose: The verbosity of sb3 output
        model_save_path: Path to the folder where the model will be saved, The default value is `None` so the model is not logged
        model_save_freq: Frequency to save the model
        gradient_save_freq: Frequency to log gradient. The default value is 0 so the gradients are not logged
    """

    def __init__(
        self,
        verbose: int = 0,
        model_save_path: str = None,
        model=None,
        model_save_freq: int = 100,
        eval_freq: Optional[int] = None,
        eval_env_fn: Optional[Callable] = None,
        eval_cam_names: Optional[List[str]] = None,
        viz_point_cloud=False,
        viz_pc_env=None,
        gradient_save_freq: int = 0,
        success_threshold=None,
        add_right_hand=None,
        add_left_hand=None,
        share_policy=False,
    ):
        super().__init__(verbose)
        if wandb.run is None:
            raise wandb.Error(
                "You must call wandb.init() before WandbCallback()")
        with wb_telemetry.context() as tel:
            tel.feature.sb3 = True
        self.model_save_freq = model_save_freq
        self.model_save_path = model_save_path

        self.policies = model.policies
        self.model = model
        self.share_policy = share_policy

        self.eval_freq = eval_freq
        self.eval_env_fn = eval_env_fn
        self.eval_cam_names = eval_cam_names
        self.viz_point_cloud = viz_point_cloud
        if self.viz_point_cloud:
            self.viz_pc_env = viz_pc_env
        self.success_threshold = success_threshold
        self.success_counter = {}
        self.add_right_hand = add_right_hand
        self.add_left_hand = add_left_hand

        self.handness_name = []
        self.num_agents = 0

        if self.add_left_hand:
            self.handness_name.append("left")
            self.num_agents += 1
        if self.add_right_hand:
            self.handness_name.append("right")
            self.num_agents += 1

        self.gradient_save_freq = gradient_save_freq

        # Create folder if needed
        if self.model_save_path is not None:
            os.makedirs(self.model_save_path, exist_ok=True)
        else:
            assert (
                self.model_save_freq == 0
            ), "to use the `model_save_freq` you have to set the `model_save_path` parameter"

        self.roll_out = 0

    def _init_callback(self) -> None:

        pass

    def eval_online(self):

        last_obs = self.eval_env_fn.reset()
        dones = [False]

        while not dones[0]:

            actions = []
            if self.share_policy:
                for agent_id in range(self.num_agents):

                    rollout_action, _ = self.policies[0].predict(
                        last_obs[:, agent_id], deterministic=True)

                    actions.append(np.expand_dims(rollout_action, 1))

            else:
                for agent_id, hand_policy in enumerate(self.policies):

                    rollout_action, _ = hand_policy.predict(last_obs[:,
                                                                     agent_id],
                                                            deterministic=True)

                    actions.append(np.expand_dims(rollout_action, 1))
            rollout_action = np.concatenate(actions, axis=1)
            self.eval_env_fn.step_async(
                torch.as_tensor(rollout_action).to("cuda:0"))
            rollout = self.eval_env_fn.step_wait()
            dones = rollout[-2][..., -1]

            last_obs = copy.deepcopy(rollout[0])
            if self.viz_point_cloud:
                pointcloud = self.viz_pc_env.render_pc()
                wandb.log({"point_cloud": wandb.Object3D(pointcloud)},
                          step=self.roll_out + 1)

        return last_obs

    def _on_rollout_end(self, save_path=False) -> None:
        if save_path:
            self.save_model()
            return

        need_restore = self.model.__dict__.get("need_restore", False)
        current_restore_step = self.model.__dict__.get("current_restore_step",
                                                       0)
        # wandb.log({"rollout/restore": current_restore_step},
        #           step=self.roll_out + 1)
        if need_restore and current_restore_step <= 5:
            return

        if self.model_save_freq > 0:
            if self.model_save_path is not None:
                if self.roll_out % self.model_save_freq == 0:
                    self.save_model()
        if self.roll_out % self.model_save_freq == 0:
            self.eval_online()

        self.current_restore_step = 0
        self.roll_out += 1

    def _on_training_end(self) -> None:
        if self.model_save_path is not None:
            self.save_model()

    def save_model(self) -> None:
        path = os.path.join(self.model_save_path, f"model_{self.roll_out}")

        for agent_id, hand_policy in enumerate(self.policies):
            hand_policy.policy.save(path +
                                    f"_{self.handness_name[agent_id]}.zip")
            # wandb.save(path, base_path=self.model_save_path)
        if self.verbose > 1:
            logger.info("Saving model checkpoint to " + path)

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True


def setup_wandb(parser_config, exp_name, tags=None, project="isaaclab"):
    run = wandb.init(
        project=project,
        name=exp_name,
        config=parser_config,
        monitor_gym=True,
        save_code=False,  # optional
        tags=tags,
        entity="entongsu",
        resume="allow")
    return run
