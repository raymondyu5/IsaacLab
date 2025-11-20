import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn  # noqa: F401
from typing import Any

from stable_baselines3.common.utils import constant_fn
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvObs, VecEnvStepReturn
from scripts.workflows.hand_manipulation.env.rl_env.torch_layers import PointNetStateExtractor, ImageStateExtractor
from scripts.rsrl.agent.asyn_residual_sac import AsynResidualSAC
from scripts.rsrl.agent.asyn_residual_td3 import AsynResidualTD3
from typing import Any, Dict, Optional
import logging

from isaac_scripts.sb3.td3_bc import TD3BC


def deserialize_space(d: dict) -> gym.Space:
    t = d["type"]

    if t == "Box":
        low = np.array(d["low"], dtype=d["dtype"]) if isinstance(
            d["low"], (list, np.ndarray)) else d["low"]
        high = np.array(d["high"], dtype=d["dtype"]) if isinstance(
            d["high"], (list, np.ndarray)) else d["high"]
        return spaces.Box(low=low,
                          high=high,
                          shape=tuple(d["shape"]),
                          dtype=np.dtype(d["dtype"]))

    if t == "Discrete":
        return spaces.Discrete(d["n"])

    if t == "MultiDiscrete":
        return spaces.MultiDiscrete(np.array(d["nvec"], dtype=np.int64))

    if t == "MultiBinary":
        return spaces.MultiBinary(d["n"])

    if t == "Tuple":
        return spaces.Tuple(tuple(deserialize_space(s) for s in d["spaces"]))

    if t == "Dict":
        return spaces.Dict({
            k: deserialize_space(v)
            for k, v in d["spaces"].items()
        })


def process_sb3_cfg(cfg: dict) -> dict:
    """Convert simple YAML types to Stable-Baselines classes/components.

    Args:
        cfg: A configuration dictionary.

    Returns:
        A dictionary containing the converted configuration.

    Reference:
        https://github.com/DLR-RM/rl-baselines3-zoo/blob/0e5eb145faefa33e7d79c7f8c179788574b20da5/utils/exp_manager.py#L358
    """

    def update_dict(hyperparams: dict[str, Any]) -> dict[str, Any]:
        for key, value in hyperparams.items():
            if isinstance(value, dict):
                update_dict(value)
            else:

                if key in [
                        "policy_kwargs", "replay_buffer_class",
                        "replay_buffer_kwargs"
                ]:

                    hyperparams[key] = eval(value)
                elif key in [
                        "learning_rate", "clip_range", "clip_range_vf",
                        "delta_std"
                ]:
                    if isinstance(value, str):
                        _, initial_value = value.split("_")
                        initial_value = float(initial_value)
                        hyperparams[
                            key] = lambda progress_remaining: progress_remaining * initial_value
                    elif isinstance(value, (float, int)):
                        # Negative value: ignore (ex: for clipping)
                        if value < 0:
                            continue
                        hyperparams[key] = constant_fn(float(value))
                    else:
                        raise ValueError(
                            f"Invalid value for {key}: {hyperparams[key]}")

        return hyperparams

    # parse agent configuration and convert to classes
    return update_dict(cfg)


def _dtype_str(dtype):
    # Robustly turn dtype to string
    return np.dtype(dtype).name


def _maybe_scalar_list(arr):
    """Return a Python scalar if all values are equal, else a (nested) list."""
    a = np.array(arr)
    if a.size == 1:
        return a.item()
    # if all equal -> scalar
    if np.all(a == a.flat[0]):
        return a.flat[0].item()
    return a.tolist()


def serialize_space(space):
    """Serialize a Gym/Gymnasium space to a JSON-serializable dict."""
    if isinstance(space, spaces.Box):
        return {
            "type": "Box",
            "shape": list(space.shape),
            "dtype": _dtype_str(space.dtype),
            # low/high can be arrays; keep JSON-safe
            "low": _maybe_scalar_list(space.low),
            "high": _maybe_scalar_list(space.high),
        }
    elif isinstance(space, spaces.Discrete):
        return {
            "type": "Discrete",
            "n": int(space.n),
        }
    elif isinstance(space, spaces.MultiDiscrete):
        return {
            "type": "MultiDiscrete",
            "nvec": np.asarray(space.nvec, dtype=np.int64).tolist(),
        }
    elif isinstance(space, spaces.MultiBinary):
        return {
            "type":
            "MultiBinary",
            "n":
            int(space.n) if np.isscalar(space.n) else np.asarray(
                space.n, dtype=np.int64).tolist(),
        }
    elif isinstance(space, spaces.Tuple):
        return {
            "type": "Tuple",
            "spaces": [serialize_space(s) for s in space.spaces],
        }
    elif isinstance(space, spaces.Dict):
        return {
            "type": "Dict",
            "spaces": {
                k: serialize_space(v)
                for k, v in space.spaces.items()
            },
        }
    else:
        # Fallback for custom spaces
        return {
            "type": type(space).__name__,
            "repr": str(space),
        }


def init_algo_from_dict(cfg: Dict[str, Any]) -> None:
    """Actually construct the AsynResidualSAC here."""
    logging.info("[Sb3Learner] Initializing AsynResidualSAC...")

    observation_space = deserialize_space(cfg.pop("observation_space"))

    action_space = deserialize_space(cfg.pop("action_space"))

    policy_arch = cfg.pop("policy")

    cfg_dict = process_sb3_cfg(cfg)
    rl_type = cfg_dict.pop("rl_type", "td3")

    # cfg_dict.pop("n_timesteps")
    # cfg_dict.pop("rollout_id", 0)

    if rl_type == "td3":
        algo = TD3BC(policy_arch,
                     observation_space=observation_space,
                     action_space=action_space,
                     verbose=1,
                     **cfg_dict)
    elif rl_type == "sac":

        algo = AsynResidualSAC(policy_arch,
                               observation_space=observation_space,
                               action_space=action_space,
                               verbose=1,
                               **cfg_dict)
    return algo, observation_space, action_space
