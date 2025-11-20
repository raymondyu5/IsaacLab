from scripts.rsrl.agent.agentlace_trainer import TrainerServer, TrainerClient, TrainerSMInterface, make_trainer_config
import time
from scripts.rsrl.utils.sb3_utils import init_algo_from_dict

cfg = make_trainer_config(port_number=5555, broadcast_port=5556)
if "send_dict" not in cfg.request_types:
    cfg.request_types.append("send_dict")
from scripts.rsrl.utils.sb3_datastore import QueuedDataStore

data_store = QueuedDataStore(2000,
                             latest_seq_id=0)  # the queue size on the actor
client = TrainerClient(
    name="actor_env",
    server_ip="127.0.0.1",
    config=cfg,
    data_store=data_store,
    wait_for_server=True,
)

import numpy as np
import gymnasium as gym
from gymnasium import spaces


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


# Suppose you already have an env
# env = make_your_env()

obs_space = spaces.Dict({
    "rgb":
    spaces.Box(
        low=0,
        high=255,
        shape=(1, 128, 128, 3),
        dtype=np.uint8,
    ),
    "state":
    spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(10, ),
        dtype=np.float32,
    ),
})

obs_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(10, ),
    dtype=np.float32,
)
# === Define action space manually ===
# Example: 7-DoF arm with continuous actions
act_space = spaces.Box(
    low=-1.0,
    high=1.0,
    shape=(22, ),
    dtype=np.float32,
)

# === Use the same serialize helper from before ===
obs_space_dict = serialize_space(obs_space)
act_space_dict = serialize_space(act_space)

import yaml
from box import Box
with open("source/config/rl/hand_manipulation/sb3_td3_cfg.yaml",
          "r",
          encoding="utf-8") as file:
    yaml_data = yaml.safe_load(file)
agent_cfg = Box(yaml_data)

agent_cfg.seed = agent_cfg["seed"]
# from scripts.workflows.hand_manipulation.env.rl_env.sb3_wrapper import process_sb3_cfg

# agent_cfg = process_sb3_cfg(agent_cfg)
cfg_dict = agent_cfg.to_dict()

cfg_with_spaces = {
    **cfg_dict, "save_dir": "logs/residual_sac",
    "model_save_freq": 5,
    "observation_space": obs_space_dict,
    "action_space": act_space_dict,
    "rl_type": "td3"
}
# Send to the trainer server

import copy

obs = {
    # "rgb": np.zeros((1, 128, 128, 3), dtype=np.uint8),
    "state": np.zeros((1, 10), dtype=np.float32)
}
obs = np.zeros((1, 10), dtype=np.float32)
res = client.request("init_rl", cfg_with_spaces)
print("Server response:", res)

train_cfg = {
    "n_timesteps": cfg_with_spaces.pop("n_timesteps"),
    "rollout_id": cfg_with_spaces.pop("rollout_id", 0),
    "save_dir": cfg_with_spaces.pop("save_dir", "logs"),
    "model_save_freq": cfg_with_spaces.pop("model_save_freq", 10000),
}
agent = init_algo_from_dict(cfg_with_spaces | {"initial_buffer": False})


def update_params(params):

    agent.policy.load_state_dict(params)


client.recv_network_callback(update_params)
while True:

    data_store.insert(
        dict(obs=copy.deepcopy(obs),
             action=np.zeros((1, 22)),
             next_obs=copy.deepcopy(obs),
             reward=np.zeros(1),
             masks=0.0,
             done=np.zeros(1).astype(np.bool_),
             base_actions=np.zeros((1, 22)),
             infos=[dict()],
             next_base_actions=np.zeros((1, 22))))
    client.update()
    time.sleep(0.01)
