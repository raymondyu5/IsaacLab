import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"  # optional

from scripts.serl.agent.serl_sac_agent import SerlSACAgent
from scripts.serl.agent.serl_rlpd_agent import SerlRLPDAgent
from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml
from scripts.serl.utils.serl_algo_wrapper import serl_parser, FLAGS
import sys
# launch omniverse app

AppLauncher.add_app_launcher_args(serl_parser)
args_cli, _ = serl_parser.parse_known_args(sys.argv[1:])

# Parse the command line arguments for the serl_launcher

absl_names = {f"--{name}" for name in FLAGS}
# Keep only absl-known flags in sys.argv for app.run
absl_argv = [
    a for a in sys.argv
    if not (a.startswith("--") and a.split("=")[0] not in absl_names)
]
sys.argv = absl_argv
FLAGS(sys.argv)
import gymnasium as gym

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""
from scripts.serl.env.residual_rl_wrapper import ResidualRLDatawrapperEnv

import gymnasium as gym
import torch


def setup_env(args_cli, save_config):
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args_cli.task,
                            device=args_cli.device,
                            num_envs=args_cli.num_envs,
                            use_fabric=not args_cli.disable_fabric,
                            config_yaml=save_config)

    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    return gym.make(
        args_cli.task,
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None).unwrapped


from scripts.serl.env.serl_wrapper import SerlEnvWrapper, CustomSerlEnvWrapper


def main():
    """Zero actions agent with Isaac Lab environment."""

    save_config, config = save_params_to_yaml(args_cli, args_cli.log_dir)
    # create environment
    save_config["params"]["add_right_hand"] = args_cli.add_right_hand
    save_config["params"]["add_left_hand"] = args_cli.add_left_hand
    save_config["params"]["num_envs"] = args_cli.num_envs
    save_config["params"]["rl_train"] = True
    save_config["params"]["sample_points"] = True
    save_config["params"]["real_eval_mode"] = True
    if args_cli.target_object_name is not None:

        object_name = args_cli.target_object_name
        save_config["params"]["multi_cluster_rigid"]["right_hand_object"][
            "objects_list"] = [object_name]
    if "Joint-Rel" in args_cli.task:
        args_cli.log_dir = os.path.join(args_cli.log_dir, "joint_pose")

    env = setup_env(args_cli, save_config)
    obs, _ = env.reset()
    for i in range(10):
        action = env.action_space.sample() * 0.0

    obs, _ = env.reset()

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment

    rl_env = ResidualRLDatawrapperEnv(
        env,
        save_config,
        args_cli=args_cli,
        use_relative_pose=True if "Rel" in args_cli.task else False,
    )

    # wrap around environment for stable baselines
    rl_agent_env = SerlEnvWrapper(rl_env, )
    if args_cli.rl_type == "sac":
        # create SAC agent
        agent = SerlSACAgent
    elif args_cli.rl_type == "rlpd":
        agent = SerlRLPDAgent

    serl_agent = agent(rl_agent_env, FLAGS)

    serl_agent.evaluate()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
