import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.1"  # optional
# os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
# launch omniverse app
import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--agent",
                        type=str,
                        default="sac",
                        help="Name of agent.")
    parser.add_argument("--exp_name",
                        type=str,
                        default="Leap_hand",
                        help="Name of the experiment for wandb logging.")
    parser.add_argument("--max_traj_length",
                        type=int,
                        default=100,
                        help="Maximum length of trajectory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--save_model",
                        action="store_true",
                        help="Whether to save model.")
    parser.add_argument("--batch_size",
                        type=int,
                        default=256,
                        help="Batch size.")
    parser.add_argument("--critic_actor_ratio",
                        type=int,
                        default=8,
                        help="critic to actor update ratio.")

    parser.add_argument("--max_steps",
                        type=int,
                        default=100000000000,
                        help="Maximum number of training steps.")
    parser.add_argument("--replay_buffer_capacity",
                        type=int,
                        default=10000,
                        help="Replay buffer capacity.")

    parser.add_argument("--random_steps",
                        type=int,
                        default=300,
                        help="Sample random actions for this many steps.")
    parser.add_argument("--training_starts",
                        type=int,
                        default=300,
                        help="Training starts after this step.")
    parser.add_argument("--steps_per_update",
                        type=int,
                        default=30,
                        help="Number of steps per update the server.")

    parser.add_argument("--log_period",
                        type=int,
                        default=10,
                        help="Logging period.")
    parser.add_argument("--eval_period",
                        type=int,
                        default=2000,
                        help="Evaluation period.")
    parser.add_argument("--eval_n_trajs",
                        type=int,
                        default=5,
                        help="Number of trajectories for evaluation.")

    parser.add_argument("--learner",
                        action="store_true",
                        help="Is this a learner or a trainer.")
    parser.add_argument("--actor",
                        action="store_true",
                        help="Is this a learner or a trainer.")
    parser.add_argument("--render",
                        action="store_true",
                        help="Render the environment.")
    parser.add_argument("--ip",
                        type=str,
                        default="localhost",
                        help="IP address of the learner.")
    parser.add_argument("--checkpoint_period",
                        type=int,
                        default=20000,
                        help="Period to save checkpoints.")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=
        "/home/ensu/Documents/weird/IsaacLab/logs/serl_demo_checkpoints",
        help="Path to save checkpoints.")

    parser.add_argument(
        "--debug", action="store_true",
        help="Debug mode.")  # debug mode will disable wandb logging

    parser.add_argument("--log_rlds_path",
                        type=str,
                        default=None,
                        help="Path to save RLDS logs.")
    parser.add_argument("--preload_rlds_path",
                        type=str,
                        default=None,
                        help="Path to preload RLDS data.")

    parser.add_argument("--utd_ratio",
                        type=int,
                        default=1,
                        help="Update to data ratio.")
    parser.add_argument("--eval_checkpoint_step",
                        type=int,
                        default=180000,
                        help="evaluate the policy from ckpt at this step")

    parser.add_argument("--demo_path",
                        type=str,
                        default=None,
                        help="evaluate the policy from ckpt at this step")

    parser.add_argument("--cta_ratio",
                        type=int,
                        default=2,
                        help="Update to data ratio.")
    parser.add_argument("--encoder_type",
                        type=str,
                        default="resnet-pretrained",
                        help="Encoder type.")
    parser.add_argument("--rl_type",
                        type=str,
                        default="drq",
                        help="Type of RL algorithm.")
    parser.add_argument("--flush_every",
                        type=int,
                        default=1,
                        help="Flush every n steps.")
    parser.add_argument("--gemini_rewards",
                        action="store_true",
                        help="use gemini for the reward labeling")

    parser.add_argument("--critic_input",
                        type=str,
                        choices=['res', 'sum', 'concat'],
                        default='res')

    return parser.parse_args()


from scripts.serl.env.serl_wrapper import ImageCustomSerlEnvWrapper, StateCustomSerlEnvWrapper
from scripts.serl.agent.serl_drq_agent import DrqAgent
from scripts.serl.agent.serl_sac_agent import SerlSACAgent


def main():
    args = get_args()

    if "drq" in args.rl_type:
        rl_agent_env = ImageCustomSerlEnvWrapper()
        serl_agent = DrqAgent(rl_agent_env, args)
    elif "sac" in args.rl_type:
        rl_agent_env = StateCustomSerlEnvWrapper()
        serl_agent = SerlSACAgent(rl_agent_env, args)

    serl_agent.run_agent()


if __name__ == "__main__":
    main()
