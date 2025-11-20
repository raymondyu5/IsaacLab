import numpy as np
import os
import random
from datetime import datetime

import os
import yaml
from box import Box

from scripts.workflows.utils.parse_setting import parser
import copy

serl_parser = copy.deepcopy(parser)
serl_parser.add_argument("--seed",
                         type=int,
                         default=None,
                         help="Seed used for the environment")
serl_parser.add_argument("--max_iterations",
                         type=int,
                         default=None,
                         help="RL Policy training iterations.")

serl_parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
serl_parser.add_argument(
    "--add_left_hand",
    action="store_true",
)

serl_parser.add_argument(
    "--action_framework",
    default=None,
)

serl_parser.add_argument(
    "--diffusion_path",
    default=None,
)

serl_parser.add_argument(
    "--latent_dim",
    default=32,
    type=int,
)
serl_parser.add_argument(
    "--use_relative_finger_pose",
    action="store_true",
)
serl_parser.add_argument(
    "--rl_type",
    default="ppo",
)

serl_parser.add_argument(
    "--action_scale",
    default=1.0,
    type=float,
)

serl_parser.add_argument(
    "--bc_dir",
    default=None,
    type=str,
)

serl_parser.add_argument(
    "--use_residual_action",
    action="store_true",
)

serl_parser.add_argument(
    "--use_chunk_action",
    action="store_true",
)
serl_parser.add_argument(
    "--use_interpolate_chunk",
    action="store_true",
)
serl_parser.add_argument(
    "--residual_step",
    default=1,
    type=int,
)

serl_parser.add_argument(
    "--resume",
    action="store_true",
)

serl_parser.add_argument(
    "--analysis",
    action="store_true",
)

serl_parser.add_argument(
    "--checkpoint",
    default=None,
)

serl_parser.add_argument("--video",
                         action="store_true",
                         default=False,
                         help="Record videos during training.")

serl_parser.add_argument("--video_length",
                         type=int,
                         default=200,
                         help="Length of the recorded video (in steps).")
serl_parser.add_argument("--video_interval",
                         type=int,
                         default=20000,
                         help="Interval between video recordings (in steps).")

serl_parser.add_argument("--distributed",
                         action="store_true",
                         default=False,
                         help="Run training with multiple GPUs or nodes.")
serl_parser.add_argument(
    "--use_visual_obs",
    action="store_true",
)

serl_parser.add_argument(
    "--target_object_name",
    type=str,
    default=None,  # Options: tomato_soup_can, banana, cereal_box, etc.
)

serl_parser.add_argument(
    "--use_base_action",
    action="store_true",
)

serl_parser.add_argument(
    "--random_camera_pose",
    action="store_true",
)

serl_parser.add_argument(
    "--diffusion_checkpoint",
    type=str,
    default="latest",  # Options: open_loop, close_loop, replay
)

serl_parser.add_argument(
    "--eval_mode",
    type=str,
    default="close_loop",  # Options: open_loop, close_loop, replay
)

serl_parser.add_argument(
    "--use_last_checkpoint",
    action="store_true",
    help=
    "When no checkpoint provided, use the last saved model. Otherwise use the best saved model.",
)

serl_parser.add_argument(
    "--collect_relative_finger_pose",
    action="store_true",
)

serl_parser.add_argument(
    "--real_eval_mode",
    action="store_true",
)

# Training hyperparameters
serl_parser.add_argument("--batch_size",
                         type=int,
                         default=256,
                         help="Batch size.")
serl_parser.add_argument("--critic_actor_ratio",
                         type=int,
                         default=8,
                         help="critic to actor update ratio.")
serl_parser.add_argument("--max_steps",
                         type=int,
                         default=1_000_000,
                         help="Maximum number of training steps.")
serl_parser.add_argument("--replay_buffer_capacity",
                         type=int,
                         default=1_000_000,
                         help="Replay buffer capacity.")
serl_parser.add_argument("--random_steps",
                         type=int,
                         default=300,
                         help="Sample random actions for this many steps.")
serl_parser.add_argument("--training_starts",
                         type=int,
                         default=300,
                         help="Training starts after this step.")
serl_parser.add_argument("--steps_per_update",
                         type=int,
                         default=30,
                         help="Number of steps per update the server.")

# Logging / eval
serl_parser.add_argument("--log_period",
                         type=int,
                         default=10,
                         help="Logging period (environment steps).")
serl_parser.add_argument("--eval_period",
                         type=int,
                         default=2000,
                         help="Evaluation period (environment steps).")
serl_parser.add_argument("--eval_n_trajs",
                         type=int,
                         default=5,
                         help="Number of trajectories for evaluation.")

# Roles & runtime
serl_parser.add_argument("--learner",
                         action="store_true",
                         help="Is this a learner process.")
serl_parser.add_argument("--actor",
                         action="store_true",
                         help="Is this an actor process.")
serl_parser.add_argument("--render",
                         action="store_true",
                         help="Render the environment.")
serl_parser.add_argument("--ip",
                         type=str,
                         default="localhost",
                         help="IP address of the learner/server.")

# Checkpointing
serl_parser.add_argument(
    "--checkpoint_period",
    type=int,
    default=20000,
    help="Period to save checkpoints (steps). 0 disables.")
serl_parser.add_argument("--checkpoint_path",
                         type=str,
                         default="logs/serl_checkpoints",
                         help="Path to save checkpoints.")

# Misc
serl_parser.add_argument("--debug",
                         action="store_true",
                         help="Debug mode (disables wandb logging).")
serl_parser.add_argument("--log_rlds_path",
                         type=str,
                         default=None,
                         help="Path to save RLDS logs.")
serl_parser.add_argument("--preload_rlds_path",
                         type=str,
                         default=None,
                         help="Path to preload RLDS data.")

serl_parser.add_argument("--demo_path",
                         type=str,
                         default=None,
                         help="Path to preload RLDS data.")
from absl import app, flags

FLAGS = flags.FLAGS

flags.DEFINE_string("agent", "sac", "Name of agent.")
flags.DEFINE_string("exp_name", "Leap_hand",
                    "Name of the experiment for wandb logging.")
flags.DEFINE_integer("max_traj_length", 100, "Maximum length of trajectory.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_bool("save_model", False, "Whether to save model.")
flags.DEFINE_integer("batch_size", 256, "Batch size.")
flags.DEFINE_integer("critic_actor_ratio", 8, "critic to actor update ratio.")

flags.DEFINE_integer("max_steps", 100000000000,
                     "Maximum number of training steps.")
flags.DEFINE_integer("replay_buffer_capacity", 10000,
                     "Replay buffer capacity.")

flags.DEFINE_integer("random_steps", 300,
                     "Sample random actions for this many steps.")
flags.DEFINE_integer("training_starts", 300,
                     "Training starts after this step.")
flags.DEFINE_integer("steps_per_update", 30,
                     "Number of steps per update the server.")

flags.DEFINE_integer("log_period", 10, "Logging period.")
flags.DEFINE_integer("eval_period", 2000, "Evaluation period.")
flags.DEFINE_integer("eval_n_trajs", 5,
                     "Number of trajectories for evaluation.")

# flag to indicate if this is a leaner or a actor
flags.DEFINE_boolean("learner", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("actor", False, "Is this a learner or a trainer.")
flags.DEFINE_boolean("render", False, "Render the environment.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_integer("checkpoint_period", 20000, "Period to save checkpoints.")
flags.DEFINE_string(
    "checkpoint_path",
    "/home/ensu/Documents/weird/IsaacLab/logs/serl_checkpoints",
    "Path to save checkpoints.")

flags.DEFINE_boolean("debug", False,
                     "Debug mode.")  # debug mode will disable wandb logging

flags.DEFINE_string("log_rlds_path", None, "Path to save RLDS logs.")
flags.DEFINE_string("preload_rlds_path", None, "Path to preload RLDS data.")

flags.DEFINE_integer("utd_ratio", 1, "Update to data ratio.")
flags.DEFINE_integer("eval_checkpoint_step", 180000,
                     "evaluate the policy from ckpt at this step")

flags.DEFINE_string("demo_path", None,
                    "evaluate the policy from ckpt at this step")
flags.DEFINE_string(
    "rl_type", None,
    "Type of RL algorithm. Options: sac, ppo, rlpd, drq,residual_drq")
flags.DEFINE_integer("cta_ratio", 2, "Update to data ratio.")
flags.DEFINE_string("encoder_type", "resnet-pretrained", "Encoder type.")
flags.DEFINE_integer("flush_every", 1, "Flush every n steps.")
flags.DEFINE_boolean("gemini_rewards", False,
                     "use gemini for the reward labeling")
flags.DEFINE_integer("latest_seq_id", -1, "latest sequence id for data store")
flags.DEFINE_string("critic_input", 'sum',
                    "critic input type. Options: sum, concat, res")
