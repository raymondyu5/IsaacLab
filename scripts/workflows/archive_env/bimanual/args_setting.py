import argparse

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Random agent for Isaac Lab environments.")

parser.add_argument("--disable_fabric",
                    action="store_true",
                    default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs",
                    type=int,
                    default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--env_config",
                    type=str,
                    default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--random_params",
                    type=str,
                    default=None,
                    help="randomness_params.")

parser.add_argument("--Date", type=str, default=None, help="date.")
parser.add_argument("--beign_traing_id",
                    type=int,
                    default=None,
                    help="begin training id ")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_gripper",
                    action="store_true",
                    default=False,
                    help="")
parser.add_argument("--parmas_range",
                    type=list,
                    default=None,
                    help="parmas_range.")
parser.add_argument("--num_explore_actions", type=str, default=[2], help="")
parser.add_argument("--name", type=str, default=None, help="")
parser.add_argument("--fix_params", type=str, default=None, help="")
