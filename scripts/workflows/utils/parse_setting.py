import os
import yaml
import argparse
import gym
import random
import copy
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
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--config_file",
                    type=str,
                    default=None,
                    help="Name of the task.")
parser.add_argument("--log_dir", type=str, default=None, help="log dir.")
parser.add_argument("--filename", type=str, default=None, help="test")
parser.add_argument("--num_demos",
                    type=int,
                    default=2000,
                    help="number of demos")
parser.add_argument("--normalize_action",
                    action="store_true",
                    default=False,
                    help="Disable fabric and use USD I/O operations.")
parser.add_argument("--eval_type",
                    default="train",
                    help="the eval type for the training")
parser.add_argument("--open_loop",
                    action="store_true",
                    help="Set to True to use open loop evaluation")
parser.add_argument("--init_grasp",
                    action="store_true",
                    help="Set to True to initialize grasp")
parser.add_argument("--init_open",
                    action="store_true",
                    help="Set to True to initialize cabinet")
parser.add_argument("--init_placement",
                    action="store_true",
                    help="Set to True to initialize placement")
parser.add_argument("--init_close",
                    action="store_true",
                    help="Set to True to initialize close")

parser.add_argument("--end_name",
                    default="",
                    help="the eval type for the training")

parser.add_argument("--noise_pc",
                    action="store_true",
                    help="Set to True to initialize close")

parser.add_argument("--load_path",
                    default=None,
                    help="the eval type for the training")
parser.add_argument("--save_path",
                    default=None,
                    help="the eval type for the training")
parser.add_argument("--failure_type",
                    default=None,
                    help="the eval type for the training")
parser.add_argument("--replay",
                    action="store_true",
                    help="the eval type for the training")
parser.add_argument("--failure_config",
                    type=str,
                    default=None,
                    help="Name of the task.")
parser.add_argument("--rl_config",
                    type=str,
                    default=None,
                    help="rl config file.")
parser.add_argument("--client_url",
                    type=str,
                    default=None,
                    help="url for the inference clinet link.")

parser.add_argument("--save_replay_path",
                    default=None,
                    help="the eval type for the training")
parser.add_argument("--load_replay_path",
                    default=None,
                    help="the eval type for the training")
parser.add_argument("--load_openvla_dir",
                    default=None,
                    help="the eval type for the training")
parser.add_argument(
    "--eval_freq",
    default=10,
)
parser.add_argument(
    "--render_image",
    action="store_true",
)


def scale_joint_values(env_config):
    scale_factor = env_config["params"]["ArticulationObject"]["kitchen"][
        "scale"][0]
    for joint, value in env_config["params"]["ArticulationObject"]["kitchen"][
            "joints"].items():
        if "prismatic" in joint:
            env_config["params"]["ArticulationObject"]["kitchen"]["joints"][
                joint] = value * scale_factor

    return env_config


def deep_merge(dict1, dict2):
    """Recursively merge two dictionaries."""

    for key, value in dict2.items():
        if isinstance(value, dict) and key in dict1 and isinstance(
                dict1[key], dict):
            dict1[key] = deep_merge(dict1[key], value)
        else:
            dict1[key] = value

    return dict1


def save_params_to_yaml(
    args_cli,
    save_config_path="logs/",
    random_shuffle=True,
):
    config_file = args_cli.config_file

    with open(config_file, 'r') as file:
        env_config = yaml.safe_load(file)
    # if args_cli.failure_config is not None:
    #     with open(args_cli.failure_config, 'r') as file:
    #         failure_config = yaml.safe_load(file)
    #     env_config = deep_merge(env_config, failure_config)
    rigid_object_config = env_config["params"].get("RigidObject")
    rigid_object_config_file = []
    random_shuffle = env_config["params"].get("random_shuffle", True)

    if rigid_object_config is not None and rigid_object_config.get(
            "config_file") is not None:

        if isinstance(rigid_object_config["config_file"], str):
            rigid_object_config_file += [rigid_object_config["config_file"]]
        else:

            cluster_config = {}

            env_config["params"]["multi_cluster_rigid"] = {}

            for i in range(len(rigid_object_config["config_file"])):
                rigid_object_config_file += [
                    rigid_object_config["config_file"][i]["file"]
                ]
                target_manipulated_object = env_config["params"].get(
                    "target_manipulated_object", "all")

                if isinstance(rigid_object_config["config_file"][i]["file"],
                              str):

                    with open(rigid_object_config["config_file"][i]["file"],
                              'r') as file:
                        rigid_configs_setting = yaml.safe_load(file)
                        if target_manipulated_object != "all":
                            for object_name in list(
                                    rigid_configs_setting["params"]
                                ["RigidObject"].keys()):
                                if object_name not in target_manipulated_object:
                                    rigid_configs_setting["params"][
                                        "RigidObject"].pop(object_name)

                        env_config = deep_merge(env_config,
                                                rigid_configs_setting)

                else:
                    rigid_configs_setting = None
                    for file in rigid_object_config["config_file"][i]["file"]:

                        with open(file, 'r') as file:
                            rigid_configs = yaml.safe_load(file)
                            if rigid_configs_setting is None:
                                rigid_configs_setting = rigid_configs
                            else:
                                rigid_configs_setting = deep_merge(
                                    rigid_configs, rigid_configs_setting)
                    if target_manipulated_object != "all":

                        for object_name in list(rigid_configs_setting["params"]
                                                ["RigidObject"].keys()):
                            if object_name not in target_manipulated_object:
                                rigid_configs_setting["params"]["RigidObject"][
                                    object_name].pop(object_name)
                    env_config = deep_merge(env_config, rigid_configs_setting)

                rigid_cofig = rigid_object_config["config_file"][i]
                cluster_config[rigid_object_config["config_file"][i]
                               ["name"]] = rigid_cofig
                key = list(
                    rigid_configs_setting["params"]["RigidObject"].keys())
                if random_shuffle:
                    random.shuffle(key)

                if target_manipulated_object != "all":
                    key = target_manipulated_object

                cluster_config[rigid_object_config["config_file"][i]
                               ["name"]]["objects_list"] = key

            env_config["params"]["multi_cluster_rigid"] = env_config["params"][
                "multi_cluster_rigid"] | copy.deepcopy(cluster_config)

        # for cofig_file in rigid_object_config_file:
        #     with open(cofig_file, 'r') as file:

        #         rigid_config = yaml.safe_load(file)
        #     if target_manipulated_object != "all":

        #         for object_name in list(
        #                 rigid_config["params"]["RigidObject"].keys()):
        #             if object_name not in target_manipulated_object:
        #                 rigid_config["params"]["RigidObject"].pop(object_name)

        #     env_config = deep_merge(env_config, rigid_config)

        env_config["params"]["RigidObject"].pop("config_file", None)

    articulation_objects = env_config["params"].get("ArticulationObject")
    if articulation_objects is not None and articulation_objects.get(
            "config_file") is not None:
        articulation_config_file = env_config["params"]["ArticulationObject"][
            "config_file"]

        if isinstance(articulation_config_file, str):
            articulation_config_file = [articulation_config_file]

        for cofig_file in articulation_config_file:

            with open(cofig_file["file"], 'r') as file:
                articulation_config = yaml.safe_load(file)
            env_config["params"]["ArticulationObject"].pop("config_file", None)
            env_config = deep_merge(env_config, articulation_config)
        env_config["params"]["articulation_cfg"] = cofig_file

        env_config["params"]["spawn_multi_articulation"] = cofig_file

    os.makedirs(save_config_path, exist_ok=True)
    save_config = f'{save_config_path}/config.yaml'

    with open(save_config, 'w') as file:
        yaml.dump(env_config, file, default_flow_style=False)

    return env_config, save_config
