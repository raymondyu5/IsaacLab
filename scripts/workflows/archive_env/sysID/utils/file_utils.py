import os
import wandb

import yaml
import numpy as np
import json


def create_log_dirs(base_dir="logs/rabbit/static_gs/", arg=None):
    if arg.name is not None:
        base_dir += f"_{arg.name}"
    else:
        random_params = arg.random_params.strip('[]').split(',')
        for params in random_params:
            base_dir += f"_{params}"
    os.makedirs(base_dir + "/video", exist_ok=True)
    os.makedirs(base_dir + "/result", exist_ok=True)

    return base_dir


def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def save_params_to_yaml(args_cli, save_config_path="logs/"):
    env_config = args_cli.env_config
    task = args_cli.task
    random_params = args_cli.random_params
    use_gripper = args_cli.use_gripper
    num_explore_actions = args_cli.num_explore_actions
    fix_params = args_cli.fix_params
    os.makedirs(save_config_path, exist_ok=True)
    # Open the YAML file
    with open(env_config, 'r') as file:
        # Load the contents of the file
        config = yaml.safe_load(file)

    with open(f"{env_config.rpartition('/')[0]}/default_config.yaml",
              'r') as file:
        # Load the contents of the file
        object_config = yaml.safe_load(file)

    if "Abs" not in task and "Rel" not in task:
        object_config["params"]["Robot"]["init"] = False

    if not use_gripper:
        for name in object_config["params"]["RigidObject"].keys():
            if "gripper" in name:
                object_config["params"]["RigidObject"][name]["init"] = False

    object_config["params"]["DeformableObject"][
        "deform_object"] = recursive_update(
            object_config["params"]["DeformableObject"]["deform_object"],
            config["params"]["DeformableObject"]["deform_object"]
            ["env_setting"])

    if random_params is not None:
        random_params = random_params.strip('[]').split(',')
        for param in random_params:

            object_config["params"]["DeformableObject"]["deform_object"][
                "physical_params"]["fix_params"].pop(param, None)
        template_random_params = [
            *object_config["params"]["DeformableObject"]["deform_object"]
            ["physical_params"]["params_range"].keys()
        ]
        for key in template_random_params:
            if key not in random_params:
                object_config["params"]["DeformableObject"]["deform_object"][
                    "physical_params"]["params_range"].pop(key)

        template_random_params = [
            *object_config["params"]["DeformableObject"]["deform_object"]
            ["physical_params"]["params_values"].keys()
        ]
        for key in template_random_params:
            if key not in random_params:
                object_config["params"]["DeformableObject"]["deform_object"][
                    "physical_params"]["params_values"].pop(key)

    if fix_params is not None:

        fix_params = json.loads(fix_params)

        object_config["params"]["DeformableObject"]["deform_object"][
            "physical_params"]["fix_params"].update(fix_params)

    num_explore_actions = args_cli.num_explore_actions
    num_explore_actions = np.array(
        num_explore_actions.strip('[]').split(',')).astype(int)

    object_config["params"]["DeformableObject"][
        "deform_object"]["env_setting"]["num_explore_actions"] = int(
            np.sum(num_explore_actions))
    object_config["params"]["DeformableObject"]["deform_object"][
        "env_setting"]["num_robot_actions"] = int(num_explore_actions[0])
    object_config["params"]["DeformableObject"]["deform_object"][
        "env_setting"]["num_gripper_actions"] = int(num_explore_actions[1])

    if not args_cli.use_gripper:
        if num_explore_actions[1] != 0:
            AssertionError("Gripper actions are not allowed")
    if "Abs" not in args_cli.task and "IK" not in args_cli.task:
        if num_explore_actions[0] != 0:
            AssertionError("Robot actions are not allowed")

    save_config = f'{save_config_path}/config.yaml'

    with open(save_config, 'w') as file:
        yaml.dump(object_config, file, default_flow_style=False)
    return save_config, object_config
