import os

import numpy as np

import yaml
import copy
import argparse
import json
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
parser.add_argument("--deformable_list",
                    type=str,
                    default=None,
                    help="Number of environments to simulate.")
parser.add_argument("--rigid_list",
                    type=str,
                    default=None,
                    help="Number of environments to simulate.")

parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--use_gripper",
                    action="store_true",
                    default=False,
                    help="")
# phyiscal property setting
parser.add_argument("--fix_params", type=str, default=None, help="")
parser.add_argument("--parmas_range",
                    type=list,
                    default=None,
                    help="parmas_range.")

parser.add_argument("--random_params",
                    type=str,
                    default=None,
                    help="randomness_params.")


def recursive_update(d, u):
    for k, v in u.items():
        # If the value is None, skip this iteration
        if v is None:
            continue

        if isinstance(v, dict):
            # Ensure d[k] is a dictionary or set it to an empty dictionary if None
            if d.get(k) is None:
                d[k] = {}
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def find_file(filename, search_path):
    for root, dirs, files in os.walk(search_path):
        if filename in files:
            return os.path.join(root, filename)
    return None  # Return None if the file is not found


def update_shelf_setting(object_config):

    shelf_params = object_config["params"]["Shelf"]
    shelf_size = shelf_params["size"]
    shelf_pos = shelf_params["pos"]
    shelf_scale = shelf_params["scale"]

    # Creating block sizes (assuming each block has similar dimensions to the shelf's depth for simplicity)
    block_depth = shelf_scale[
        0]  # You can adjust the thickness of the block as needed

    left_block_size = [block_depth, shelf_size[0], shelf_size[2]]
    left_block_pos = [
        shelf_pos[0],
        shelf_pos[1] - shelf_size[1] / 2 -
        block_depth / 2,  # Position it on the left of the shelf
        shelf_pos[2]
    ]
    right_block_size = [block_depth, shelf_size[0], shelf_size[2]]
    right_block_pos = [
        shelf_pos[0],
        shelf_pos[1] + shelf_size[1] / 2 +
        block_depth / 2,  # Position it on the right of the shelf
        shelf_pos[2]
    ]
    front_block_size = [block_depth, shelf_size[1], shelf_size[2]]
    front_block_pos = [
        shelf_pos[0] + shelf_size[0] / 2 +
        block_depth / 2,  # Position it in front of the shelf
        shelf_pos[1],
        shelf_pos[2]
    ]

    back_block_size = [block_depth, shelf_size[1], shelf_size[2]]
    back_bloc_pos = [
        shelf_pos[0] - shelf_size[0] / 2 -
        block_depth / 2,  # Position it in front of the shelf
        shelf_pos[1],
        shelf_pos[2]
    ]
    shelf_pos = [
        left_block_pos, right_block_pos, front_block_pos, back_bloc_pos
    ]
    shelf_block_size = [
        left_block_size, right_block_size, front_block_size, back_block_size
    ]

    target_orientation = shelf_params["target_orientation"]
    for index, ori_setting in enumerate(target_orientation[:3]):
        object_config["params"]["AssstObject"][f"shelfblock_{index}"] = {}
        object_config["params"]["AssstObject"][f"shelfblock_{index}"][
            "path"] = shelf_params["path"]
        object_config["params"]["AssstObject"][f"shelfblock_{index}"][
            "scale"] = shelf_block_size[index]
        object_config["params"]["AssstObject"][f"shelfblock_{index}"][
            "pos"] = shelf_pos[index]
        object_config["params"]["AssstObject"][f"shelfblock_{index}"][
            "rot"] = {}
        object_config["params"]["AssstObject"][f"shelfblock_{index}"]["rot"][
            "axis"] = ori_setting[0]
        object_config["params"]["AssstObject"][f"shelfblock_{index}"]["rot"][
            "angles"] = ori_setting[1]
    return object_config


def update_deformable_setting(object_config, deformable_list, fix_params,
                              random_params):

    object_count = 2
    # Open the YAML file
    for index, object_name in enumerate(deformable_list):
        env_config = f"source/config/task/stow/deformable_object/{object_name}_stow.yaml"

        with open(env_config, 'r') as file:
            # Load the contents of the file
            config = yaml.safe_load(file)

        object_config["params"]["DeformableObject"][object_name] = {}
        object_config["params"]["DeformableObject"][
            object_name] = copy.deepcopy(
                config["params"]["DeformableObject"]["deform_object"])

        object_config["params"]["DeformableObject"][
            object_name] = copy.deepcopy(
                recursive_update(
                    object_config["params"]["DeformableObject"]
                    ["deform_object"], config["params"]["DeformableObject"]
                    ["deform_object"]["env_setting"]))
        object_config["params"]["DeformableObject"][object_name]["pos"][
            1] = 0.20 * object_count
        object_config["params"]["DeformableObject"][object_name]["pos"][
            0] = 2.0
        object_config["params"]["DeformableObject"][object_name]["pos"][
            2] = 0.3
        object_count += 1
        if fix_params is None:
            object_config["params"]["DeformableObject"][object_name][
                "physical_params"]["fix_params"] = {}
        if random_params is None:
            object_config["params"]["DeformableObject"][object_name][
                "physical_params"]["params_range"] = {}
            object_config["params"]["DeformableObject"][object_name][
                "physical_params"]["params_values"] = {}

    del object_config["params"]["DeformableObject"]["deform_object"]
    return object_config


def update_rigid_setting(object_config, rigid_list):
    rigid_template = object_config["params"]["RigidObjectTemplate"][
        "rigid_object"]

    for index, object_name in enumerate(rigid_list):

        file_path = find_file(object_name + ".usd", "source")
        object_config["params"]["RigidObject"][
            f"rigid_object_{object_name}"] = {}
        object_config["params"]["RigidObject"][
            f"rigid_object_{object_name}"] = copy.deepcopy(rigid_template)
        object_config["params"]["RigidObject"][f"rigid_object_{object_name}"][
            "path"] = file_path
        object_config["params"]["RigidObject"][f"rigid_object_{object_name}"][
            "pos"][1] = (index * 0.2)
        object_config["params"]["RigidObject"][f"rigid_object_{object_name}"][
            "pos"][0] = 1.0
        object_config["params"]["RigidObject"][f"rigid_object_{object_name}"][
            "pos"][2] = 0.3
        object_config["params"]["RigidObject"][f"rigid_object_{object_name}"][
            "name"] = object_name
    return object_config


def merge_config():
    with open("source/config/task/stow/default_config.yaml", 'r') as file:
        # Load the contents of the file
        object_config = yaml.safe_load(file)
    with open("source/config/task/stow/default_camera.yaml", 'r') as file:
        # Load the contents of the file
        camera_config = yaml.safe_load(file)
    with open("source/config/task/stow/default_assets.yaml", 'r') as file:
        # Load the contents of the file
        assets_config = yaml.safe_load(file)
    with open("source/config/task/stow/default_deformable.yaml", 'r') as file:
        # Load the contents of the file
        deformable_config = yaml.safe_load(file)
    with open("source/config/task/stow/default_rigid.yaml", 'r') as file:
        # Load the contents of the file
        rigid_config = yaml.safe_load(file)
    object_config["params"].update(camera_config)
    object_config["params"].update(assets_config)
    object_config["params"].update(deformable_config)
    object_config["params"].update(rigid_config)
    return object_config


def set_physical_property(object_config, fix_params, random_params):
    if fix_params is not None or random_params is not None:
        object_config["params"]["DeformableObject"]["deform_object"][
            "random_parmas"] = True
    else:
        return object_config

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
    else:
        object_config["params"]["DeformableObject"]["deform_object"][
            "physical_params"]["params_values"] = {}
        object_config["params"]["DeformableObject"]["deform_object"][
            "physical_params"]["params_range"] = {}
    if fix_params is not None:
        fix_params = json.loads(fix_params)

        object_config["params"]["DeformableObject"]["deform_object"][
            "physical_params"]["fix_params"].update(fix_params)
    else:
        object_config["params"]["DeformableObject"]["deform_object"][
            "physical_params"]["fix_params"] = {}

    return object_config


def save_params_to_yaml(args_cli, save_config_path="logs/"):

    fix_params = args_cli.fix_params
    random_params = args_cli.random_params

    deformable_list = args_cli.deformable_list.strip('[]').split(',')
    if args_cli.rigid_list is not None:
        rigid_list = args_cli.rigid_list.strip('[]').split(',')
    task = args_cli.task
    use_gripper = args_cli.use_gripper

    os.makedirs(save_config_path, exist_ok=True)
    object_config = merge_config()

    if "Abs" not in task and "Rel" not in task:
        object_config["params"]["Robot"]["init"] = False

    if not use_gripper:
        for name in object_config["params"]["RigidObject"].keys():
            if "gripper" in name:
                object_config["params"]["RigidObject"][name]["init"] = False
    object_config = set_physical_property(object_config, fix_params,
                                          random_params)
    object_config = update_deformable_setting(object_config, deformable_list,
                                              fix_params, random_params)

    if args_cli.rigid_list is not None:
        object_config = update_rigid_setting(object_config, rigid_list)

    # make shelf based on the information
    object_config = update_shelf_setting(object_config)

    save_config = f'{save_config_path}/config.yaml'

    with open(save_config, 'w') as file:
        yaml.dump(object_config, file, default_flow_style=False)
    return save_config, object_config
