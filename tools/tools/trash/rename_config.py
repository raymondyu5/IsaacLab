import yaml
import argparse


def replace_rabbit_with_sweep(config, old_name, new_name):
    if isinstance(config, dict):
        for key, value in config.items():
            if isinstance(value, str):
                config[key] = value.replace(old_name, new_name)
            else:
                replace_rabbit_with_sweep(value, old_name, new_name)
    elif isinstance(config, list):
        for i in range(len(config)):
            if isinstance(config[i], str):
                config[i] = config[i].replace(old_name, new_name)
            else:
                replace_rabbit_with_sweep(config[i], old_name, new_name)


parser = argparse.ArgumentParser(
    description="Random agent for Isaac Lab environments.")

parser.add_argument(
    "--name_list",
    type=str,
    default=False,
)
parser.add_argument(
    "--env_config",
    type=str,
    default=None,
)

args_cli = parser.parse_args()

with open(args_cli.env_config, 'r') as file:
    # Load the contents of the file
    config = yaml.safe_load(file)

import pdb

# Replace 'rabbit' with 'sweep'
random_params = args_cli.name_list.strip('[]').split(',')
old_name = "rabbit"
for new_name in random_params:
    replace_rabbit_with_sweep(config, old_name, new_name)
    with open(f"source/config/{new_name}_env_random.yaml", 'w') as file:
        yaml.dump(config, file, default_flow_style=False)

    old_name = new_name
