import argparse
import tqdm
import importlib
import os

from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress debug warning messages
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import wandb
import tensorflow as tf
from isaaclab.app import AppLauncher

import gymnasium as gym
import torch

parser.add_argument('dataset_name', help='name of the dataset to visualize')

args_cli = parser.parse_args()

# create TF dataset
dataset_name = args_cli.dataset_name
print(f"Visualizing data from dataset: {dataset_name}")
module = importlib.import_module(dataset_name)
ds = tfds.load(dataset_name, split='train')
import os

os.makedirs('logs/trash_action/', exist_ok=True)
# visualize action and state statistics
actions, states = [], []
index = 0
for episode in tqdm.tqdm(ds.take(500)):
    actions.append([])

    for step in episode['steps']:
        actions[-1].append(step['action'].numpy())
    print(step["language_instruction"])
    np.save(f'logs/trash_action/{index}.npy', np.array(actions[-1]))
    index += 1
