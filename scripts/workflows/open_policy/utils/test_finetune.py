# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Script to run an environment with zero action agent."""
"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher
from scripts.workflows.utils.parse_setting import save_params_to_yaml, parser

AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
"""Rest everything follows."""

import gymnasium as gym
import torch

# from scripts.workflows.utils.client.openvla_client import OpenVLAClient
from scripts.workflows.utils.client.openvla_localclient import OpenVLAClient
from scripts.workflows.open_policy.utils.openvla_dataset import DummyDataset

from transformers import AutoModelForVision2Seq, AutoProcessor
import sys
import h5py
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("submodule/openvla")
from prismatic.models.backbones.llm.prompting import PromptBuilder, PurePromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer

inference_client = OpenVLAClient(args_cli.client_url)
openvla_path, attn_implementation = "openvla/openvla-7b", "flash_attention_2"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
    "cpu")

# Load VLA Model using HF AutoClasses
processor = AutoProcessor.from_pretrained(openvla_path, trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    openvla_path,
    attn_implementation=attn_implementation,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).to(device)
action_tokenizer = ActionTokenizer(processor.tokenizer)

vla_dataset = DummyDataset(
    action_tokenizer,
    processor.tokenizer,
    image_transform=processor.image_processor.apply_transform,
    prompt_builder_fn=PurePromptBuilder,
    data_path="logs/openvla_eggplant/replay_rl_data.hdf5",
    prompt="put the eggplant into the sink")
# vla_dataset.gs_images[10]
# vla_dataset.raw_actions[10]
# response = inference_client.step("pick the eggplant into the sink",
#                                  vla_dataset.gs_images[10],
#                                  unnorm_key="dummy_dataset")
# Assuming `vla_dataset.raw_actions` is your data
columns = [f"Dim_{i+1}" for i in range(7)]  # Dummy column names
df = pd.DataFrame(vla_dataset.raw_actions, columns=columns)

# Create a grid for the histograms
fig, axes = plt.subplots(
    2, 4, figsize=(16, 8))  # 2 rows, 4 columns (extra space for alignment)
axes = axes.flatten()  # Flatten axes for easier iteration

# Plot histograms for each dimension
for i, column in enumerate(df.columns):
    ax = axes[i]
    ax.hist(df[column], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.set_title(f"Distribution of {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    ax.grid(axis='y', linestyle='--', alpha=0.7)

# Hide any unused subplots (in case of mismatch between rows/columns and dimensions)
for j in range(len(df.columns), len(axes)):
    fig.delaxes(axes[j])

# Adjust layout and save
plt.tight_layout()
plt.savefig("distribution_histograms.png")  # Save as PNG
plt.show()
