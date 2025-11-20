"""
datasets.py

Lightweight PyTorch Dataset Definition for wrapping RLDS TFDS Pipeline; just defines transform from RLDS default
format to OpenVLA, IterableDataset shim.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, Type

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizerBase
import sys
import h5py

sys.path.append("submodule/openvla")

import json
from prismatic.models.backbones.llm.prompting import PromptBuilder, PurePromptBuilder
from prismatic.models.backbones.vision import ImageTransform
from prismatic.util.data_utils import tree_map
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets.rlds import make_interleaved_dataset, make_single_dataset
from prismatic.vla.datasets.rlds.oxe import OXE_NAMED_MIXTURES, get_oxe_dataset_kwargs_and_weights
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType
from torch.utils.data import DataLoader
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from scripts.workflows.utils.client.openvla_client import resize_image

IGNORE_INDEX = -100


class DummyDataset(Dataset):

    def __init__(self,
                 action_tokenizer: ActionTokenizer,
                 base_tokenizer: PreTrainedTokenizerBase,
                 image_transform: ImageTransform,
                 prompt_builder_fn: Type[PromptBuilder],
                 data_path: str,
                 prompt=None,
                 raw_actions=None,
                 resize_images=None,
                 dataset_statistics=None) -> None:
        self.action_tokenizer = action_tokenizer
        self.base_tokenizer = base_tokenizer
        self.image_transform = image_transform
        self.prompt_builder_fn = prompt_builder_fn
        self.prompt = prompt
        self.data_path = data_path
        self.raw_actions = raw_actions
        self.gs_images = resize_images
        self.dataset_statistics = dataset_statistics
        with open("source/config/task/bridge_kitchen/dataset_statistics.json",
                  "r") as json_file:
            self.statistics_norm_data = json.load(json_file)

        self.statistics_norm_data["bridge_orig"]
        self.norms_actions = self.normalize_actions(self.raw_actions)

        # Note =>> We expect the dataset to store statistics for action de-normalization. Specifically, we store the
        # per-dimension 1st and 99th action quantile. The values below correspond to "no normalization" for simplicity.

    def normalize_actions(self, unnormalized_actions):
        """
        Normalize unnormalized actions to the range [-1, 1].
        
        Args:
            unnormalized_actions (numpy.ndarray or torch.Tensor): The unnormalized action values.
            action_low (float or numpy.ndarray): The lower bound of the action range.
            action_high (float or numpy.ndarray): The upper bound of the action range.

        Returns:
            numpy.ndarray or torch.Tensor: Normalized actions in the range [-1, 1].
        """

        low = self.dataset_statistics["dummy_dataset"]["action"]["q01"]
        high = self.dataset_statistics["dummy_dataset"]["action"]["q99"]

        return np.clip(
            2 * (unnormalized_actions - low) / (high - low + 1e-8) - 1, -1, 1)

    def __len__(self):
        # TODO =>> Replace with number of elements in your dataset!
        return len(self.gs_images)

    def __getitem__(self, idx):
        # TODO =>> Load image, action and instruction from disk -- we use dummy values

        image = Image.fromarray(self.gs_images[idx])
        action = np.asarray(self.norms_actions[idx], dtype=np.float32)
        instruction = self.prompt
        # print("gt actions", self.raw_actions[idx], self.norms_actions[idx])

        # Add instruction to VLA prompt
        prompt_builder = self.prompt_builder_fn("openvla")
        conversation = [
            {
                "from": "human",
                "value": f"What action should the robot take to {instruction}?"
            },
            {
                "from": "gpt",
                "value": self.action_tokenizer(action)
            },
        ]
        for turn in conversation:
            prompt_builder.add_turn(turn["from"], turn["value"])

        # Tokenize (w/ `base_tokenizer`)

        input_ids = self.base_tokenizer(prompt_builder.get_prompt(),
                                        add_special_tokens=True).input_ids
        labels = list(input_ids)

        # Tensorize =>> Run Image Transform to get `pixel_values` =>> Return
        #   =>> IMPORTANT :: IF WE'RE USING HF .forward(..., labels=labels), SHIFTING HAPPENS _INSIDE_ MODEL!
        input_ids, labels = torch.tensor(input_ids), torch.tensor(labels)
        pixel_values = self.image_transform(image)

        # [CRITICAL] We do not want to take the loss for anything but the predicted action tokens!
        labels[:-(len(action) + 1)] = IGNORE_INDEX

        return dict(pixel_values=pixel_values,
                    input_ids=input_ids,
                    labels=labels)


# from transformers import AutoModelForVision2Seq, AutoProcessor

# openvla_path, attn_implementation = "openvla/openvla-7b", "flash_attention_2"
# device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device(
#     "cpu")

# # Load VLA Model using HF AutoClasses
# processor = AutoProcessor.from_pretrained(openvla_path, trust_remote_code=True)
# vla = AutoModelForVision2Seq.from_pretrained(
#     openvla_path,
#     attn_implementation=attn_implementation,
#     torch_dtype=torch.bfloat16,
#     low_cpu_mem_usage=True,
#     trust_remote_code=True,
# ).to(device)
# action_tokenizer = ActionTokenizer(processor.tokenizer)

# vla_dataset = DummyDataset(
#     action_tokenizer,
#     processor.tokenizer,
#     image_transform=processor.image_processor.apply_transform,
#     prompt_builder_fn=PurePromptBuilder,
#     data_path="logs/openvla_eggplant/replay_rl_data.hdf5",
#     prompt="put the eggplant into the sink")

# collator = PaddedCollatorForActionPrediction(
#     processor.tokenizer.model_max_length,
#     processor.tokenizer.pad_token_id,
#     padding_side="right")

# dataloader = DataLoader(
#     vla_dataset,
#     batch_size=10,
#     sampler=None,
#     collate_fn=collator,
#     num_workers=0,
# )
# for batch_idx, batch in enumerate(dataloader):
#     with torch.autocast("cuda", dtype=torch.bfloat16):

#         import pdb
#         pdb.set_trace()
