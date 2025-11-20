import sys
import os

sys.path.append("submodule/diffusion_policy")
import dill
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import hydra

import torch
from scripts.workflows.hand_manipulation.utils.diffusion.dataset.pickandplace_image_dataset import PickandPlaceImageDataset
from torch.utils.data import DataLoader
import imageio

import numpy as np


def load_diffusion_model(diffusion_path, device):

    checkpoint = os.path.join(diffusion_path, "checkpoints", f"latest.ckpt")

    payload = torch.load(open(checkpoint, 'rb'), pickle_module=dill)

    cfg = payload['cfg']

    cfg.policy.num_inference_steps = 3
    cfg._target_ = "scripts.workflows.hand_manipulation.utils.diffusion.train_cfm_image_policy.TrainCFMUnetImageWorkspace"
    cls = hydra.utils.get_class(cfg._target_)

    workspace = cls(cfg, )
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # get policy from workspace
    policy = workspace.model
    if cfg.training.use_ema:
        policy = workspace.ema_model

    device = torch.device(device)
    policy.to(device)
    policy.eval()

    return policy, cfg


if __name__ == "__main__":
    diffusion_path = "logs/trash/image_cfm/"
    device = "cuda:0"
    model, cfg = load_diffusion_model(diffusion_path, device)
    import pdb
    pdb.set_trace()
    dino = model.obs_encoder.key_model_map.rgb_0

    dataset = PickandPlaceImageDataset(
        "logs/data_1007/rl_data/grasp/ycb/image",
        load_list=["tomato_soup_can"],
        horizon=1,
        pad_before=0,
        pad_after=0,
        obs_key=['right_ee_pose', 'right_hand_joint_pos'],
        image_key=["rgb_0"],
        action_key='action',
        noise_key=["right_hand_joint_pos"],
        noise_scale=0.05,
        seed=42,
        val_ratio=0.05,
        num_demo=1,
        resize_shape=(64, 64),
        max_train_episodes=None,
        add_randomizer=False,
    )
    train_dataloader = DataLoader(dataset, **cfg.dataloader)

    # --- Evaluation ---
    model.eval()

    for i in range(len(dataset)):

        sample = dataset[i]
        obs = sample["obs"]

        # For each obs_key, unsqueeze(0) to add batch dimension
        obs = {
            k: v.unsqueeze(0) if torch.is_tensor(v) else v
            for k, v in obs.items()
        }

        # Move to device (e.g. CUDA)
        obs = {k: v.to(device) for k, v in obs.items()}

        # Run through the model
        with torch.no_grad():

            pred = model.predict_action(obs)["action_pred"]
            truth = sample["action"]
            mse = torch.nn.functional.mse_loss(pred, truth.to(pred.device))
        print("deviation", mse)

# pred = model.predict_action(dataset[0]["obs"])["action_pred"]
# video = imageio.get_writer("logs/test_video.mp4", fps=30)
# with torch.no_grad():
#     for batch in train_dataloader:
#         # Move to device
#         batch = {
#             k: v.to(device) if torch.is_tensor(v) else v
#             for k, v in batch.items()
#         }

#         # Depending on your policy interface:
#         # Many diffusion_policy models have `.predict_action(batch)` or `.forward(batch)`

#         pred = model.predict_action(batch["obs"])["action_pred"]
#         truth = batch["action"]
#         mse = torch.nn.functional.mse_loss(pred, truth.to(pred.device))

#         images = []
#         for i in range(batch["obs"]["rgb_0"].shape[0]):

#             video.append_data(
#                 (batch["obs"]["rgb_0"][i][0].cpu().numpy().transpose(
#                     1, 2, 0) * 255).astype(np.uint8))

#         print("Predicted action:", mse)
#         import pdb
#         pdb.set_trace()
#         break  # Only test one batch
