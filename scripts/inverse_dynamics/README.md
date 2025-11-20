# Inverse Dynamics Training

Train inverse dynamics models from Zarr trajectory data (low-dimensional observations only).

## Structure

```
inverse_dynamics/
├── train_id.py                  # Training script
├── configs/
│   └── tomato_soup_can.yaml     # Example config
└── lib/
    ├── zarr_dataset.py          # Zarr data loader
    ├── models.py                # Model architectures
    └── training.py              # Training utilities
```

## Quick Start

```bash
./isaaclab.sh -p scripts/inverse_dynamics/train_id.py \
    --config scripts/inverse_dynamics/configs/tomato_soup_can.yaml \
    --output_dir logs/inverse_dynamics/test \
    --wandb_mode disabled
```

## Data Format

Expects Zarr episodes in the following structure:
```
data_path/
├── episode_0.zarr/
│   └── data/
│       ├── actions/              # (T, action_dim)
│       ├── right_hand_joint_pos/ # (T, 16)
│       ├── right_ee_pose/        # (T, 7)
│       └── ...
├── episode_1.zarr/
└── ...
```

## Configuration

Edit `configs/tomato_soup_can.yaml`:

```yaml
model:
  _target_: scripts.inverse_dynamics.lib.models.InverseDynamicsMLP
  hidden_dims: [256, 256, 256]

data:
  data_path: logs/rl_data_collection/ycb/pcd_residual_chunk/tomato_soup_can
  obs_keys:
    - right_hand_joint_pos
    - right_ee_pose
  num_episodes: 1000

training:
  num_epochs: 100
  batch_size: 256
  learning_rate: 1.0e-4
```

## What's New?

This is adapted from `policy_translation/IsaacLab/scripts/inverse_dynamics/train_id.py`:
- **Zarr support**: Loads data from individual `episode_N.zarr` directories
- **Same training**: Uses identical training loop and model architectures
- **Low-dim only**: No point cloud support (not needed for inverse dynamics)
