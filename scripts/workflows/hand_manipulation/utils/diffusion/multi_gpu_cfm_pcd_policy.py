import os
import sys
import copy
import random
import contextlib
import collections.abc as cabc

import hydra
import torch
import torch.distributed as dist
import numpy as np
import wandb
import tqdm

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.append("submodule/diffusion_policy")
from diffusion_policy.common.pytorch_util import optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ---------- utils ----------
def move_to_device(obj, device):
    """Recursively move tensors/ndarrays in obj to device."""
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)
    if isinstance(obj, np.ndarray):
        return torch.as_tensor(obj, device=device)
    if isinstance(obj, cabc.Mapping):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        out = [move_to_device(v, device) for v in obj]
        return type(obj)(out) if isinstance(obj, tuple) else out
    
    return obj


def build_loader(dataset, cfg_dl, use_ddp: bool, is_train: bool):
    """Sanitize kwargs, wire DistributedSampler, avoid sampler/shuffle conflicts."""
    dl_kwargs = dict(OmegaConf.to_container(cfg_dl, resolve=True))
    dl_kwargs.pop("shuffle", None)
    dl_kwargs.pop("sampler", None)
    dl_kwargs.pop("batch_sampler", None)  # just in case

    sampler = DistributedSampler(
        dataset, shuffle=is_train, drop_last=is_train
    ) if use_ddp else None

    if sampler is not None:
        return DataLoader(dataset, sampler=sampler, **dl_kwargs)
    else:
        # default: shuffle for train, not for val (unless cfg provided)
        default_shuffle = dl_kwargs.pop("default_shuffle", None)
        if default_shuffle is None:
            default_shuffle = is_train
        return DataLoader(dataset, shuffle=default_shuffle, **dl_kwargs)


# ---------- main workspace ----------
class TrainCFMUnetPCDWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, args_cli=None):
        # Output dir
        if args_cli is not None:
            output_dir = args_cli.log_dir + f"/cfm/pcd_cfm/horizon_{cfg.horizon}_nobs_{cfg.n_obs_steps}"
        else:
            output_dir = "logs/trash"
        os.makedirs(output_dir, exist_ok=True)
        OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

        # Dataset
        if args_cli is not None:
            cfg.dataset.data_path = args_cli.data_path
            cfg.dataset.load_list = args_cli.load_list
            self.dataset = hydra.utils.instantiate(cfg.dataset)
            cfg.shape_meta.action.shape = [self.dataset.action_dim]
            cfg.shape_meta.obs.agent_pos.shape = [self.dataset.low_obs_dim]

        super().__init__(cfg, output_dir=output_dir)
        self.args_cli = args_cli

        # Seeds
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Distributed init
        self.use_ddp = "LOCAL_RANK" in os.environ
        if self.use_ddp:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(self.local_rank)
            dist.init_process_group(backend="nccl", init_method="env://")
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.local_rank = 0
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Model
        self.model: DiffusionUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)
        self.model.to(self.device)
        if self.use_ddp:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                broadcast_buffers=False,
                find_unused_parameters=False
            )

        # EMA (unwrapped)
        self.ema_model = None
        if cfg.training.use_ema:
            base = self.model.module if isinstance(self.model, DDP) else self.model
            self.ema_model = copy.deepcopy(base)
            self.ema_model.to(self.device)

        # Optimizer
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())

        self.global_step = 0
        self.epoch = 0

    def is_main(self):
        return (not self.use_ddp) or (dist.get_rank() == 0)

    def cleanup(self):
        if self.use_ddp and dist.is_initialized():
            dist.destroy_process_group()

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # Resume
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                if self.is_main():
                    print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # Samplers & Loaders
        train_dataloader = build_loader(self.dataset, cfg.dataloader, self.use_ddp, is_train=True)
        val_dataset = self.dataset.get_validation_dataset()
        val_dataloader = build_loader(val_dataset, cfg.val_dataloader, self.use_ddp, is_train=False)

        # Normalizer
        self.model_inner = self.model.module if isinstance(self.model, DDP) else self.model
        normalizer = self.dataset.get_normalizer()
        self.model_inner.set_normalizer(normalizer)

        # LR scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1
        )

        # EMA wrapper
        ema = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=self.ema_model)

        # Logging
        self.log_wandb = False
        if self.is_main():
            try:
                wandb_run = wandb.init(
                    dir=str(self.output_dir),
                    config=OmegaConf.to_container(cfg, resolve=True),
                    **cfg.logging
                )
                wandb.config.update({"output_dir": self.output_dir})
                self.log_wandb = True
            except Exception:
                pass

        # Checkpoint manager
        if self.is_main():
            topk_manager = TopKCheckpointManager(
                save_dir=os.path.join(self.output_dir, 'checkpoints'),
                **cfg.checkpoint.topk
            )

        # Device for optimizer
        optimizer_to(self.optimizer, self.device)
        if self.ema_model is not None:
            self.ema_model.to(self.device)

        # Training
        train_sampling_batch = None
        best_val_loss = float('inf')
        log_path = os.path.join(self.output_dir, 'logs.json.txt')

        with JsonLogger(log_path) as json_logger:
            for _ in range(cfg.training.num_epochs):

                # Make DDP shuffles differ each epoch
                if self.use_ddp:
                    ts = getattr(train_dataloader, "sampler", None)
                    vs = getattr(val_dataloader, "sampler", None)
                    if isinstance(ts, DistributedSampler):
                        ts.set_epoch(self.epoch)
                    if isinstance(vs, DistributedSampler):
                        vs.set_epoch(self.epoch)

                train_losses = []
                iterator = tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}") if self.is_main() else train_dataloader

                for batch_idx, batch in enumerate(iterator):
                    batch = move_to_device(batch, self.device)
                    if train_sampling_batch is None:
                        train_sampling_batch = batch

                    # gradient accumulation (use no_sync for DDP)
                    accum = cfg.training.gradient_accumulate_every
                    for micro_idx in range(accum):
                        with self.model.no_sync() if self.use_ddp and (micro_idx < accum - 1) else contextlib.nullcontext():
                            raw_loss = self.model_inner.compute_loss(batch)
                            (raw_loss / accum).backward()

                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    lr_scheduler.step()

                    if cfg.training.use_ema:
                        ema.step(self.model_inner)

                    loss_val = raw_loss.item()
                    train_losses.append(loss_val)

                    # mid-epoch logging
                    if self.is_main() and not (batch_idx == len(train_dataloader) - 1):
                        step_log = {
                            'train_loss': loss_val,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        if self.log_wandb:
                            wandb.log(step_log, step=self.global_step)
                        json_logger.log(step_log)
                        self.global_step += 1

                # end of epoch
                train_loss_epoch = float(np.mean(train_losses)) if len(train_losses) else 0.0
                step_log = {'train_loss': train_loss_epoch}

                # Validation
                if (self.epoch % cfg.training.val_every) == 0:
                    policy = self.ema_model if (cfg.training.use_ema and self.ema_model is not None) else self.model_inner
                    policy.eval()
                    val_losses = []
                    with torch.no_grad():
                        viter = tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}") if self.is_main() else val_dataloader
                        for batch_idx, batch in enumerate(viter):
                            batch = move_to_device(batch, self.device)
                            loss = self.model_inner.compute_loss(batch)
                            val_losses.append(loss.item())
                            if cfg.training.max_val_steps is not None and batch_idx >= (cfg.training.max_val_steps - 1):
                                break
                    if len(val_losses) > 0:
                        val_loss = float(np.mean(val_losses))
                        step_log['val_loss'] = val_loss
                        if val_loss < best_val_loss and self.is_main():
                            best_val_loss = val_loss
                            self.save_checkpoint(path=os.path.join(self.output_dir, 'checkpoints', 'best_val_loss.ckpt'))

                # Checkpoint
                if self.is_main() and (self.epoch % cfg.training.checkpoint_every) == 0:
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()
                    self.save_checkpoint(path=os.path.join(self.output_dir, 'checkpoints', f'checkpoint_{self.epoch}.ckpt'))

                # epoch logging
                if self.is_main():
                    if self.log_wandb:
                        wandb.log(step_log, step=self.global_step)
                    json_logger.log(step_log)

                self.global_step += 1
                self.epoch += 1


def main(args_cli):
    cfg = OmegaConf.load(args_cli.config_file)
    cfg.horizon = args_cli.horizon
    cfg.n_obs_steps = args_cli.nobs
    cfg.n_action_steps = args_cli.naction
    cfg.policy.num_inference_steps = args_cli.num_inference
    cfg.dataset.num_demo = args_cli.num_demo

    workspace = TrainCFMUnetPCDWorkspace(cfg, args_cli)
    try:
        workspace.run()
    finally:
        workspace.cleanup()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--load_list", nargs='+', default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--nobs", type=int, default=1)
    parser.add_argument("--naction", type=int, default=8)
    parser.add_argument("--num_inference", type=int, default=5)
    parser.add_argument("--num_demo", type=int, default=10)
    args_cli, _ = parser.parse_known_args()
    main(args_cli)
