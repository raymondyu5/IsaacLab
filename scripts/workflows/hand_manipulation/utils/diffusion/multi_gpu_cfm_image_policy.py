import os
import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader, DistributedSampler
import copy
import numpy as np
import random
import wandb
import tqdm
import argparse
import sys

sys.path.append("submodule/diffusion_policy")
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_lowdim_policy import DiffusionUnetLowdimPolicy
from diffusion_policy.env_runner.base_lowdim_runner import BaseLowdimRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from diffusers.training_utils import EMAModel

OmegaConf.register_new_resolver("eval", eval, replace=True)


def setup_ddp():
    """Initialize distributed training if possible."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        print(f"[Rank {rank}] DDP initialized with world size {world_size}.")
        return True, rank, world_size, local_rank
    else:
        print("Running in single-GPU mode (no DDP).")
        return False, 0, 1, 0


class TrainCFMUnetImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']

    def __init__(self, cfg: OmegaConf, args_cli=None, rank=0, world_size=1):
        if args_cli is not None:
            if cfg.policy.obs_encoder.freeze_backbone:
                output_dir = f"{args_cli.log_dir}/cfm/image_cfm/horizon_{cfg.horizon}_nobs_{cfg.n_obs_steps}/freeze_{cfg.policy.obs_encoder.rgb_model.name}"
            else:
                output_dir = f"{args_cli.log_dir}/cfm/image_cfm/horizon_{cfg.horizon}_nobs_{cfg.n_obs_steps}/finetune_{cfg.policy.obs_encoder.rgb_model.name}"
        else:
            output_dir = "logs/trash"

        # Only rank 0 creates folders
        if rank == 0:
            os.makedirs(output_dir, exist_ok=True)
            OmegaConf.save(cfg, os.path.join(output_dir, "config.yaml"))

        # Dataset setup
        if args_cli is not None:
            cfg.dataset.data_path = args_cli.data_path
            cfg.dataset.load_list = args_cli.load_list
            self.dataset = hydra.utils.instantiate(cfg.dataset)
            cfg.shape_meta.action.shape = [self.dataset.action_dim]
            cfg.shape_meta.obs.agent_pos.shape = [self.dataset.low_obs_dim]

        super().__init__(cfg, output_dir=output_dir)
        self.rank = rank
        self.world_size = world_size
        self.args_cli = args_cli

        # Set seed per rank
        seed = cfg.training.seed + rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Model setup
        self.model: DiffusionUnetLowdimPolicy = hydra.utils.instantiate(cfg.policy)
        self.ema_model: DiffusionUnetLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # Optimizer
        self.optimizer = hydra.utils.instantiate(cfg.optimizer, params=self.model.parameters())
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # DDP setup
        is_ddp, rank, world_size, local_rank = setup_ddp()
        self.rank = rank

        # Move model to GPU
        self.model.to(device)
        if cfg.training.use_ema and self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # Wrap model in DDP
        if is_ddp:
            self.model = DDP(self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
            if self.ema_model is not None:
                self.ema_model = DDP(self.ema_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

        # Load checkpoint if available
        if cfg.training.resume and rank == 0:
            last_ckpt = self.get_checkpoint_path()
            if last_ckpt.is_file():
                print(f"[Rank {rank}] Resuming from {last_ckpt}")
                self.load_checkpoint(path=last_ckpt)

        # Datasets and samplers
        train_sampler = DistributedSampler(self.dataset, num_replicas=world_size, rank=rank, shuffle=True) if is_ddp else None
        train_loader_kwargs = dict(cfg.dataloader)
        if train_sampler is not None:
            train_loader_kwargs.pop("shuffle", None)

        train_dataloader = DataLoader(self.dataset, sampler=train_sampler, **train_loader_kwargs)
        normalizer = self.dataset.get_normalizer()
        val_dataset = self.dataset.get_validation_dataset()
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_ddp else None
        val_loader_kwargs = dict(cfg.val_dataloader)
        if val_sampler is not None:
            val_loader_kwargs.pop("shuffle", None)

        val_dataloader = DataLoader(val_dataset, sampler=val_sampler, **val_loader_kwargs)


        # Normalizer
        self.model.module.set_normalizer(normalizer) if is_ddp else self.model.set_normalizer(normalizer)
        if cfg.training.use_ema and self.ema_model is not None:
            (self.ema_model.module if is_ddp else self.ema_model).set_normalizer(normalizer)

        # Scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(len(train_dataloader) * cfg.training.num_epochs) // cfg.training.gradient_accumulate_every,
            last_epoch=self.global_step - 1,
        )

        # EMA
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(cfg.ema, model=(self.ema_model.module if is_ddp else self.ema_model))

        # Logging only on rank 0
        if rank == 0:
            try:
                wandb_run = wandb.init(dir=str(self.output_dir), config=OmegaConf.to_container(cfg, resolve=True), **cfg.logging)
                wandb.config.update({"output_dir": self.output_dir})
                self.log_wandb = True
            except Exception as e:
                print(f"W&B init failed: {e}")
                self.log_wandb = False
            topk_manager = TopKCheckpointManager(save_dir=os.path.join(self.output_dir, "checkpoints"), **cfg.checkpoint.topk)
            json_logger = JsonLogger(os.path.join(self.output_dir, "logs.json.txt"))
        else:
            self.log_wandb = False
            topk_manager = None
            json_logger = None
        import pdb
        
        pdb.set_trace()

        # Training loop
        for epoch in range(cfg.training.num_epochs):
            if is_ddp:
                train_sampler.set_epoch(epoch)
            train_losses = []

            with tqdm.tqdm(train_dataloader, disable=(rank != 0), desc=f"[Rank {rank}] Train epoch {epoch}") as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                   
                    loss = self.model.module.compute_loss(batch) if is_ddp else self.model.compute_loss(batch)
                    loss = loss / cfg.training.gradient_accumulate_every
                    loss.backward()

                    if self.global_step % cfg.training.gradient_accumulate_every == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    if cfg.training.use_ema:
                        ema.step(self.model.module if is_ddp else self.model)

                    train_losses.append(loss.item())
                    self.global_step += 1

            avg_loss = np.mean(train_losses)
            if rank == 0:
                step_log = {"train_loss": avg_loss, "epoch": epoch, "global_step": self.global_step}
                if self.log_wandb:
                    wandb_run.log(step_log, step=self.global_step)
                # if rank == 0 and json_logger is not None:
                #     json_logger.log(step_log)
                self.save_checkpoint()

            if is_ddp:
                dist.barrier()

        # if rank == 0:
        #     if rank == 0 and json_logger is not None:
        #         json_logger.close()
        if is_ddp:
            dist.destroy_process_group()


def main(args_cli):
    cfg = OmegaConf.load(args_cli.config_file)
    cfg.horizon = args_cli.horizon
    cfg.n_obs_steps = args_cli.nobs
    cfg.n_action_steps = args_cli.naction
    cfg.policy.num_inference_steps = args_cli.num_inference
    cfg.dataset.num_demo = args_cli.num_demo
    cfg.policy.obs_encoder.freeze_backbone = args_cli.freeze_backbone

    workspace = TrainCFMUnetImageWorkspace(cfg, args_cli)
    workspace.run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str)
    parser.add_argument("--config_file", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--load_list", nargs="+", default=None)
    parser.add_argument("--vae_path", type=str, default=None)
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--nobs", type=int, default=1)
    parser.add_argument("--naction", type=int, default=8)
    parser.add_argument("--num_inference", type=int, default=5)
    parser.add_argument("--num_demo", type=int, default=10)
    parser.add_argument("--freeze_backbone", action="store_true")
    args_cli, _ = parser.parse_known_args()
    main(args_cli)
