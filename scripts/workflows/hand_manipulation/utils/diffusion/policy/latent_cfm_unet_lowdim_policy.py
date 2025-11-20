from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from scripts.workflows.hand_manipulation.utils.diffusion.utils.inference_utils import RK2
import copy


class LatentCFMnetLowdimPolicy(BaseLowdimPolicy):

    def __init__(
            self,
            model: ConditionalUnet1D,
            noise_scheduler: ConditionalFlowMatcher,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_local_cond=False,
            obs_as_global_cond=False,
            pred_action_steps_only=False,
            oa_step_convention=False,
            latent_model=None,
            # inference_method='rk2',
            # parameters passed to step
            **kwargs):
        super().__init__()
        assert not (obs_as_local_cond and obs_as_global_cond)
        if pred_action_steps_only:
            assert obs_as_global_cond

        self.latent_model = latent_model
        self.inference_method = kwargs[
            "inference_method"] if "inference_method" in kwargs else RK2

        self.model = model
        self.noise_scheduler = noise_scheduler
        self.mask_generator = LowdimMaskGenerator(
            action_dim=self.latent_model.latent_dim,
            obs_dim=0 if
            (obs_as_local_cond or obs_as_global_cond) else obs_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False)
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.obs_as_local_cond = obs_as_local_cond
        self.obs_as_global_cond = obs_as_global_cond
        self.pred_action_steps_only = pred_action_steps_only
        self.oa_step_convention = oa_step_convention
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    # ========= inference  ============
    def conditional_sample(
            self,
            condition_data,
            condition_mask,
            local_cond=None,
            global_cond=None,
            generator=None,
            noise=None,
            # keyword arguments to scheduler.step
            **kwargs):
        model = self.model

        if noise is not None:
            trajectory = noise.clone()
        else:

            trajectory = torch.randn(size=condition_data.shape,
                                     dtype=condition_data.dtype,
                                     device=condition_data.device,
                                     generator=generator)

        trajectory = self.inference_method(
            model,
            self.num_inference_steps,
            local_cond=local_cond,
            global_cond=global_cond,
            trajectory=trajectory,
        )

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        return trajectory

    def predict_action(self,
                       obs_dict: Dict[str, torch.Tensor],
                       noise=None) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """

        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict  # not implemented yet

        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        B, _, Do = nobs.shape
        To = self.n_obs_steps
        assert Do == self.obs_dim
        T = self.horizon
        Da = self.latent_model.latent_dim

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_local_cond:
            # condition through local feature
            # all zero except first To timesteps
            local_cond = torch.zeros(size=(B, T, Do),
                                     device=device,
                                     dtype=dtype)
            local_cond[:, :To] = nobs[:, :To]
            shape = (B, T, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        elif self.obs_as_global_cond:
            # condition throught global feature
            global_cond = nobs[:, :To].reshape(nobs.shape[0], -1)
            shape = (B, T, Da)
            if self.pred_action_steps_only:
                shape = (B, self.n_action_steps, Da)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            shape = (B, T, Da + Do)
            cond_data = torch.zeros(size=shape, device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs[:, :To]
            cond_mask[:, :To, Da:] = True
        if noise is not None:
            glob_cond_noise = (torch.rand_like(global_cond) * 2 - 1) * 0.03
        else:
            glob_cond_noise = torch.zeros_like(global_cond)

        # run sampling
        nsample = self.conditional_sample(cond_data,
                                          cond_mask,
                                          local_cond=local_cond,
                                          global_cond=global_cond +
                                          glob_cond_noise,
                                          noise=noise,
                                          **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = naction_pred.clone()
        # action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        if self.pred_action_steps_only:
            action = action_pred
        else:
            start = To
            if self.oa_step_convention:
                start = To - 1
            end = start + self.n_action_steps
            action = action_pred[:, start:end]

        result = {'action': action, 'action_pred': action_pred}
        if not (self.obs_as_local_cond or self.obs_as_global_cond):
            nobs_pred = nsample[..., Da:]
            obs_pred = self.normalizer['obs'].unnormalize(nobs_pred)
            action_obs_pred = obs_pred[:, start:end]
            result['action_obs_pred'] = action_obs_pred
            result['obs_pred'] = obs_pred

        result['action_pred'] = self.latent_model.decode({
            "action":
            result['action_pred'],
            "obs":
            obs_dict['obs']
        })

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        assert 'valid_mask' not in batch

        latent_batch = copy.deepcopy(batch)
        action = self.latent_model.encode(latent_batch)

        # nbatch = self.normalizer.normalize(latent_batch)
        obs = self.normalizer["obs"].normalize(latent_batch["obs"])
        # action = nbatch['action']

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = action
        if self.obs_as_local_cond:
            # zero out observations after n_obs_steps
            local_cond = obs
            local_cond[:, self.n_obs_steps:, :] = 0
        elif self.obs_as_global_cond:
            global_cond = obs[:, :self.n_obs_steps, :].reshape(
                obs.shape[0], -1)
            if self.pred_action_steps_only:
                To = self.n_obs_steps
                start = To
                if self.oa_step_convention:
                    start = To - 1
                end = start + self.n_action_steps
                trajectory = action[:, start:end]
        else:
            trajectory = torch.cat([action, obs], dim=-1)

        # generate impainting mask

        if self.pred_action_steps_only:
            condition_mask = torch.zeros_like(trajectory, dtype=torch.bool)
        else:
            condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images

        noise = torch.randn(trajectory.shape).to(device=trajectory.device)
        bsz = trajectory.shape[0]
        timesteps = torch.rand(bsz, device=trajectory.device)

        timesteps, noisy_trajectory, ut = self.noise_scheduler.sample_location_and_conditional_flow(
            noise, trajectory)

        loss_mask = ~condition_mask

        # Predict the noise residual
        pred = self.model(noisy_trajectory,
                          timesteps.reshape(-1),
                          local_cond=local_cond,
                          global_cond=global_cond.clone())

        target = ut.clone()
        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
