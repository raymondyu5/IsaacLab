from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
from torchcfm.conditional_flow_matching import *
from torchcfm.utils import *
from torchcfm.models.models import *
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.model.diffusion.mask_generator import LowdimMaskGenerator
from scripts.workflows.hand_manipulation.utils.diffusion.utils.inference_utils import RK2

from diffusion_policy.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from diffusion_policy.common.pytorch_util import dict_apply
# from tools.visualization_utils import vis_pc, visualize_pcd


class CFMnetPCDPolicy(BaseImagePolicy):

    def __init__(
            self,
            shape_meta: dict,
            noise_scheduler: ConditionalFlowMatcher,
            obs_encoder: MultiImageObsEncoder,
            horizon,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=False,
            diffusion_step_embed_dim=256,
            down_dims=(256, 512, 1024),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            n_obs_stack=1,
            n_past_actions=0,
            # parameters passed to step
            **kwargs):
        super().__init__()
        action_shape = shape_meta['action']['shape']
        assert len(action_shape) == 1
        action_dim = action_shape[0]

        down_dims = down_dims[:horizon // 2 + 1]
        # get feature dim
        obs_feature_dim = obs_encoder.output_shape()[0]

        self.noise_scheduler = noise_scheduler

        input_dim = action_dim + obs_feature_dim

        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            # need to cmopute dim  
            if n_obs_stack > 1 or n_past_actions > 0:
                # PCD feature dim (from encoder, single timestep)
                pcd_feature_dim = 0
                for key in obs_encoder.pcd_keys:
                    # dummy input to calc
                    dummy_pcd = torch.zeros((1, 3, shape_meta['obs'][key]['shape'][-1]),
                                           dtype=torch.float32, device=obs_encoder.device)
                    pcd_feature_dim += obs_encoder.key_model_map[key](dummy_pcd).shape[-1]

                # Proprio dim, can just get from shape meta
                proprio_dim = shape_meta['obs']['agent_pos']['shape'][0]

                # Global cond = pcd_features + (proprio * n_obs_stack) + (action_dim * n_past_actions)
                global_cond_dim = pcd_feature_dim + (proprio_dim * n_obs_stack) + (action_dim * n_past_actions)
            else:
                # original - obs_feature_dim * n_obs_steps
                global_cond_dim = obs_feature_dim * n_obs_steps
        self.obs_encoder = obs_encoder
        model = model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale)
        self.model = model

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0 if obs_as_global_cond else obs_feature_dim,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False)
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.n_obs_stack = n_obs_stack
        self.n_past_actions = n_past_actions

        self.obs_as_global_cond = obs_as_global_cond

        self.kwargs = kwargs
        self.inference_method = kwargs[
            "inference_method"] if "inference_method" in kwargs else RK2

        if num_inference_steps is None:
            num_inference_steps = noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def _process_observations_with_stacking(self, nobs, batch_size, n_steps_to_use):
        """
        Process observations with optional stacking for proprio and past actions.

        Args:
            nobs: normalized observations dict with keys: 'agent_pos', 'seg_pc', 'past_action'
            batch_size: batch size
            n_steps_to_use: number of timesteps to extract (n_obs_steps basically)

        Returns:
            global_cond: [B, feature_dim] if obs_as_global_cond
        """
        # Extract point cloud features
        pcd_obs = {}
        for key in self.obs_encoder.pcd_keys:
            if key in nobs:
                # Only use first timestep for point clouds
                pcd_obs[key] = nobs[key][:, 0:1, ...].reshape(-1, *nobs[key].shape[2:])

        # Encode point cloud
        pcd_features = []
        for key in self.obs_encoder.pcd_keys:
            if key in pcd_obs:
                pcd = pcd_obs[key][:, :3]  # only xyz
                pcd = pcd.to(next(self.obs_encoder.parameters()).device, non_blocking=True)
                pcds = torch.cat([pcd], dim=0)
                feature = self.obs_encoder.key_model_map[key](pcds)
                feature = feature.reshape(batch_size, feature.shape[-1])
                pcd_features.append(feature)

        # Stack proprio observations over n_obs_stack timesteps
        proprio_features = []
        if 'agent_pos' in nobs:
            # Extract up to n_obs_stack timesteps of proprio
            n_stack = min(self.n_obs_stack, n_steps_to_use, nobs['agent_pos'].shape[1])
            stacked_proprio = nobs['agent_pos'][:, :n_stack, :]  # [B, n_stack, D_proprio]
            stacked_proprio = stacked_proprio.reshape(batch_size, -1)  # [B, n_stack * D_proprio]
            proprio_features.append(stacked_proprio)

        # Extract past actions
        past_action_features = []
        if self.n_past_actions > 0 and 'past_action' in nobs:
            # Extract past n_past_actions actions
            n_past = min(self.n_past_actions, n_steps_to_use)
            if n_past > 0:
                past_actions = nobs['past_action'][:, :n_past, :]  # [B, n_past, D_action]
                past_actions = past_actions.reshape(batch_size, -1)  # [B, n_past * D_action]
                past_action_features.append(past_actions)

        # concat
        all_features = pcd_features + proprio_features + past_action_features
        global_cond = torch.cat(all_features, dim=-1)

        return global_cond

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
            # if noise is provided, we use it as the initial trajectory
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

        # for t in range(0, self.num_inference_steps):
        #     # 1. apply conditioning
        #     # trajectory[condition_mask] = condition_data[condition_mask]
        #     timesteps = t / self.num_inference_steps

        #     vt = model(trajectory,
        #                t / self.num_inference_steps,
        #                local_cond=local_cond,
        #                global_cond=global_cond)
        #     # trajectory = (vt * 1 / self.num_inference_steps + trajectory)

        #     trajectory = trajectory + timesteps * vt

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

        # normalize input
        nobs = self.normalizer.normalize(obs_dict)

        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        if self.obs_as_global_cond:
            # condition through global feature
            if self.n_obs_stack > 1 or self.n_past_actions > 0:
                global_cond = self._process_observations_with_stacking(
                    nobs, B, To
                )
            else:
                this_nobs = dict_apply(
                    nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
                this_nobs_filtered = {k: v for k, v in this_nobs.items() if k != 'past_action'}
                nobs_features = self.obs_encoder(this_nobs_filtered)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(B, -1)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da),
                                    device=device,
                                    dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        else:
            # condition through impainting
            this_nobs = dict_apply(
                nobs, lambda x: x[:, :To, ...].reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(B, To, -1)
            cond_data = torch.zeros(size=(B, T, Da + Do),
                                    device=device,
                                    dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            cond_data[:, :To, Da:] = nobs_features
            cond_mask[:, :To, Da:] = True

        # run sampling

        # run sampling
        nsample = self.conditional_sample(cond_data,
                                          cond_mask,
                                          local_cond=local_cond,
                                          global_cond=global_cond,
                                          noise=noise,
                                          **self.kwargs)

        # unnormalize prediction
        naction_pred = nsample[..., :Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        result = {'action': action, 'action_pred': action_pred}
        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def compute_loss(self, batch):
        # normalize input
        
        

        assert 'valid_mask' not in batch
        nobs = self.normalizer.normalize(batch['obs'])
        # import pdb
        # pdb.set_trace()
        # o3d = vis_pc(nobs["seg_pc"][10,0].transpose(0, 1).cpu().numpy()[:,:3],nobs["seg_pc"][10,0].transpose(0, 1).cpu().numpy()[:,3:6])
        # visualize_pcd([o3d])

        nactions = self.normalizer['action'].normalize(batch['action'])
        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = nactions.to(batch['action'].device, non_blocking=True)
        cond_data = trajectory
        if self.obs_as_global_cond:
            if self.n_obs_stack > 1 or self.n_past_actions > 0:
                global_cond = self._process_observations_with_stacking(
                    nobs, batch_size, self.n_obs_steps
                )
            else:
                this_nobs = dict_apply(
                    nobs, lambda x: x[:, :self.n_obs_steps, ...].reshape(
                        -1, *x.shape[2:]))
                this_nobs_filtered = {k: v for k, v in this_nobs.items() if k != 'past_action'}
                nobs_features = self.obs_encoder(this_nobs_filtered)
                # reshape back to B, Do
                global_cond = nobs_features.reshape(batch_size, -1)
        else:
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs, lambda x: x.reshape(-1, *x.shape[2:]))
            nobs_features = self.obs_encoder(this_nobs)
            # reshape back to B, T, Do
            nobs_features = nobs_features.reshape(batch_size, horizon, -1)
            cond_data = torch.cat([nactions, nobs_features], dim=-1)
            trajectory = cond_data.detach()
        

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

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
