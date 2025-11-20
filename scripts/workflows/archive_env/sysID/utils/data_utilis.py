import torch
import sys

sys.path.append(".")
import cv2

import gc
import numpy as np
import torchvision.transforms as transforms
from numpy.random import choice

import wandb

import weakref


def initialize_buffers(env,
                       obs,
                       device,
                       exclude_obs_keys=[],
                       target_object_seg_id=None,
                       require_segmentation=True):
    obs_keys = [*obs[0]["policy"].keys()]

    # remove the whole point cloud and segmentation point cloud
    for name in exclude_obs_keys:
        if name in obs_keys:
            obs_keys.remove(name)

    return DataBuffer(env,
                      device,
                      obs_keys,
                      target_object_seg_id=target_object_seg_id,
                      require_segmentation=require_segmentation)


class DataBuffer:

    def __init__(self,
                 env,
                 device,
                 obs_key,
                 save_next_obs=False,
                 target_object_seg_id=None,
                 require_segmentation=True):
        """
        Initializes the DataBuffer with separate buffers for target and training data.

        Args:
            env: The environment from which data will be collected.
            device: The device (CPU or GPU) where the data will be stored.
            obs_key: A list of observation keys that need to be stored in the buffer.
        """
        self.env = weakref.proxy(env)  # Use weakref for env
        self.device = device
        self.obs_key = obs_key
        self.save_next_obs = save_next_obs
        self.target_object_seg_id = target_object_seg_id
        self.require_segmentation = require_segmentation

        self.train_buffer = self._initialize_empty_buffer()
        self.eval_buffer = self._initialize_empty_buffer()
        self.target_buffer = self._initialize_empty_buffer()

        self.train_cache = self._initialize_empty_buffer()
        self.eval_cache = self._initialize_empty_buffer()
        self.target_cache = self._initialize_empty_buffer()

        # Create a dictionary of handlers using weak references
        self._handle = {
            'store_transition':
            lambda *args: weakref.proxy(self)._store_transition(*args),
            'clear_cache':
            lambda *args: weakref.proxy(self)._clear_cache(*args),
            'cache_traj':
            lambda *args: weakref.proxy(self)._cache_traj(*args),
            'get_buffer':
            lambda *args: weakref.proxy(self)._get_buffer(*args),
            'clear_buffer':
            lambda *args: weakref.proxy(self)._clear_buffer(*args),
            'seg_pc_transition':
            lambda *args: weakref.proxy(self)._seg_pc_transition(*args),
        }

    def _initialize_empty_buffer(self):
        buffer = {}
        for key in self.obs_key:
            buffer[key] = []
            if self.save_next_obs:
                buffer[f'next_{key}'] = []
        buffer['actions'] = []
        buffer['rewards'] = []
        buffer['dones'] = []
        return buffer

    def _resort_buffer(self, buffer, cache, key, cache_type):

        if cache_type == 'train':

            cache = torch.stack(
                cache)  #(num_interaction,num_rollout,num_env,dim)

            cache = cache.permute(1, 2, 0, *range(3, cache.ndimension()))
            cache = cache.contiguous().view(-1, *cache.shape[2:])

        else:
            cache = torch.stack(cache)

            cache = cache.contiguous().view(1, -1, *cache.shape[3:])

        if key == 'seg_pc' and self.require_segmentation:

            cache = self._seg_pc_transition(cache)

        else:
            cache = cache.to(self.device)
        assert len(buffer) == 0

        buffer = cache

        del cache
        gc.collect()
        return buffer

    def _store_transition(self, cache_type='train'):
        """
        Stores a single transition in the specified buffer by concatenating it with the existing data.
        """
        cache = self.train_cache if cache_type == 'train' else self.target_cache if cache_type == "target" else self.eval_cache
        buffer = self.train_buffer if cache_type == 'train' else self.target_buffer if cache_type == "target" else self.eval_buffer

        for key in self.obs_key:
            buffer[key] = self._resort_buffer(buffer[key], cache[key], key,
                                              cache_type)
            if self.save_next_obs:
                buffer[f'next_{key}'] = self._resort_buffer(
                    buffer[f'next_{key}'], cache[key], key, cache_type)

        del cache
        torch.cuda.empty_cache()
        gc.collect()

    def _clear_cache(self, cache_type='train'):
        if cache_type == 'train':
            del self.train_cache
            self.train_cache = self._initialize_empty_buffer()

        elif cache_type == 'eval':
            del self.eval_cache
            self.eval_cache = self._initialize_empty_buffer()
        else:
            del self.target_cache
            self.target_cache = self._initialize_empty_buffer()
        gc.collect()
        torch.cuda.empty_cache()

    def _cache_transition(self,
                          transition,
                          next_obs,
                          target_count,
                          reset=False):

        if "obs" not in transition.keys():
            transition["obs"] = {}
        for key in self.obs_key:
            if reset:
                if key not in transition["obs"].keys():
                    transition["obs"][key] = []

                transition["obs"][key].append([])

            last_cache = transition["obs"][key][-1]
            if len(last_cache) > 0:

                append_cache = torch.cat([
                    last_cache,
                    next_obs["policy"][key][:target_count].unsqueeze(1)
                ],
                                         dim=1)
                transition["obs"][key][-1] = append_cache
            else:
                transition["obs"][key][-1] = next_obs["policy"][
                    key][:target_count].unsqueeze(1)

        return transition

    def _cache_traj(self,
                    transition,
                    next_obs,
                    reward,
                    actions,
                    target_count,
                    cache_type='train'):

        if "obs" not in transition.keys():
            transition["obs"] = {}

        for key in self.obs_key:
            if key not in transition["obs"].keys():
                transition["obs"][key] = []

            transition["obs"][key].append(
                next_obs["policy"][key][:target_count])

        return transition

    def _get_buffer(self, cache_type='train'):
        """
        Returns the specified buffer, converting lists to tensors.
        """
        buffer = self.train_buffer if cache_type == 'train' else self.target_buffer if cache_type == "target" else self.eval_buffer
        tensor_buffer = {}

        for key in buffer.keys():
            tensor_buffer[key] = torch.stack(
                buffer[key], dim=0) if buffer[key] else torch.tensor(
                    [], device=self.device)

        gc.collect()
        return tensor_buffer

    def _clear_buffer(self, buffer_type='train'):
        """
        Clears the specified buffer by resetting it to an empty state.
        """
        if buffer_type == 'train':
            del self.train_buffer
            self.train_buffer = self._initialize_empty_buffer()
        elif buffer_type == 'target':
            del self.target_buffer
            self.target_buffer = self._initialize_empty_buffer()
        elif buffer_type == 'eval':
            del self.eval_buffer
            self.eval_buffer = self._initialize_empty_buffer()

        torch.cuda.empty_cache()
        gc.collect()

    def _seg_pc_transition(self, seg_pc, num_samples=1024):
        num_env, num_exploration_action, seq, num_pc, dim = seg_pc.size()

        if num_env == 1:  # target or eval
            repeat_target_object_seg_id = self.target_object_seg_id[:num_exploration_action].clone(
            ).view(1, -1).repeat_interleave(seq).expand(num_env, -1)
        else:

            repeat_target_object_seg_id = self.target_object_seg_id.clone(
            ).view(-1).repeat_interleave(
                seq, -1).repeat_interleave(num_exploration_action).repeat(
                    int(num_env / len(self.target_object_seg_id)))
            # repeat_target_object_seg_id = self.target_object_seg_id.clone(
            # ).view(-1).repeat_interleave(
            #     num_exploration_action * int(num_env / len(self.target_object_seg_id)))

        flatten_seg_pc = seg_pc.reshape(num_env * num_exploration_action * seq,
                                        num_pc, dim)

        pc_mask = flatten_seg_pc[...,
                                 -1] == repeat_target_object_seg_id.reshape(
                                     -1).unsqueeze(1).to(self.device)

        if not pc_mask.any(dim=1).all():

            # import open3d as o3d
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(
                flatten_seg_pc[1, :, :3].cpu().numpy())
            coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
                size=0.2, origin=[0.0, 0, 0])
            o3d.visualization.draw_geometries([pcd, coordinate_frame])
            import pdb
            pdb.set_trace()
            assert False, "No sample found"
        sample_pc_indices = torch.multinomial(pc_mask.float(),
                                              num_samples=num_samples,
                                              replacement=True)
        sample_pc_indices_expanded = sample_pc_indices.unsqueeze(-1).expand(
            -1, -1, dim)
        seg_pc_sampled = flatten_seg_pc.gather(1, sample_pc_indices_expanded)
        seg_pc_sampled = seg_pc_sampled.reshape(num_env,
                                                num_exploration_action, seq,
                                                num_samples, dim)

        del seg_pc, pc_mask, sample_pc_indices, sample_pc_indices_expanded, repeat_target_object_seg_id, flatten_seg_pc, num_exploration_action, num_env, num_pc, dim, num_samples
        torch.cuda.empty_cache()
        gc.collect()
        return seg_pc_sampled

    def create_transitions(self, transition, cache_type=None):

        buffer = self.train_cache if cache_type == 'train' else self.target_cache if cache_type == "target" else self.eval_cache

        for key in self.obs_key:

            stack_obs = torch.stack(transition["obs"][key])

            # stack_obs = stack_obs.permute(1, 0,
            #                               *range(2, stack_obs.ndimension()))
            buffer[key].append(stack_obs.to(self.device))
            if self.save_next_obs:
                buffer[f'next_{key}'].append(transition['next_obs'][key].to(
                    self.device))
        del transition
        torch.cuda.empty_cache()

    def __del__(self):
        # Unsubscribe handlers and clean up
        self._handle = None
        gc.collect()
        print(
            "DataBuffer object is being deleted, and handlers are unsubscribed."
        )
