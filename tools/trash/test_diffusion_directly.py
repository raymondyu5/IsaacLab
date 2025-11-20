import sys
import h5py
import numpy as np
import torch


def unnormalize(arr, stats, scale=100):

    min_val, max_val = stats["min"], stats["max"]
    arr = np.array(arr)
    result = 0.5 * (arr + 1) * (max_val - min_val) + min_val
    result[..., :-1] /= scale
    return result


sys.path.append("../robomimic")
import robomimic.utils.file_utils as FileUtils

policy, ckpt_dict = FileUtils.policy_from_checkpoint(
    ckpt_path=
    "/media/lme/data4/weird/robomimic/bc_trained_models/diffusion/20241031094443/models/model_epoch_585_best_validation_0.0011904271435923875.pth",
    device="cuda:0",
    verbose=True)

policy.start_episode()

normalized_grasp = h5py.File(
    f"/media/lme/data4/weird/IsaacLab/logs/103002/grasp_normalized.hdf5", 'r+')
demo_index = 0
gt_actions = torch.as_tensor(
    np.array(normalized_grasp["data"][f"demo_{demo_index}"]["actions"])).to(
        "cuda:0")
obs_buffer = {}
stats = np.load("/media/lme/data4/weird/IsaacLab/logs/103002/grasp_stats.npy",
                allow_pickle=True).item()
while True:
    obs = normalized_grasp["data"][f"demo_{demo_index}"]["obs"]
    for key in obs.keys():
        obs_buffer[key] = torch.cat([
            torch.as_tensor(obs[key][0]).unsqueeze(0),
            torch.as_tensor(obs[key][0]).unsqueeze(0)
        ],
                                    dim=0).to("cuda:0")
    for actions_id in range(len(gt_actions)):
        for key in obs.keys():
            obs_buffer[key][0] = obs_buffer[key][-1].clone()

            obs_buffer[key][-1] = torch.as_tensor(
                obs[key][actions_id]).to("cuda:0").unsqueeze(0)
        import copy
        target_obs = copy.deepcopy(obs_buffer)
        target_obs["seg_pc"] = target_obs["seg_pc"].permute(0, 2, 1)[:, :3, :]
        predicted_action = policy(target_obs)
        import pdb
        pdb.set_trace()
