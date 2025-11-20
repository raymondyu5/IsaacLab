import wandb
import torch
import os

def setup_wandb(parser_config, exp_name, tags=None, project="cem"):
    run = wandb.init(
        project=project,
        name=exp_name,
        config=parser_config,
        monitor_gym=True,
        save_code=False,  # optional
        tags=tags,
        entity="entongsu")
    return run


def log_media(target_explore_actions,
              buffer,
              log_path,
              num_loop,
              log_type="target"):

    media_dict = {}
    seg_pc_batch = buffer.target_buffer["seg_pc"][..., :6]

    crop_seg_pc_batch = seg_pc_batch[:min(target_explore_actions,
                                          len(seg_pc_batch))][:, 0].cpu()
    y_offsets = torch.arange(len(crop_seg_pc_batch)).view(-1, 1) * 0.5
    crop_seg_pc_batch[:, :, 1] += y_offsets

    point_cloud_data = wandb.Object3D(crop_seg_pc_batch.numpy().reshape(-1, 6))
    media_dict[f"{log_type}_point_cloud"] = point_cloud_data

    if os.path.exists(f"{log_path}/{log_type}/video/loop_{num_loop}.mp4"):
        # Collect your video data
        video_data = wandb.Video(
            f"{log_path}/{log_type}/video/loop_{num_loop}.mp4",
            fps=4,
            format="gif")
        media_dict[f"{log_type}_video"] = video_data
        del video_data

    # Log all the collected media in one block
    wandb.log(media_dict, step=num_loop)
    del seg_pc_batch, media_dict, point_cloud_data, crop_seg_pc_batch, y_offsets
    torch.cuda.empty_cache()


def log_paramas(params_nam, real_data, predicted_data, init_mean, num_loop):

    if real_data[0].shape != ():
        real_data = real_data[0]
        predicted_data = predicted_data[0]
        init_mean = init_mean[0]

    for index, parmam in enumerate(params_nam):

        wandb.log(
            {
                f"real_{parmam}": real_data[index],
                f"predicted_{parmam}": predicted_data[index],
                f"init_mean_{parmam}": init_mean[index]
            },
            step=num_loop)
