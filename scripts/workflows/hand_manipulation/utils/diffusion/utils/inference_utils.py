import torch


def RK2(
    model=None,
    num_inference_steps=0,
    local_cond=None,
    global_cond=None,
    trajectory=None,
):
    for t in range(num_inference_steps):
        t_start = t / num_inference_steps
        t_end = (t + 1) / num_inference_steps
        delta_t = t_end - t_start

        t_start_tensor = torch.ones(trajectory.shape[0]).to(
            trajectory.device) * t_start
        t_mid_tensor = torch.ones(trajectory.shape[0]).to(
            trajectory.device) * (t_start + delta_t / 2)

        # First evaluation
        v0 = model(trajectory,
                   t_start_tensor,
                   local_cond=local_cond,
                   global_cond=global_cond)

        # Midpoint estimation
        x_mid = trajectory + (delta_t / 2) * v0
        v_mid = model(x_mid,
                      t_mid_tensor,
                      local_cond=local_cond,
                      global_cond=global_cond)

        # RK2 update
        trajectory = trajectory + delta_t * v_mid
    return trajectory
