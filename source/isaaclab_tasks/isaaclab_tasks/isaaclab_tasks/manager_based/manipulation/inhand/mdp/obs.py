import torch


def get_contact_obs_func(env, asset_name, sensor_name, hand_side=None):
    """
    Get the observation function for the specified robot and hand type.
    """

    sensor_data = []
    for name in sensor_name:
        sensor = env.scene[f"{hand_side}_{name}_contact"]

        force_data = torch.linalg.norm(sensor._data.net_forces_w, dim=2)
        sensor_data.append(force_data)
    sensor_data = torch.cat(sensor_data, dim=1)

    return (sensor_data > 2.0).int()
