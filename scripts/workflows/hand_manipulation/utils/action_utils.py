import numpy as np
import gymnasium as gym


def contruction_space(self, env, obs_keys, sqqueeze_first_dim=False):
    concatenate_obs = True
    if getattr(env, "use_last_action", False):
        obs_keys.append("last_action")

    low_dim_obs_dim = 0
    for key, value in self.unwrapped.observation_space.spaces["policy"].items(
    ):
        if key in obs_keys and key not in ["rgb", "seg_pc"]:
            low_dim_obs_dim += value.shape[-1]

    lowdim_observation_space = gym.spaces.Box(
        -np.inf,
        np.inf,
        shape=(int(low_dim_obs_dim), ),  # Ensure shape is a tuple
        dtype=np.float32)

    if "rgb" in obs_keys:

        # if sqqueeze_first_dim:
        #     visual_observation_space = gym.spaces.Box(
        #         low=0,
        #         high=255,
        #         shape=(224, 224, 3),  # Ensure shape is a tuple
        #         dtype=np.uint8)
        # else:

        visual_observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1, 224, 224, 3),  # Ensure shape is a tuple
            dtype=np.uint8)

        observation_space = gym.spaces.Dict({
            "state": lowdim_observation_space,
            # the RGB image
            "rgb": visual_observation_space
        })

        observation_space = gym.spaces.Dict(observation_space)
        concatenate_obs = False
    else:
        observation_space = lowdim_observation_space
        concatenate_obs = True

    # action_space = self.unwrapped.single_action_space
    action_space = self.unwrapped.action_space

    action_space = gym.spaces.Box(low=-1,
                                  high=1,
                                  shape=(action_space.shape[-1], ))

    return concatenate_obs, observation_space, action_space
