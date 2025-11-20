import numpy as np
import gymnasium as gym


def construct_space(self):
    if isinstance(self.observation_space,
                  gym.spaces.Dict) and not self.args_cli.use_visual_obs:
        keys_to_exclude = ["seg_pc", "rgb"]

        obs_dim = 0
        for key, value in self.observation_space.spaces["policy"].items():
            if key in keys_to_exclude:
                continue
            obs_dim += value.shape[-1]

        observation_space = gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=(int(obs_dim), ),  # Ensure shape is a tuple
            dtype=np.float32)
    elif isinstance(self.observation_space,
                    gym.spaces.Dict) and self.args_cli.use_visual_obs:

        num_envs = self.env.num_envs  # Adjust based on your setting
        raw_observation_space = self.observation_space["policy"]
        observation_space = {}
        for key, value in raw_observation_space.spaces.items():
            assert isinstance(value, gym.Space), f"{key} is not a gym.Space"
            assert hasattr(value, "shape"), f"{key} has no shape attribute"

            if "rgb" in key:
                observation_space[key] = gym.spaces.Box(
                    low=0, high=255, shape=(value.shape[1:]), dtype=np.uint8)
            else:

                observation_space[key] = gym.spaces.Box(
                    -np.inf, np.inf, shape=(value.shape[1:]), dtype=np.float32)

            # Now wrap'

        observation_space = gym.spaces.Dict(observation_space)

    else:
        observation_space = self.unwrapped.single_observation_space

    action_space = self.action_space

    # if isinstance(action_space,
    #               gym.spaces.Box) and not action_space.is_bounded("both"):

    action_space = gym.spaces.Box(low=-1,
                                  high=1,
                                  shape=(action_space.shape[-1], ))

    # if self.args_cli.action_framework is not None:
    #     framework_action_space = action_space.shape[
    #         0] - self.num_hand_joints + self.num_finger_actions
    #     action_space = gym.spaces.Box(low=-1,
    #                                   high=1,
    #                                   shape=np.array([framework_action_space]))
    return observation_space, action_space
