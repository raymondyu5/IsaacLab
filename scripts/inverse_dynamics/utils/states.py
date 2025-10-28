"""
State extraction utilities for new policy_data format.

Extracts instantaneous states from trajectory datasets collected with
the new recorder that saves both observations and states.
"""

import h5py
import numpy as np
from typing import Tuple


class StateExtractor:
    """
    Extract instantaneous states from trajectory datasets.

    State representation: s_t = [q_t, k_obj,t]
    - q_t: Robot joint positions (dimension auto-detected from data)
    - k_obj,t: Object pose - position + quaternion (7 dims)

    Dimensions are determined from the data itself, making this format-agnostic.
    """

    def __init__(self):
        pass

    def extract_states_from_demo(
        self,
        demo_group: h5py.Group
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract states and actions from a demo with new policy_data format.

        Expected structure:
        - demo['policy_data']['state']: (T, state_dim) instantaneous states
        - demo['actions']: (T, action_dim) actions

        Args:
            demo_group: HDF5 demo group

        Returns:
            states: (T, state_dim) state array
            actions: (T, action_dim) aligned actions
        """
        # Check for new policy_data format
        if 'policy_data' not in demo_group:
            raise KeyError(
                f"Expected 'policy_data' in demo group. "
                f"Found keys: {list(demo_group.keys())}"
            )

        policy_data = demo_group['policy_data']

        if 'state' not in policy_data:
            raise KeyError(
                f"Expected 'state' in policy_data. "
                f"Found keys: {list(policy_data.keys())}"
            )

        # Load states and actions
        states = np.array(policy_data['state'])
        actions = np.array(demo_group['actions'])

        assert states.shape[0] == actions.shape[0], \
            f"States and actions must have same length. " \
            f"Got states: {states.shape[0]}, actions: {actions.shape[0]}"

        return states, actions


# Singleton instance
_extractor = StateExtractor()


def extract_states_from_demo(demo_group: h5py.Group) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to extract states from a demo.

    Args:
        demo_group: HDF5 demo group with policy_data format

    Returns:
        states: (T, state_dim) state array
        actions: (T, action_dim) aligned actions
    """
    return _extractor.extract_states_from_demo(demo_group)
