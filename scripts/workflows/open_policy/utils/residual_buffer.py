import sys

sys.path.append("submodule/robomimic_openrt")
import robomimic.utils.file_utils as FileUtils


class ResidualBuffer:

    def __init__(self, obs_policy, command_policy, base_policy):
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=base_policy, device="cuda", verbose=True)
        policy.start_episode()
