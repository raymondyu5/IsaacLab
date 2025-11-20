import torch


class GrippersEnv:

    def __init__(self):
        if self.gripper_object == []:
            self.has_gripper = False

    def reset_gripper_for_sample_pc(self, *args):

        self.move_gripper_away()
        self.env.scene["deform_object"].remove_attachment(self.num_gripper)

    def reset_gripper(self, explore_type, explore_action_index):
        self.env.scene["deform_object"].remove_attachment(self.num_gripper)
        if explore_type == "train":
            actions = self.plan_trajectories[explore_action_index,
                                             self.env.episode_length_buf[0]]
        elif explore_type == "target" or explore_type == "eval":
            actions = self.plan_trajectories[explore_action_index,
                                             self.env.episode_length_buf[0]]

            actions[:self.
                    num_explore_actions] = self.plan_trajectories[:, self.env.
                                                                  episode_length_buf[
                                                                      0],
                                                                  0].clone()

        for index, gripper in enumerate(self.gripper_object):

            gripper.data.default_root_state[:, :3] = actions[:, index *
                                                             8:index * 8 + 3]
            gripper.data.default_root_state[:, 3:7] = actions[:, index * 8 +
                                                              3:index * 8 + 7]

            target_root_state = gripper.data.default_root_state.clone()

            gripper.reset_default_root_state(target_root_state)

    def move_gripper_away(self):

        for gripper in self.gripper_object:
            target_root_state = gripper.data.default_root_state.clone()

            target_root_state[:, 2] += 1.0
            gripper.data.default_root_state = target_root_state.clone()

            gripper.reset_default_root_state(target_root_state)
