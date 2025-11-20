from scripts.workflows.hand_manipulation.env.rl_env.tdmpc_modem import TDMPC
import yaml


class TDMPCDemoWrapper:

    def __init__(self, env, env_config, args_cli):
        self.env = env
        self.env_config = env_config
        self.args_cli = args_cli
        if self.args_cli.add_right_hand:
            self.hand_side = "right"
        if self.args_cli.add_left_hand:
            self.hand_side = "left"
        # Load from a file
        with open('source/config/rl/hand_manipulation/modem_cfg.yaml',
                  'r') as file:
            config = yaml.safe_load(file)
        self.tdmpc = TDMPC(config)

    def reset(self):
        obs = self.env.reset()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
