from isaaclab.app import AppLauncher
from argparse import ArgumentParser
import sys

parser = ArgumentParser()

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()

args_cli.headless = True

# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

def main():
    print('hi')
    breakpoint()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
