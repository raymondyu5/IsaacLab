import argparse
import traceback

parser = argparse.ArgumentParser(description="Real robot teleoperation.")
parser.add_argument(
    "--host_ip",
    type=str,
    default="0.0.0.0",
    help="IP address for the teleoperation server to listen on.",
)

parser.add_argument(
    "--port",
    type=int,
    default=8888,
    help="Port for the teleoperation server to listen on.",
)
parser.add_argument(
    "--teleop_user",
    type=str,
    default="Entong",
)
parser.add_argument(
    "--add_right_hand",
    action="store_true",
)
parser.add_argument(
    "--add_left_hand",
    action="store_true",
)

parser.add_argument(
    "--send_command_to_robot",
    action="store_true",
)
parser.add_argument("--avp_address", type=str, default="192.168.0.50")
args_cli = parser.parse_args()
from scripts.workflows.hand_manipulation.real_robot.teleoperation.vision_pro_track.real_teleoperation_wrapper import RealTeleoperationWrapper


def main():

    teleop_wrapper = RealTeleoperationWrapper(args_cli=args_cli)

    try:
        while True:
            teleop_wrapper.run()
    except Exception as e:
        print("❌ Exception occurred:")
        traceback.print_exc(
        )  # 🔍 shows full traceback with file, line number, and code
        teleop_wrapper.teleop_server.send_command("Need to debug")

    except KeyboardInterrupt:

        print("🛑 Teleoperation stopped by user.")
        teleop_wrapper.teleop_server.send_command(
            "Teleoperation stopped by user")


if __name__ == "__main__":
    # run the main function
    main()
