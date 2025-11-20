from scripts.workflows.hand_manipulation.real_robot.utlis.leap_hand_utils.leap_hand_node import LeapNode
from scripts.workflows.hand_manipulation.real_robot.utlis.leap_hand_utils.teleoperation_utils import LeapvHandBunnyVisionProTeleop
import time


class TeleopLeapNode(LeapNode):

    def __init__(self, use_left=False, use_right=True, AVP_IP="10.0.0.160"):
        super().__init__()
        self.telop_client = LeapvHandBunnyVisionProTeleop(record=True,
                                                          use_left=use_left,
                                                          use_right=use_right,
                                                          AVP_IP=AVP_IP)

    def steam_data(self):
        """
        This method is used to stream data from the VisionPro and control the LEAP Hand.
        """
        hand_pose = self.telop_client.get_avp_data()
        self.set_allegro(hand_pose[-16:])
        print("Position: " + str(self.read_pos()))
        print("target pose: ", hand_pose[-16:])
        time.sleep(0.05)


if __name__ == "__main__":
    teleop_node = TeleopLeapNode(use_left=False, use_right=True)
    print("Teleoperation node initialized.")
    # You can add more functionality here to test the teleoperation.
    # For example, you can start streaming data or control the hand.
    while True:

        teleop_node.steam_data()
