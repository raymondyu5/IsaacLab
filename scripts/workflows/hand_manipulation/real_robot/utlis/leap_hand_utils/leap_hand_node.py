import numpy as np

from scripts.workflows.hand_manipulation.real_robot.utlis.leap_hand_utils.dynamixel_client import *
import scripts.workflows.hand_manipulation.real_robot.utlis.leap_hand_utils.leap_hand_utils as lhu
import time
#######################################################
"""This can control and query the LEAP Hand

I recommend you only query when necessary and below 90 samples a second.  Used the combined commands if you can to save time.  Also don't forget about the USB latency settings in the readme.

#Allegro hand conventions:
#0.0 is the all the way out beginning pose, and it goes positive as the fingers close more and more in radians.

#LEAP hand conventions:
#3.14 rad is flat out home pose for the index, middle, ring, finger MCPs.
#Applying a positive angle closes the joints more and more to curl closed in radians.
#The MCP is centered at 3.14 and can move positive or negative to that in radians.

#The joint numbering goes from Index (0-3), Middle(4-7), Ring(8-11) to Thumb(12-15) and from MCP Side, MCP Forward, PIP, DIP for each finger.
#For instance, the MCP Side of Index is ID 0, the MCP Forward of Ring is 9, the DIP of Ring is 11

"""


########################################################
class LeapNode:

    def __init__(self):
        ####Some parameters
        # I recommend you keep the current limit from 350 for the lite, and 550 for the full hand
        # Increase KP if the hand is too weak, decrease if it's jittery.
        self.kP = 600
        self.kI = 0
        self.kD = 50
        self.curr_lim = 350  ##set this to 550 if you are using full motors!!!!
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(
            np.zeros(16))
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        # For example ls /dev/serial/by-id/* to find your LEAP Hand. Then use the result.
        # For example: /dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT7W91VW-if00-port0
        self.motors = motors = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1',
                                                  4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
                self.dxl_client.connect()
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * 5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors,
                                   np.ones(len(motors)) * self.kP, 84,
                                   2)  # Pgain stiffness
        self.dxl_client.sync_write(
            [0, 4, 8],
            np.ones(3) * (self.kP * 0.75), 84,
            2)  # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors,
                                   np.ones(len(motors)) * self.kI, 82,
                                   2)  # Igain
        self.dxl_client.sync_write(motors,
                                   np.ones(len(motors)) * self.kD, 80,
                                   2)  # Dgain damping
        self.dxl_client.sync_write(
            [0, 4, 8],
            np.ones(3) * (self.kD * 0.75), 80,
            2)  # Dgain damping for side to side should be a bit less
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors,
                                   np.ones(len(motors)) * self.curr_lim, 102,
                                   2)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #allegro compatibility joint angles.  It adds 180 to make the fully open position at 0 instead of 180
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Sim compatibility for policies, it assumes the ranges are [-1,1] and then convert to leap hand ranges.
    def set_ones(self, pose):
        pose = lhu.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #read position of the robot
    def read_pos(self):
        return self.dxl_client.read_pos()

    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()

    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()

    #These combined commands are faster FYI and return a list of data
    def pos_vel(self):
        return self.dxl_client.read_pos_vel()

    #These combined commands are faster FYI and return a list of data
    def pos_vel_eff_srv(self):
        return self.dxl_client.read_pos_vel_cur()


#-----------------------------------------------------------------------------------
def LEAPhandsim_order_to_real_order():
    sim_jonts_names = [
        'j1', 'j12', 'j5', 'j9', 'j0', 'j13', 'j4', 'j8', 'j2', 'j14', 'j6',
        'j10', 'j3', 'j15', 'j7', 'j11'
    ]
    real_joints_names = [
        'j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9', 'j10',
        'j11', 'j12', 'j13', 'j14', 'j15'
    ]
    sim2real_index = [
        sim_jonts_names.index(name) for name in real_joints_names
    ]
    real2sim_index = [
        real_joints_names.index(name) for name in sim_jonts_names
    ]
    return np.array(sim2real_index), np.array(real2sim_index)


#echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
if __name__ == "__main__":
    teleop_node = LeapNode()
    print("Teleoperation node initialized.")
    # You can add more functionality here to test the teleoperation.
    # For example, you can start streaming data or control the hand.

    action_buffer = []
    leap_sim2real_order, leap_real2sim_order = LEAPhandsim_order_to_real_order(
    )
    import h5py

    with h5py.File('logs/data_1007/teleop_data/test.hdf5', 'r') as f:
        demo_keys = list(f["data"].keys())

        for demo_key in demo_keys:

            action = np.array(f["data"][demo_key]["actions"])
            action_buffer.append(action)

    for demo_id in range(0, len(action_buffer)):
        for index, act in enumerate(action_buffer[demo_id]):
            start_time = time.time()

            actions = act[-16:][leap_sim2real_order]
            teleop_node.set_allegro(actions)

            # print(1 / (time.time() - start_time))
            time.sleep(0.01)
        teleop_node.set_allegro(np.zeros(16))
