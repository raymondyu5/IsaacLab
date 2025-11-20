from dynamixel_sdk import *

portHandler = PortHandler('/dev/ttyUSB0')
packetHandler = PacketHandler(2.0)

if not portHandler.openPort():
    print("Failed to open port")
    exit()
if not portHandler.setBaudRate(
        57600):  # Try other common values: 1000000, 115200
    print("Failed to set baudrate")
    exit()

for motor_id in range(1, 21):
    dxl_model_number, dxl_comm_result, dxl_error = packetHandler.ping(
        portHandler, motor_id)
    if dxl_comm_result == COMM_SUCCESS:
        print(f"Found motor ID {motor_id} (model: {dxl_model_number})")
    else:
        print(f"ID {motor_id} not responding")

portHandler.closePort()
