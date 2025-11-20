import socket


def send_command_to_vision_pro(ip_address: str, port: int, command: str):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ip_address, port))
            s.sendall(command.encode('utf-8'))
            print(f"✅ Sent command to Vision Pro: {command}")
    except Exception as e:
        print(f"❌ Failed to send command: {e}")


# Replace this with the actual IP of the Vision Pro
avp_ip = "10.0.0.160"
port = 8888

# Example usage
send_command_to_vision_pro(avp_ip, port, "start")
send_command_to_vision_pro(avp_ip, port, "replay")
send_command_to_vision_pro(avp_ip, port, "stop")

from avp_stream import VisionProStreamer

s = VisionProStreamer(ip=avp_ip, record=True)

while True:
    r = s.latest
    print(r['head'])
