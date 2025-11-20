import zmq
import json


def run_client(port=1024, host="localhost"):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.SUB)

    # Connect to the server
    socket.connect(f"tcp://{host}:{port}")
    socket.setsockopt(zmq.SUBSCRIBE, b"")

    print(f"📡 Connected to MediaPipeServer at tcp://{host}:{port}")

    while True:
        try:
            msg = socket.recv_string()
            data = json.loads(msg)
            print(data)

        except KeyboardInterrupt:
            print("🛑 Client stopped.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    run_client()
