import zmq
import json
import pickle
import threading


class ManoClient:

    def __init__(
        self,
        port=1024,
        host="localhost",
    ):

        # Socket config
        self.ctx = zmq.Context()
        if host == "localhost":
            sub_bind_to = f"tcp://localhost:{port}"
        else:
            sub_bind_to = f"tcp://{host}:{port}"
        self.sub_bind_to = sub_bind_to

        # Thread-safe shared data
        self._lock = threading.Lock()
        self.latest_info = None

        self._shared_server_started = True

        # Start background thread
        self._threads = []
        thread = threading.Thread(target=self.run, args=(0, ))
        thread.daemon = True
        thread.start()
        self._threads.append(thread)
        self.init_caliberate = False

    def update_cmd(self, msg):
        with self._lock:
            try:
                # Deserialize the message
                data = json.loads(msg[0].decode('utf-8'))
                # Update the latest info
                self.latest_info = data
            except Exception as e:
                print(f"❌ Error deserializing message: {e}")

    def run(self, thread_id=0):
        print(f"🧵 Thread {thread_id} listening on {self.sub_bind_to}")
        sub_socket = self.ctx.socket(zmq.SUB)
        sub_socket.setsockopt(zmq.SUBSCRIBE, b"")
        sub_socket.connect(self.sub_bind_to)

        while True:
            try:
                msg = sub_socket.recv()
                self.update_cmd([msg])
            except Exception as e:
                print(f"❌ Error in thread {thread_id}: {e}")

    def get_latest_info(self):
        with self._lock:
            if self.latest_info is None:
                return None

            joints = self.latest_info["joint"]
            if joints[0] is None:
                return None

            return joints


if __name__ == "__main__":
    client = ManoClient()
    while True:
        data = client.get_latest_info()
        print(data)
