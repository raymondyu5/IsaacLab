import zmq
import time
from zmq.eventloop import zmqstream
import pickle
import threading

from tornado import ioloop


class TeleopClient:

    def __init__(self, port: int = 6000, host: str = "localhost"):

        self.ctx = zmq.Context()
        if host == "localhost":
            sub_bind_to = f"tcp://localhost:{port}"
        else:
            sub_bind_to = f"tcp://{host}:{port}"
        self.init_bind_to = ":".join(
            sub_bind_to.split(":")[:-1] +
            [str(int(sub_bind_to.split(":")[-1]) + 1)])
        self.sub_bind_to = sub_bind_to
        self.sub_socket = None

        # Multi-thread variable
        self._lock = threading.Lock()
        self.latest_data = {"commands": None, "init_arm_qpos": None}

        # Setup background IO loop
        self._loop = None
        self._started = threading.Event()
        self._stream = None
        self._thread = threading.Thread(target=self.run)
        self._thread.daemon = True
        self._thread.start()

    def update_teleop_cmd(self, message):
        cmd = pickle.loads(message[0])

        self.latest_data = cmd
        return self.latest_data

    def get_teleop_cmd(self):

        with self._lock:
            return self.latest_data

    def run(self):
        self._loop = ioloop.IOLoop()
        self._loop.initialize()
        self._loop.make_current()
        self.sub_socket = self.ctx.socket(zmq.SUB)
        self._stream = zmqstream.ZMQStream(self.sub_socket,
                                           io_loop=ioloop.IOLoop.current())

        # Wait for server start
        self.sub_socket.connect(self.sub_bind_to)
        self.sub_socket.setsockopt(zmq.SUBSCRIBE, b"")

        self._stream.on_recv(self.update_teleop_cmd)
        self._started.set()
        self._loop.start()


if __name__ == "__main__":
    teleop_client = TeleopClient()

    while True:
        start_time = time.time()
        teleop_cmd = teleop_client.get_teleop_cmd()
        print(teleop_cmd)
