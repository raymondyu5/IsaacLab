from tools.trash.amazon import pyspacemouse
import threading
import numpy as np
import time
from typing import Tuple


class SpaceMouseExpert:
    """
    Thread-based SpaceMouse reader.
    Continuously updates the latest state without blocking other code.
    """

    def __init__(self):
        pyspacemouse.open()

        self.action = [0.0] * 6
        self.buttons = [0, 0, 0, 0]
        self._lock = threading.Lock()
        self._running = True

        # Background thread
        self.thread = threading.Thread(target=self._read_spacemouse,
                                       daemon=True)
        self.thread.start()

    def _read_spacemouse(self):
        while self._running:
            state = pyspacemouse.read_all()
            action = [0.0] * 6
            buttons = [0, 0, 0, 0]

            if len(state) == 2:
                action = [
                    -state[0].y, state[0].x, state[0].z, -state[0].roll,
                    -state[0].pitch, -state[0].yaw, -state[1].y, state[1].x,
                    state[1].z, -state[1].roll, -state[1].pitch, -state[1].yaw
                ]
                buttons = state[0].buttons + state[1].buttons
            elif len(state) == 1:
                action = [
                    -state[0].y, state[0].x, state[0].z, -state[0].roll,
                    -state[0].pitch, -state[0].yaw
                ]
                buttons = state[0].buttons

            with self._lock:
                self.action = action
                self.buttons = buttons

            time.sleep(0.01)  # avoid busy loop

    def get_action(self) -> Tuple[np.ndarray, list]:
        with self._lock:
            return np.array(self.action), list(self.buttons)

    def close(self):
        self._running = False
        self.thread.join()
        # pyspacemouse.close()


if __name__ == "__main__":
    sm = SpaceMouseExpert()
    try:
        while True:
            action, buttons = sm.get_action()
            print(f"Action: {action}, Buttons: {buttons}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        sm.close()
