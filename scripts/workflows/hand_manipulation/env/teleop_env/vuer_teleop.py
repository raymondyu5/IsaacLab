import yaml
from multiprocessing import Array, Process, shared_memory, Queue, Manager, Event, Semaphore

import numpy as np

from scripts.workflows.hand_manipulation.env.teleop_env.reprocessor_vuer import VuerPreprocessor

from scripts.workflows.hand_manipulation.env.teleop_env.OpenTeleVision import OpenTeleVision
import h5py


class VuerTeleop:

    def __init__(
        self,
        env,
        args_cli,
        config_file_path="scripts/workflows/hand_manipulation/utils/dex_retargeting/configs/opentelevision/leap_hand.yml"
    ):

        self.env = env
        self.args_cli = args_cli
        self.resolution = (720, 1280)
        self.crop_size_w = 0
        self.crop_size_h = 0
        self.resolution_cropped = (self.resolution[0] - self.crop_size_h,
                                   self.resolution[1] - 2 * self.crop_size_w)

        self.img_shape = (self.resolution_cropped[0],
                          2 * self.resolution_cropped[1], 3)
        self.img_height, self.img_width = self.resolution_cropped[:2]

        self.shm = shared_memory.SharedMemory(create=True,
                                              size=np.prod(self.img_shape) *
                                              np.uint8().itemsize)
        self.img_array = np.ndarray((self.img_shape[0], self.img_shape[1], 3),
                                    dtype=np.uint8,
                                    buffer=self.shm.buf)
        image_queue = Queue()
        toggle_streaming = Event()
        self.processor = VuerPreprocessor()
        self.tv = OpenTeleVision(self.resolution_cropped,
                                 self.shm.name,
                                 image_queue,
                                 toggle_streaming,
                                 cert_file='../cer/cert.pem',
                                 key_file='../cer/key.pem')
        self.load_data()

    def load_data(self):
        self.trajectory = h5py.File(self.args_cli.data_dir, 'r')
        import pdb
        pdb.set_trace()

    def run(self):
        self.tv.start()
        self.processor.process(self.tv)
        self.tv.stop()
        self.shm.close()
        self.shm.unlink()
