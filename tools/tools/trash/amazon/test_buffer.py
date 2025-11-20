import numpy as np
import cv2
from multiprocessing import shared_memory, Process, Queue


def writer(shm_name, shape, dtype, q: Queue):
    # Attach to existing shared memory
    shm = shared_memory.SharedMemory(name=shm_name)
    buf = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    while True:
        idx = q.get()
        if idx is None:  # shutdown signal
            break
        frame = buf.copy()  # read from shared memory
        cv2.imwrite(f"frame_{idx:06d}.jpg", frame)

    shm.close()


if __name__ == "__main__":
    # Example frame (simulate teleop capture)
    shape = (1080, 1920, 3)
    dtype = np.uint8
    frame = np.random.randint(0, 255, shape, dtype=dtype)

    # Allocate shared memory buffer
    shm = shared_memory.SharedMemory(create=True, size=frame.nbytes)
    buf = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

    q = Queue()
    p = Process(target=writer, args=(shm.name, shape, dtype, q))
    p.start()
    import time

    # Teleop loop simulation
    for i in range(200):
        start_time = time.time()
        buf[:] = frame  # write into shared memory
        q.put(i)  # tell writer "frame ready"
        print(time.time() - start_time)

    # Shutdown
    q.put(None)
    p.join()
    shm.close()
    shm.unlink()
