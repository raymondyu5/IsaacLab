import numpy as np
import time

B, C, H, W = 1000, 3, 64, 64
filepath = "/gscratch/weirdlab/Entong/tmp/memmap_buffer.dat"

memmap_array = np.memmap(filepath,
                         dtype='float32',
                         mode='w+',
                         shape=(B, C, H, W))

start = time.time()
for i in range(B):
    memmap_array[i] = np.random.rand(C, H, W).astype('float32')
memmap_array.flush()
print("Elapsed time:", time.time() - start)
