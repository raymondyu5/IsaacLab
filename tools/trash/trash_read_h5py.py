import h5py
import numpy as np
# Open the HDF5 file in read mode
file_path = "/home/ensu/Documents/weird/test/raw_data.hdf5"
actions_buffer = []
with h5py.File(file_path, "r") as f:
    # List all the groups in the file

    for demo in f["data"]:
        delta_pos = np.array(f["data"]["demo_93"]["actions"])
        actions_buffer.append(delta_pos)
import pdb

pdb.set_trace()
actions = np.concatenate(actions_buffer, axis=0)
