import h5py
import shutil

src1 = "/media/ensu/data/datasets/grab/raw_data.hdf5"
src2 = "/media/ensu/data/datasets/dexycb/raw_data.hdf5"
dst = "file_merged.hdf5"

# Copy the first file as the base
shutil.copy(src1, dst)

# Open the destination (merged) file in read/write mode
with h5py.File(dst, "a") as f_dst:
    # Open the second source file

    with h5py.File(src2, "r") as f_src2:
        for key in f_src2.keys():
            if key in f_dst:
                print(
                    f"Warning: {key} already exists in merged file. Skipping.")
                continue
            f_src2.copy(key, f_dst)  # copy whole group or dataset
