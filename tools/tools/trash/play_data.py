import h5py
import numpy as np

# Set a random seed for reproducibility if desired
np.random.seed(42)

# Open the existing file in append mode
with h5py.File("logs/1119_placement2/grasp_normalized_noise_aug.hdf5",
               'a') as h5_file:
    # To add a new group for masks
    # del h5_file["mask"]
    grp_mask = h5_file.create_group("mask")

    # Load existing data
    data = h5_file["data"]
    num_demo = len(data)
    training_set_size = int(0.9 * num_demo)

    # Generate randomized indices for training and testing
    all_indices = np.arange(num_demo)
    training_indices = np.random.choice(all_indices,
                                        size=training_set_size,
                                        replace=False)
    test_indices = np.setdiff1d(all_indices, training_indices)

    # Create the demo keys based on randomized indices
    demo_key = {
        "train": ["demo_" + str(i) for i in training_indices],
        "test": ["demo_" + str(i) for i in test_indices]
    }

    # Save the train and test sets in the HDF5 file
    grp_mask.create_dataset("train",
                            data=np.array(demo_key["train"], dtype='S'))
    grp_mask.create_dataset("test", data=np.array(demo_key["test"], dtype='S'))
