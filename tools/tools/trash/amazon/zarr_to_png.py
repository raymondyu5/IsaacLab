import zarr
import os
import cv2


def list_zarr_files(root_dir):
    zarr_files = []
    for root, dirs, files in os.walk(root_dir):
        # Check all directories (since .zarr is usually a folder)
        for d in dirs:
            if d.lower().endswith(".zarr"):
                zarr_files.append(os.path.join(root, d))
        # (Optional) also check for rare cases where .zarr is a file
        for f in files:
            if f.lower().endswith(".zarr"):
                zarr_files.append(os.path.join(root, f))
    return zarr_files


if __name__ == "__main__":
    zarr_path = "logs/trash/image"
    output_dir = "logs/trash/sort_image"

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    zarr_files = list_zarr_files(zarr_path)

    for zarr_file in zarr_files:
        zarr_data = zarr.open(zarr_file, mode='r')

        num_images = zarr_data['data/rgb_0'].shape[0]

        base_filename = os.path.splitext(os.path.basename(zarr_file))[0]

        os.makedirs(output_dir + f"/{base_filename}", exist_ok=True)

        for i in range(num_images):
            image = zarr_data['data/rgb_0'][i]
            output_path = os.path.join(output_dir + f"/{base_filename}",
                                       f"img_{i:05d}.png")

            cv2.imwrite(output_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
