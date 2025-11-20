import zipfile
import os

# Step 1: Load ShapeNet synset ID to label dictionary
shapenet_dict = {}
with open(
        '/media/ensu/data/datasets/shapenet/ShapeNetCore/shapenet_synset_list.txt',
        'r') as file:
    for line in file:
        parts = line.strip().split()
        if len(parts) >= 2:
            synset_id = parts[0]
            label = "_".join(
                parts[1:])  # use underscore to avoid spaces in folder names
            shapenet_dict[synset_id] = label


# Step 2: Function to unzip and rename
def unzip_file(zip_path, extract_root, new_name):
    tmp_dir = os.path.join(extract_root, "tmp_extract")
    os.makedirs(tmp_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(tmp_dir)

    extracted_folders = os.listdir(tmp_dir)
    for folder in extracted_folders:
        src = os.path.join(tmp_dir, folder)
        dst = os.path.join(extract_root, new_name)
        os.rename(src, dst)

    os.rmdir(tmp_dir)
    print(f"Unzipped and renamed to: {dst}")


# Step 3: Iterate over zip files and rename
object_zip_path = '/media/ensu/data/datasets/shapenet/ShapeNetCore'
target_dir = '/media/ensu/data/datasets/shapenet/raw'

for filename in os.listdir(object_zip_path):
    if filename.endswith('.zip'):
        synset_id = filename.replace('.zip', '')

        label = shapenet_dict.get(synset_id, synset_id)
        zip_path = os.path.join(object_zip_path, filename)
        unzip_file(zip_path, target_dir, f"{label}")
