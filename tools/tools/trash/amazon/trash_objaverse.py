import objaverse
import json
from download_objaverse import load_objects
import os
import glob
import shutil
# Specify the path to your JSON file
file_path = '/media/ensu/data/datasets/objaverse/lvis-annotations.json'

# Open and load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

dest_dir = "/media/ensu/data/datasets/objaverse"
import multiprocessing

processes = multiprocessing.cpu_count()

# object_list = [
#     "hammer", "hat", "vase", "lamp", "cup", "pen", "sushi", "cupboard",
#     "teacup", "trophy_cup", "Dixie_cup", "book", "bowl", "fishbowl",
#     "pipe_bowl", "sugar_bowl", "soup_bowl", "plate", "paper_plate", "apple",
#     "banana", "ball", "baseball", "peach", "pepper", "potato", "tomato",
#     "watermelon", "carrot", "bottle", "water_bottle", "wine_bottle", "toy",
#     "magazine", "car_(automobile)", "oven", "pan_(for_cooking)", "basket",
#     "blanket", "boat", "bucket", "alarm_clock", "spatula", "earphone", "watch"
# ]
object_list = ["teddy_bear", "rabbit"]
for name in object_list:
    if os.path.exists(os.path.join(dest_dir, name)):
        num_file = len(glob.glob(dest_dir + f"/{name}/*/*/*.glb"))

        # if num_file > 10:
        #     continue
        # else:
        shutil.rmtree(os.path.join(dest_dir, name))
    print(name)
    save_path = os.path.join(dest_dir, name)
    os.makedirs(save_path, exist_ok=True)

    try:
        load_objects(uids=data[name],
                     save_dir=save_path,
                     download_processes=20)
    except Exception as e:
        shutil.rmtree(os.path.join(dest_dir, name))
        print(f"[Warning] Failed to load objects for {name}: {e}")
        continue  # optionall
