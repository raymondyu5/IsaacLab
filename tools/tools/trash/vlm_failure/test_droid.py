import tensorflow_datasets as tfds     
import requests
from google.cloud import storage
ds = tfds.load("droid", 
    data_dir="gs://gresearch/robotics", split="train")

for episode in ds.take(5):
    import pdb
        
    pdb.set_trace()
    gcs_path = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
    bucket_name = gcs_path.split("/")[2]
    file_path = "/".join(gcs_path.split("/")[3:])

    # Initialize GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(file_path)

    # Download the file
    blob.download_to_filename("trajectory.h5")
    print("Download complete: trajectory.h5")
    for ep in episode["steps"]:
        image = ep["observation"]["exterior_image_1_left"]
        wrist_image = ep["observation"]["wrist_image_left"]
        action = ep["action"]
        instruction = ep["language_instruction"]
        
       