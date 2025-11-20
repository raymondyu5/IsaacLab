from gradio_client import Client, handle_file
from PIL import Image
import numpy as np
# Replace with your Gradio app URL
url = "https://2a59ea9dd953324a2f.gradio.live"

# Initialize the client
client = Client(url)

# Prompt and image path
prompt = "In: What action should the robot take to pick up the eggplant?\nOut:"
image_path = Image.open(
    "/home/ensu/Documents/weird/IsaacLab/logs/gs_image_0.png")
image_array = np.array(image_path)

# # Use handle_file to manage file inputs
# with handle_file(image_path) as file:
# Send the request to the Gradio app
while True:
    data = {"prompt": prompt, "image": image_array.tolist()}
    response = client.predict(
        data,
        api_name="/predict"  # Optional: Specify the API name if needed
    )

    # Print the response
    print("Response:", response)
