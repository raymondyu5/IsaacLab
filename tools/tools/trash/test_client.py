import requests

url = "http://127.0.0.1:5000/predict"
data = {
    "prompt":
    "In: What action should the robot take to pick up the eggplant?\nOut:",
    "image_path": "/home/ensu/Documents/weird/IsaacLab/logs/gs_image_0.png"
}

response = requests.post(url, json=data)
print(response.json())
