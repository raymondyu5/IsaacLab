import imageio
import os

# Specify the directory containing the images
image_folder = '/media/aurmr/data1/weird/IsaacLab/logs/cat/static_gs/raw_imgs'

# Specify the output video file name
output_video = '/media/aurmr/data1/weird/IsaacLab/logs/cat/raw_imgs.mp4'

# Get all image file names in the directory
images = [img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'))]
images.sort()  # Ensure the images are in the correct order

# Define the video writer object, adjust fps (frames per second) as needed
writer = imageio.get_writer(output_video, fps=24)

# Loop through all images and append them to the video
for image in images:
    img_path = os.path.join(image_folder, image)
    img = imageio.imread(img_path)
    writer.append_data(img)

# Close the writer object to finalize the video
writer.close()

print(f"Video saved as {output_video}")
