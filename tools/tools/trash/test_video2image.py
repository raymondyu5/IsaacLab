import cv2
import os

# Set video path
video_path = "/home/ensu/Downloads/cabinet_normalized (1).mp4"  # Change to your video file
output_folder = "/home/ensu/Downloads/extracted_frames"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break  # End of video

    # Save frame as image
    frame_filename = os.path.join(output_folder,
                                  f"frame_{frame_count:05d}.png")
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

# Release the video capture
cap.release()

print(f"Extracted {frame_count} frames and saved them in '{output_folder}'")
