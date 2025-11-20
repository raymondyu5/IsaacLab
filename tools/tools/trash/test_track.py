import cv2
import mediapipe as mp
import os
import numpy as np
import imageio
# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hand_tracker = mp_hands.Hands(static_image_mode=False,
                              max_num_hands=2,
                              min_detection_confidence=0.5,
                              min_tracking_confidence=0.5)

# Video path and output directory
video_path = "/home/ensu/Downloads/IMG_1637.MOV"
output_folder = "/home/ensu/Downloads/extracted_frames"
os.makedirs(output_folder, exist_ok=True)

output_video = imageio.get_writer(f"{output_folder}/output.mp4", fps=30)
# Open video
cap = cv2.VideoCapture(video_path)
frame_index = 0
index_finger_trajectory = []  # Store index finger positions
thumb_trajectory = []  # Store thumb positions

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process frame with MediaPipe
    results = hand_tracker.process(frame_rgb)

    # Extract and store fingertip positions
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape

            # Get index and thumb tip positions
            index_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[
                mp_hands.HandLandmark.THUMB_TIP]

            # Convert normalized coordinates to pixels
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)

            # Store trajectory points
            index_finger_trajectory.append((index_x, index_y))
            thumb_trajectory.append((thumb_x, thumb_y))

            cv2.circle(frame, (index_x, index_y), 12, (0, 0, 255),
                       -1)  # Green for index finger
            # cv2.circle(frame, (thumb_x, thumb_y), 10, (0, 0, 255),
            #            -1)  # Red for thumb

    # Save frame
    frame_path = os.path.join(output_folder, f"frame_{frame_index}.png")
    cv2.imwrite(frame_path, frame)
    frame_index += 1
    output_video.append_data(frame[:, :, ::-1])

# Release video capture
cap.release()

print("Video processing complete. Drawing trajectory on the last frame...")

# Load the last image
last_image_path = os.path.join(output_folder, f"frame_{frame_index - 1}.png")
last_image = cv2.imread(last_image_path)

# Draw the full trajectory on the last frame
for i in range(1, len(index_finger_trajectory), 20):
    cv2.circle(last_image, index_finger_trajectory[i], 10, (0, 255, 0), -1)

    cv2.line(last_image, index_finger_trajectory[i - 1],
             index_finger_trajectory[i], (0, 255, 0),
             2)  # Green for index finger
# for i in range(1, len(thumb_trajectory)):
#     cv2.line(last_image, thumb_trajectory[i - 1], thumb_trajectory[i],
#              (0, 0, 255), 2)  # Red for thumb

# Save the final image with trajectory
trajectory_image_path = os.path.join(output_folder, "finger_trajectory.png")
cv2.imwrite(trajectory_image_path, last_image)

print(f"Trajectory drawn and saved at: {trajectory_image_path}")
