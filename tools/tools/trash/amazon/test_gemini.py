import cv2
import os
import google.generativeai as genai
from PIL import Image
import glob
import numpy as np
import time

# ------------------------------
# 1. Extract frames (every 5th up to frame 120)
# ------------------------------
video_path = "/home/ensu/Downloads/17.mp4"
frames_dir = "logs/frames"
scores_dir = "logs/scores"
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(scores_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_id = 0
saved_frames = []
while True:
    ret, frame = cap.read()
    if not ret or frame_id > 120:
        break
    if frame_id % 5 == 0:
        fname = f"{frames_dir}/frame_{frame_id:03d}.jpg"
        cv2.imwrite(fname, frame)
        saved_frames.append((frame_id, fname))
    frame_id += 1
cap.release()

# ------------------------------
# 2. Configure Gemini
# ------------------------------
genai.configure(api_key="AIzaSyAyn7fpNfQcN9wtdXvcarP08hYeBma9UxI")
model = genai.GenerativeModel("gemini-1.5-pro")


# ------------------------------
# 3. Batch reward function
# ------------------------------
def get_rewards_batch(frame_paths):
    images = [Image.open(f) for f in frame_paths]

    prompt = f"""
    The task goal is: The robot must lift the object completely off the table.

    You are given {len(frame_paths)} frames.
    For each frame, return a numeric task_completion_percentage between 0 and 100.

    Scoring rules:
    - 0%: No progress (arm far from object, no attempt).
    - Up to 30%: Arm positioned above object but not yet gripping.
    - 50%: Hand fully gripping the object, but object has not left the table.
    - 100%: Bunny is clearly lifted off the table (goal achieved).
    - Intermediate values should reflect smooth progress between these milestones.

    Output format:
    - Return ONLY a valid Python list of floats, in the same order as the frames.
    - Do not include any words, explanations, or formatting besides the list itself.
    - Example: [0, 12.5, 30.0, 50.0, 100.0]

    """

    response = model.generate_content([prompt] + images)

    raw = response.text.strip()
    print("Raw response:", raw)  # <-- debug: see what Gemini sent

    # Try to extract the list from the text
    try:
        # Find the first '[' and last ']' to slice cleanly
        start = raw.find("[")
        end = raw.rfind("]") + 1
        scores = eval(raw[start:end])
        return [float(s) for s in scores]
    except Exception as e:
        print("Parse error:", raw)
        return [None] * len(frame_paths)


# ------------------------------
# 4. Query Gemini only on sampled frames
# ------------------------------
batch_size = 120
sampled_results = []
for i in range(0, len(saved_frames), batch_size):
    batch = saved_frames[i:i + batch_size]
    batch_ids, batch_files = zip(*batch)
    scores = get_rewards_batch(batch_files)
    for fid, fpath, s in zip(batch_ids, batch_files, scores):
        sampled_results.append((fid, s))
    time.sleep(0.2)  # throttle

# ------------------------------
# 5. Interpolate to all frames 0–120
# ------------------------------
frame_ids = np.arange(0, 121)
interp_scores = np.interp(frame_ids, [fid for fid, _ in sampled_results],
                          [s for _, s in sampled_results])

# ------------------------------
# 6. Save each frame with score in filename
# ------------------------------
cap = cv2.VideoCapture(video_path)
fid = 0
while True:
    ret, frame = cap.read()
    if not ret or fid > 120:
        break
    score = interp_scores[fid]
    fname = f"{scores_dir}/frame_{fid:03d}_score_{score:.2f}.jpg"
    cv2.imwrite(fname, frame)
    fid += 1
cap.release()

print(f"Saved scored frames in {scores_dir}")
