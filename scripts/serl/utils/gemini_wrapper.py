import time
import google.generativeai as genai
from PIL import Image
import numpy as np

import cv2


class GeminiWrapper:

    def __init__(self,
                 model_name: str = "gemini-1.5-pro",
                 max_frames=30,
                 interval=5):
        genai.configure(api_key="AIzaSyDS1ZZob_BSwAWTAOwTIbGB1UxGbeRidHo")

        self.gemini_model = genai.GenerativeModel(model_name)
        self.max_frames = max_frames
        self.interval = interval

    def get_rewards_batch(self, frame_arrays):
        pil_frames = []
        for arr in frame_arrays:
            if isinstance(arr, np.ndarray):
                pil_frames.append(Image.fromarray(arr.astype(np.uint8)))
            elif isinstance(arr, Image.Image):
                pil_frames.append(arr)
            else:
                raise TypeError(f"Unsupported frame type: {type(arr)}")

        prompt = f"""
        The task goal is: The robot must lift the object completely off the table.

        You are given {len(frame_arrays)} frames.
        Return a Python list of exactly {len(frame_arrays)} floats,
        one per frame, in the same order.

        Rules:
        - 0%: No progress (arm far from object).
        - Up to 30%: Arm above object.
        - 50%: Hand gripping but object not lifted.
        - 100%: Object lifted.
        - Intermediate values = smooth progress.

        Format:
        - Only output a Python list of {len(frame_arrays)} floats.
        - Example: [0.0, 10.0, 20.0, ..., 100.0]
        """

        response = self.gemini_model.generate_content([prompt] + pil_frames)
        raw = response.text.strip()
        print("Raw response:", raw)

        start = raw.find("[")
        end = raw.rfind("]") + 1
        scores = eval(raw[start:end])

        if len(scores) != len(frame_arrays):
            raise ValueError(
                f"Length mismatch: expected {len(frame_arrays)}, got {len(scores)}"
            )

        return [float(s) for s in scores]

    def get_reward(self, images):
        """
        images: list of np.ndarray frames
        """
        n = len(images)
        scores = np.zeros(n)

        # pick indices for inference
        key_indices = list(range(0, n, self.interval))
        key_frames = [images[idx] for idx in key_indices]

        # process in batches
        all_key_scores = []
        for i in range(0, len(key_frames), self.max_frames):
            batch = key_frames[i:i + self.max_frames]
            batch_scores = self.get_rewards_batch(batch)
            all_key_scores.extend(batch_scores)
            time.sleep(0.1)

        # expand scores: copy each key score to its interval
        for k, idx in enumerate(key_indices):
            score = all_key_scores[k]
            end_idx = min(idx + self.interval, n)
            scores[idx:end_idx] = score

        return scores


if __name__ == "__main__":
    import os

    video_path = "/home/ensu/Downloads/success_01.mp4"

    scores_dir = "logs/scores"

    os.makedirs(scores_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0
    saved_frames = []
    images = []
    while True:
        ret, frame = cap.read()
        if not ret or frame_id > 120:
            break

        images.append(frame)
        frame_id += 1
    cap.release()

    wrapper = GeminiWrapper()

    sampled_results = wrapper.get_reward(images)

    for fid in range(len(sampled_results)):

        score = sampled_results[fid]
        fname = f"{scores_dir}/frame_{fid:03d}_score_{score:.2f}.jpg"
        cv2.imwrite(fname, images[fid])

    cap.release()

    print(f"Saved scored frames in {scores_dir}")
