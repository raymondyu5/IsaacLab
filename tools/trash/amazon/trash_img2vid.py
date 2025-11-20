import os
import re
import cv2
import imageio

IMG_EXTS = (".png", ".jpg", ".jpeg")


def natural_key(s: str):
    """Sort like humans: file2.png < file10.png."""
    return [
        int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)
    ]


def images_to_video_in_folder(folder_path: str,
                              fps: int = 30,
                              out_name: str | None = None):
    """Create a video from all images inside `folder_path`."""
    # Collect images
    files = [
        f for f in os.listdir(folder_path) if f.lower().endswith(IMG_EXTS)
    ]
    if not files:
        print(f"[skip] No images in: {folder_path}")
        return

    files.sort(key=natural_key)
    image_paths = [os.path.join(folder_path, f) for f in files]

    # Read first frame to get size
    first_bgr = cv2.imread(image_paths[0])
    if first_bgr is None:
        print(f"[skip] Cannot read first image: {image_paths[0]}")
        return
    h, w = first_bgr.shape[:2]

    # Output path
    if out_name is None:
        out_name = os.path.basename(os.path.normpath(folder_path)) or "output"
    output_video_path = os.path.join(folder_path, f"{out_name}.mp4")

    # Writer
    writer = imageio.get_writer(output_video_path, fps=fps)

    # Append frames (resize if needed, convert BGR->RGB)
    for p in image_paths:
        bgr = cv2.imread(p)
        if bgr is None:
            print(f"[warn] Cannot read {p}, skipping.")
            continue
        if bgr.shape[0] != h or bgr.shape[1] != w:
            bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        writer.append_data(bgr)

    writer.close()
    print(f"[ok] Saved: {output_video_path}")


def batch_images_to_videos(parent_folder: str,
                           fps: int = 30,
                           recurse: bool = False):
    """
    For each subfolder in `parent_folder`, create <subfolder>.mp4 from its images.
    If `recurse=True`, walks all nested subfolders; else only immediate children.
    """
    if recurse:
        for root, dirs, _files in os.walk(parent_folder):
            # Skip the parent itself if it contains images directly; only process dirs
            for d in dirs:
                images_to_video_in_folder(os.path.join(root, d), fps=fps)
    else:
        for name in os.listdir(parent_folder):
            sub = os.path.join(parent_folder, name)
            if os.path.isdir(sub):
                images_to_video_in_folder(sub, fps=fps)


# ===== Example usage =====
if __name__ == "__main__":
    parent = "/media/ensu/data/ICLR/trash/reactive_vae"  # parent that contains many folders
    batch_images_to_videos(parent, fps=10, recurse=True)
