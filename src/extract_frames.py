from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
video_path = ROOT / "data/videos/cam_1.mp4"
output_dir = ROOT / "data/evaluation/images"
output_dir.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(video_path))
try:
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Cannot read the video's frame rate")
    for number in range(10):
        seconds = number * 5
        frame_index = round(seconds * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success:
            raise RuntimeError(f"Cannot read frame at {seconds} seconds")
        filename = output_dir / f"cam_1_{seconds:03d}s.jpg"
        if not cv2.imwrite(str(filename), frame):
            raise RuntimeError(f"Cannot save image: {filename}")
        print(f"Saved {filename.name}")
finally:
    cap.release()