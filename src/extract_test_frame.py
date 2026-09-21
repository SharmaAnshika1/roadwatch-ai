from pathlib import Path
import cv2
ROOT = Path(__file__).resolve().parents[1]
video_path = ROOT / "data/videos/cam_1.mp4"
output_dir = ROOT / "data/test/images"
output_dir.mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(str(video_path))
try:
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise RuntimeError("Cannot read video frame rate")
    for seconds in range(120, 170, 5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
        success, frame = cap.read()
        if not success:
            raise RuntimeError(f"Cannot read frame at {seconds} seconds")
        path = output_dir / f"cam_1_{seconds:03d}s.jpg"
        if path.exists():
            raise FileExistsError(f"Image already exists: {path}")
        if not cv2.imwrite(str(path), frame):
            raise RuntimeError(f"Cannot save image: {path}")
        print(f"Saved {path.name}")
finally:
    cap.release()