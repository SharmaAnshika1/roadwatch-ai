import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "data/videos/cam_1.mp4")
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1")
    if not args.source.is_file():
        parser.error(f"Video not found: {args.source}")
    (ROOT / ".ultralytics").mkdir(exist_ok=True)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
    import cv2
    from ultralytics import YOLO
    cap = cv2.VideoCapture(str(args.source))
    if not cap.isOpened():
        raise RuntimeError("Cannot open source video")
    writer = None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError("Invalid video metadata")
        folder = ROOT / "outputs" / f"{args.source.stem}_{time.time_ns()}"
        folder.mkdir(parents=True)
        models = ROOT / "models"
        models.mkdir(exist_ok=True)
        model = YOLO(str(models / "yolo11n.pt"))
        writer = cv2.VideoWriter(
            str(folder / "annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Cannot create output video")
        frames = 0
        detections = 0
        started = time.perf_counter()
        with (folder / "detections.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow([
                "frame_index_0based",
                "time_seconds",
                "class",
                "confidence",
                "x1",
                "y1",
                "x2",
                "y2",
            ])
            for index in range(math.ceil(args.seconds * fps)):
                ok, frame = cap.read()
                if not ok:
                    break
                result = model.predict(
                    frame,
                    classes=[2, 5, 7],
                    conf=args.conf,
                    device=args.device,
                    imgsz=640,
                    verbose=False,
                )[0]
                for box in result.boxes:
                    csv_writer.writerow([
                        index,
                        round(index / fps, 3),
                        model.names[int(box.cls.item())],
                        round(box.conf.item(), 4),
                        *[round(value, 1) for value in box.xyxy[0].tolist()],
                    ])
                    detections += 1
                annotated = result.plot()
                writer.write(annotated)
                if index == 0:
                    cv2.imwrite(str(folder / "first_frame.jpg"), annotated)
                frames += 1
                if frames % 50 == 0:
                    print(f"Processed {frames} frames", flush=True)
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if frames == 0:
        raise RuntimeError("No frames decoded")
    summary = {
        "source": str(args.source.resolve()),
        "model": "yolo11n.pt",
        "confidence": args.conf,
        "device": args.device,
        "fps": fps,
        "resolution": [width, height],
        "frames_processed": frames,
        "video_seconds": frames / fps,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
        "detection_rows": detections,
        "note": (
            "Repeated detections across frames; NOT unique vehicle counts. "
            "Classes: car, bus, truck."
        ),
    }
    (folder / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    print(f"Results: {folder}")

if __name__ == "__main__":
    main()