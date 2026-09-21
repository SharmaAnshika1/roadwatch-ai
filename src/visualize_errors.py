import json
import cv2
import numpy as np
import evaluate_detection as evaluation
IMAGE_NAME = "cam_1_130s.jpg"
def main():
    image_path = evaluation.IMAGE_DIR / IMAGE_NAME
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Cannot open image: {image_path}")
    height, width = image.shape[:2]
    stem = image_path.stem
    roi = np.loadtxt(
        evaluation.ROOT / "data/reference/ROIs/cam_1.txt",
        delimiter=",",
        dtype=np.float32,
    )
    def inside_roi(box):
        x1, y1, x2, y2 = box
        centre = ((x1 + x2) / 2, (y1 + y2) / 2)
        return cv2.pointPolygonTest(roi, centre, False) >= 0
    annotation_path = evaluation.ANNOTATION_DIR / f"{stem}.json"
    annotation = json.loads(
        annotation_path.read_text(encoding="utf-8-sig")
    )
    manual_boxes = []
    for shape in annotation["shapes"]:
        (xa, ya), (xb, yb) = shape["points"]
        box = [min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)]

        if inside_roi(box):
            manual_boxes.append({
                "label": shape["label"],
                "box": box,
            })
    prediction_path = evaluation.PREDICTION_DIR / f"{stem}.txt"
    predictions = []
    if prediction_path.is_file():
        for line in prediction_path.read_text().splitlines():
            if not line.strip():
                continue
            values = line.split()
            confidence = float(values[5])
            if confidence < 0.40:
                continue
            box = evaluation.yolo_to_xyxy(values, width, height)
            if inside_roi(box):
                predictions.append({
                    "label": evaluation.CLASS_NAMES[int(values[0])],
                    "box": box,
                    "confidence": confidence,
                })
    predictions.sort(key=lambda item: item["confidence"], reverse=True)
    matched_manual = set()
    tp = 0
    fp = 0
    def draw_box(box, text, colour):
        x1, y1, x2, y2 = [round(value) for value in box]
        cv2.rectangle(image, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(
            image,
            text,
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2,
        )
    cv2.polylines(
        image,
        [roi.astype(np.int32)],
        True,
        (255, 255, 0),
        2,
    )
    for prediction in predictions:
        best_index = None
        best_iou = -1.0
        for index, manual in enumerate(manual_boxes):
            if index in matched_manual:
                continue
            if prediction["label"] != manual["label"]:
                continue
            overlap = evaluation.calculate_iou(
                prediction["box"], manual["box"]
            )
            if overlap >= 0.5 and overlap > best_iou:
                best_index = index
                best_iou = overlap
        if best_index is not None:
            matched_manual.add(best_index)
            tp += 1
            draw_box(
                prediction["box"],
                f"TP {prediction['label']}",
                (0, 200, 0),
            )
        else:
            fp += 1
            draw_box(
                prediction["box"],
                f"FP {prediction['label']}",
                (0, 0, 255),
            )
    fn = 0
    for index, manual in enumerate(manual_boxes):
        if index not in matched_manual:
            fn += 1
            draw_box(
                manual["box"],
                f"FN {manual['label']}",
                (0, 165, 255),
            )
    expected = evaluation.match_boxes(manual_boxes, predictions)
    if (tp, fp, fn) != expected:
        raise RuntimeError("Visual counts differ from evaluator")
    image = cv2.copyMakeBorder(
        image, 0, 80, 0, 0, cv2.BORDER_CONSTANT, value=(35, 35, 35)
    )
    cv2.putText(
        image,
        f"{IMAGE_NAME} | TP: {tp}  FP: {fp}  FN: {fn}",
        (15, height + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
    )
    cv2.putText(
        image,
        "Green: matched | Red: unmatched prediction | Orange: missed label | Cyan: ROI",
        (15, height + 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )
    output_dir = evaluation.ROOT / "outputs/error_review"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{stem}_errors.jpg"
    if not cv2.imwrite(str(output_path), image):
        raise RuntimeError("Could not save comparison image")
    print(f"TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Saved: {output_path}")
if __name__ == "__main__":
    main()