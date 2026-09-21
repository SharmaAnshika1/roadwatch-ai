import json
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = ROOT / "data/test/images"
ANNOTATION_DIR = ROOT / "data/test/annotations"
PREDICTION_DIR = ROOT / "runs/detect/outputs/test_960_conf040/labels"
ROI_PATH = ROOT / "data/reference/ROIs/cam_1.txt"
CLASS_NAMES = {2: "car", 5: "bus", 7: "truck"}
MATRIX_LABELS = ["car", "bus", "truck", "background"]
CONFIDENCE_THRESHOLD = 0.40
IOU_THRESHOLD = 0.50
IMAGE_SIZE = 960
def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    left = max(ax1, bx1)
    top = max(ay1, by1)
    right = min(ax2, bx2)
    bottom = min(ay2, by2)
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0
def yolo_to_xyxy(values, image_width, image_height):
    centre_x, centre_y, width, height = map(float, values[1:5])
    return [
        (centre_x - width / 2) * image_width,
        (centre_y - height / 2) * image_height,
        (centre_x + width / 2) * image_width,
        (centre_y + height / 2) * image_height,
    ]
def match_boxes(manual_boxes, predictions, iou_threshold=IOU_THRESHOLD):
    matched_manual = set()
    true_positives = 0
    false_positives = 0
    predictions = sorted(predictions, key=lambda item: item["confidence"], reverse=True,)
    for prediction in predictions:
        best_index = None
        best_iou = -1.0
        for index, manual in enumerate(manual_boxes):
            if index in matched_manual:
                continue
            if prediction["label"] != manual["label"]:
                continue
            overlap = calculate_iou(prediction["box"], manual["box"])
            if overlap >= iou_threshold and overlap > best_iou:
                best_index = index
                best_iou = overlap
        if best_index is not None:
            matched_manual.add(best_index)
            true_positives += 1
        else:
            false_positives += 1
    false_negatives = len(manual_boxes) - len(matched_manual)
    return true_positives, false_positives, false_negatives
def build_confusion_matrix(manual_boxes, predictions, iou_threshold=IOU_THRESHOLD):
    matrix = np.zeros((4, 4), dtype=int)
    predictions = sorted(predictions, key=lambda item: item["confidence"], reverse=True,)
    matched_manual = set()
    matched_predictions = set()
    for same_class_only in (True, False):
        for prediction_index, prediction in enumerate(predictions):
            if prediction_index in matched_predictions:
                continue
            best_index = None
            best_iou = -1.0
            for manual_index, manual in enumerate(manual_boxes):
                if manual_index in matched_manual:
                    continue
                if same_class_only and prediction["label"] != manual["label"]:
                    continue
                overlap = calculate_iou(prediction["box"], manual["box"])
                if overlap >= iou_threshold and overlap > best_iou:
                    best_index = manual_index
                    best_iou = overlap
            if best_index is not None:
                actual = MATRIX_LABELS.index(manual_boxes[best_index]["label"])
                predicted = MATRIX_LABELS.index(prediction["label"])
                matrix[actual, predicted] += 1
                matched_manual.add(best_index)
                matched_predictions.add(prediction_index)
    for index, manual in enumerate(manual_boxes):
        if index not in matched_manual:
            matrix[MATRIX_LABELS.index(manual["label"]), 3] += 1
    for index, prediction in enumerate(predictions):
        if index not in matched_predictions:
            matrix[3, MATRIX_LABELS.index(prediction["label"])] += 1
    return matrix
def inside_roi(box, roi):
    x1, y1, x2, y2 = box
    centre = ((x1 + x2) / 2, (y1 + y2) / 2)
    return cv2.pointPolygonTest(roi, centre, False) >= 0
def save_confusion_matrix(matrix, image_count, output_path):
    display_matrix = matrix.astype(float)
    display_matrix[3, 3] = np.nan
    fig, ax = plt.subplots(figsize=(8, 7))
    plot = ax.imshow(display_matrix, cmap="Blues", vmin=0)
    fig.colorbar(plot, ax=ax, label="Number of objects")
    ax.set_xticks(range(4), labels=MATRIX_LABELS)
    ax.set_yticks(range(4), labels=MATRIX_LABELS)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title(
        "RoadWatch: test-set confusion matrix\n"
        f"{image_count} frames | ROI only | "
        f"confidence {CONFIDENCE_THRESHOLD:.2f} | "
        f"IoU {IOU_THRESHOLD:.2f}"
    )
    threshold = max(int(matrix.max()), 1) / 2
    for row in range(4):
        for column in range(4):
            unused = row == 3 and column == 3
            value = matrix[row, column]
            text = "N/A" if unused else str(value)
            colour = ("white" if not unused and value > threshold else "black")
            ax.text(column, row, text, ha="center", va="center", color=colour, fontsize=14,)
    fig.text(0.5, 0.02, "Background row: unmatched predictions. "
             "Background column: misses.\n"
             "N/A: true negatives are not counted.",
             ha="center",
             fontsize=9,)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
def main():
    if not PREDICTION_DIR.is_dir():
        raise FileNotFoundError(f"Missing folder: {PREDICTION_DIR}")
    roi = np.loadtxt(ROI_PATH, delimiter=",", dtype=np.float32)
    images = sorted(IMAGE_DIR.glob("*.jpg"))
    if not images:
        raise RuntimeError("No evaluation images found")
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_matrix = np.zeros((4, 4), dtype=int)
    excluded_manual = 0
    excluded_predictions = 0
    print(f"{'Image':<22} {'TP':>5} {'FP':>5} {'FN':>5}")
    for image in images:
        annotation_path = ANNOTATION_DIR / f"{image.stem}.json"
        prediction_path = PREDICTION_DIR / f"{image.stem}.txt"
        annotation = json.loads(
            annotation_path.read_text(encoding="utf-8-sig")
        )
        width = annotation["imageWidth"]
        height = annotation["imageHeight"]
        manual_boxes = []
        for shape in annotation["shapes"]:
            if shape["label"] not in CLASS_NAMES.values():
                raise ValueError(f"Unexpected label: {shape['label']}")
            if shape["shape_type"] != "rectangle":
                raise ValueError(f"Non-rectangle in {image.name}")
            (xa, ya), (xb, yb) = shape["points"]
            box = [min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb),]
            if inside_roi(box, roi):
                manual_boxes.append({
                    "label": shape["label"],
                    "box": box,
                })
            else:
                excluded_manual += 1
        predictions = []
        if prediction_path.is_file():
            for line in prediction_path.read_text().splitlines():
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 6:
                    raise ValueError(f"Invalid prediction: {line}")
                class_id = int(values[0])
                confidence = float(values[5])
                if class_id not in CLASS_NAMES:
                    raise ValueError(f"Unexpected class ID: {class_id}")
                if confidence < CONFIDENCE_THRESHOLD:
                    continue
                box = yolo_to_xyxy(values, width, height)
                if inside_roi(box, roi):
                    predictions.append({
                        "label": CLASS_NAMES[class_id],
                        "box": box,
                        "confidence": confidence,
                    })
                else:
                    excluded_predictions += 1
        tp, fp, fn = match_boxes(manual_boxes, predictions, IOU_THRESHOLD)
        total_matrix += build_confusion_matrix(manual_boxes, predictions, IOU_THRESHOLD)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        print(f"{image.name:<22} {tp:>5} {fp:>5} {fn:>5}")
    precision = (
        total_tp / (total_tp + total_fp)
        if total_tp + total_fp else None
    )
    recall = (
        total_tp / (total_tp + total_fn)
        if total_tp + total_fn else None
    )
    denominator = 2 * total_tp + total_fp + total_fn
    f1 = 2 * total_tp / denominator if denominator else None
    print(f"\nTotal TP: {total_tp}")
    print(f"Total FP: {total_fp}")
    print(f"Total FN: {total_fn}")
    print(f"Manual boxes outside ROI: {excluded_manual}")
    print(f"Predictions outside ROI: {excluded_predictions}")
    for name, value in [("Precision", precision), ("Recall", recall), ("F1", f1),]:
        result = f"{value:.2%}" if value is not None else "N/A"
        print(f"{name}: {result}")
    print("\nConfusion matrix: rows = actual, columns = predicted")
    print("Class order:", ", ".join(MATRIX_LABELS))
    print(total_matrix)
    diagonal = int(np.trace(total_matrix[:3, :3]))
    matrix_fp = int(total_matrix[:, :3].sum()) - diagonal
    matrix_fn = int(total_matrix[:3, :].sum()) - diagonal
    if (diagonal, matrix_fp, matrix_fn) != (total_tp, total_fp, total_fn):
        raise RuntimeError("Matrix totals differ from evaluation totals")
    output_dir = ROOT / "outputs/test_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    confidence_tag = f"{CONFIDENCE_THRESHOLD:.2f}".replace(".", "")
    output_path = (
        output_dir
        / f"confusion_matrix_{IMAGE_SIZE}_conf{confidence_tag}.png"
    )
    save_confusion_matrix(total_matrix, len(images), output_path)
    print(f"Saved confusion matrix: {output_path}")
if __name__ == "__main__":
    main()