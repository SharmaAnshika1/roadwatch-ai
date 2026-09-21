# Preliminary detection evaluation

Model: YOLO11n
Sample: 10 manually annotated frames from cam_1.mp4
Evaluation area: supplied camera 1 ROI, using box centres
Matching: same class, one-to-one matching, IoU >= 0.5
Ground-truth boxes inside ROI: 52

| Image size | Confidence | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| 640 | 0.25 | 29 | 10 | 23 | 74.36% | 55.77% | 63.74% |
| 960 | 0.25 | 45 | 20 | 7 | 69.23% | 86.54% | 76.92% |
| 960 | 0.40 | 38 | 5 | 14 | 88.37% | 73.08% | 80.00% |

Increasing image size improved recall on this sample.
Increasing confidence reduced false positives but increased misses.

These frames were used for tuning, so these are development results,
not independent test results. Results depend on manual-label quality.

## Confusion matrix

![Development-set confusion matrix](demo/confusion-matrix.png)

Rows represent actual labels; columns represent predictions.
There were 35 correctly detected cars, 3 correctly detected trucks,
14 missed labeled objects, and 5 unmatched car predictions.

There are no labeled buses inside this evaluation region,
so bus recall cannot be assessed.

These results use 10 development frames, not an independent test set.

### Test error review: cam_1_130s.jpg

- Correct detections: 3
- False detections: 0
- Missed vehicles: 4
- Misses cluster around the distant bend; several vehicles are
  partially obscured by branches.
- Small object size and obstruction are suspected failure factors,
  not confirmed causes.