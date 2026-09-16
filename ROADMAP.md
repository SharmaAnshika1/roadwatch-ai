# RoadWatch AI — project roadmap
## The real problem
Given a fixed-camera traffic video, estimate how many vehicles travel in each direction during a time interval. Save an annotated video and an auditable event CSV. A useful result must explain its errors, not just display boxes.
Our first version uses recorded video on your laptop. Live cameras and deployment come after offline evaluation.
## Where we are now
- Five source videos and dataset reference files are available.
- Python 3.13 and the project virtual environment are installed.
- `src/detect.py` runs pretrained YOLO11n on cars, buses, and trucks.
- The first 30-second run processed 300 frames in about 62 seconds on CPU.
- Its 1,576 detection rows are repeated frame observations, not 1,576 vehicles.
- Tracking, directional counting, evaluation tooling, and a dashboard are future milestones.
## Learning schedule
Use this as an estimated 4–6 week plan with 5–8 focused hours per week. Advance when the completion check passes; the dates are flexible.
| Stage | Build and learn | Deliverable | Completion check |
|---|---|---|---|
| 1. Setup and video basics — 2 sessions | Open the project in VS Code; learn paths, environments, frames, FPS, and image coordinates. | Existing environment and a 30-second detection run. | Explain why 30 seconds at 10 FPS gives 300 frames; open the output video. |
| 2. Detection baseline — 3 sessions | Read `detect.py`; learn bounding boxes, classes, confidence, and pretrained inference. | Annotated video, detection CSV, observation notes. | Explain the main loop and identify at least 3 concrete errors with timestamps. |
| 3. Controlled experiments — 2–3 sessions | Compare confidence 0.25 and 0.5 on the same frames, then compare normal, dawn, and rain clips. | A small experiment table with settings and findings. | Explain which errors improved and which got worse. Do not claim weather caused all differences: traffic also changes between clips. |
| 4. Tracking — 3–4 sessions | Add a tracker; learn persistent IDs, occlusion, lost tracks, and ID switches. Process every frame initially. | Future `src/track.py`, video with IDs, track records. | Follow 10 visible vehicles manually and record ID switches and lost tracks. |
| 5. Directional counting — 3–4 sessions | Define a counting line for one clear movement; use track history, crossing direction, and an event ledger. Handle line jitter and re-entry explicitly. | Future `src/count.py`, direction overlay, `events.csv`. | Check a short clip manually; a stopped or jittering vehicle must not repeatedly increase the count. |
| 6. Evaluation — 3 sessions | Compare predicted crossing events to manually labeled events using the same time interval, boundary, direction, and class rules. | Future `src/evaluate.py` and evaluation report. | Report true and predicted counts, absolute error, missed events, duplicate events, and runtime on held-out segments. |
| 7. Usable app and portfolio — 3–4 sessions | Add a local interface for selecting a video, settings, progress, and downloading results. | Future `app.py`, demo, screenshots, README. | A second person can process a short clip and understand the results without editing code. |
## What to do today in VS Code
1. Open `roadwatch-ai` as the folder.
2. Open `src/detect.py` and read the numbered comments.
3. Open a PowerShell terminal in the project folder.
4. Run:
```powershell
.\.venv\Scripts\python.exe src\detect.py --seconds 30
```
5. Open the new folder printed under `Results:`. Watch `annotated.mp4` in a video player if VS Code does not play it.
6. Create `notes.md` locally and write three timestamped observations.
7. Change only the confidence threshold:
```powershell
.\.venv\Scripts\python.exe src\detect.py --seconds 30 --conf 0.5
```
Compare the same moments in the two runs. A higher confidence threshold can remove both false positives and real vehicles.
## Dataset plan
| File | How we will use it |
|---|---|
| `cam_1.mp4` | Develop and explain the baseline. |
| `cam_1_dawn.mp4` | Inspect low-light behavior using the same camera view. |
| `cam_1_rain.mp4` | Inspect rainy-scene behavior using the same camera view. |
| `cam_6_snow.mp4` | Test a different camera and snow; both factors change together. |
| `cam_5.mp4` | Later evaluate the first minute using supplied counting annotations, after implementing matching movement rules. |
Keep fixed evaluation segments separate from segments used for tuning. There are no full detection-box labels in the extracted references, so we cannot claim detection precision or recall without manually labeling suitable frames.
The supplied counting CSV covers only the first minute of `cam_5.mp4`. Its events use the challenge ROI exit and movement definitions. Our initial line-crossing counter measures a different event until we align those rules. Challenge vehicle groups also differ from the model's car/bus/truck classes. Read `data/reference/ReadMe.txt` before benchmark comparisons.
## Evaluation rules
- For each direction, report true count, predicted count, and absolute error: `abs(predicted - true)`.
- Percentage count error is `100 * abs(predicted - true) / true` only when the true count is nonzero. For zero true count, report the absolute number of false counts.
- Equal totals can hide one missed vehicle and one duplicate. Inspect matched events as well as totals.
- For event precision and recall, define a timestamp tolerance in advance and match events one-to-one within the same direction/class.
- Keep the source file, time interval, model, settings, and hardware with each result.
- Record throughput separately from correctness. The first CPU run was slower than real time; live operation is not yet established.
## Definition of a finished V1
- A user selects a local video and a configured camera/movement.
- The system produces a tracked video and timestamped directional count events.
- Counts can be checked against a documented manual reference on held-out segments.
- The report shows errors across multiple conditions and explains limitations.
- Another person can follow the setup instructions and reproduce a short run.
## Later extensions
After V1 works: longer videos, camera-specific movement configurations, traffic volume by time bucket, performance tuning, and a live-camera trial. Custom model training should follow evidence of a detection problem and a plan for labeled data.
## How to describe it on your portfolio
"Built a traffic analytics prototype that detects, tracks, and counts vehicles from fixed-camera video, with timestamped event exports and evaluation on manually checked segments."
Use that full description only after those stages are implemented. Today, describe it as a working vehicle-detection baseline. Add measured results later; do not invent an accuracy percentage.