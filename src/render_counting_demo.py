import argparse
import csv
import math
from pathlib import Path
import cv2
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.run / 'annotated.mp4'
    with (args.run / 'crossing_events_near.csv').open(newline='', encoding='utf-8-sig') as handle:
        events = list(csv.DictReader(handle))
    events.sort(key=lambda e: float(e['first_observed_after_crossing_seconds']))
    for event in events:
        if event['direction'] not in ('right_to_left', 'left_to_right'):
            raise ValueError('Unknown direction')
        t = float(event['first_observed_after_crossing_seconds'])
        if not math.isfinite(t) or t < 0:
            raise ValueError('Invalid event time')
        if tuple(float(event[k]) for k in ('line_x', 'line_y_min', 'line_y_max')) != (600, 200, 340):
            raise ValueError('Event line differs from demo line')
    if args.output.resolve() == source.resolve():
        raise ValueError('Output must differ from source')
    cap = cv2.VideoCapture(str(source))
    writer = None
    index = 0
    cursor = 0
    counts = {'right_to_left': 0, 'left_to_right': 0}
    try:
        if not cap.isOpened():
            raise RuntimeError('Cannot open tracking video')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError('Invalid video metadata')
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError('Cannot create demo')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            now = index / fps
            while cursor < len(events) and float(events[cursor]['first_observed_after_crossing_seconds']) <= now + 1e-8:
                counts[events[cursor]['direction']] += 1
                cursor += 1
            cv2.line(frame, (600, 200), (600, 340), (255, 255, 0), 3)
            for p in ((600, 200), (600, 340)):
                cv2.circle(frame, p, 5, (255, 255, 0), -1)
            top = height - 225
            cv2.rectangle(frame, (15, top), (565, height - 15), (25, 25, 25), -1)
            lines = [
                'ROADWATCH AI | Crossing demo',
                f'Right to left: {counts["right_to_left"]}    Left to right: {counts["left_to_right"]}',
                f'Total: {sum(counts.values())}    Time: {now:.1f}s',
                'Automated counts | Partial manual verification',
                'Image directions, not turning movements',
            ]
            for n, text in enumerate(lines):
                cv2.putText(frame, text, (30, top + 35 + n * 35), cv2.FONT_HERSHEY_SIMPLEX, .62 if n < 3 else .5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(frame)
            if index == 599:
                cv2.imwrite(str(args.output.with_suffix('.jpg')), frame)
            index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if not index or cursor != len(events):
        raise RuntimeError('Video empty or some events fall beyond decoded video')
    print(f'Frames: {index}; events: {cursor}; totals: {counts}')
    print(f'Saved: {args.output}')
if __name__ == '__main__':
    main()
