import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    args = parser.parse_args()

    line_x = 600
    line_y_min = 200
    line_y_max = 340
    previous = {}
    counted = set()
    events = []

    with args.source.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            track_id = row["track_id"]
            if not track_id:
                continue

            x = (float(row["x1"]) + float(row["x2"])) / 2
            y = (float(row["y1"]) + float(row["y2"])) / 2
            time_seconds = float(row["time_seconds"])
            direction = None

            if track_id in previous and track_id not in counted:
                previous_x, previous_y, previous_time = previous[track_id]

                if previous_x > line_x >= x:
                    direction = "right_to_left"
                elif previous_x < line_x <= x:
                    direction = "left_to_right"

                if direction:
                    fraction = (line_x - previous_x) / (x - previous_x)
                    crossing_y = previous_y + fraction * (y - previous_y)

                    if line_y_min <= crossing_y <= line_y_max:
                        events.append({
                            "track_id": track_id,
                            "class": row["class"],
                            "direction": direction,
                            "previous_observation_seconds": previous_time,
                            "first_observed_after_crossing_seconds": time_seconds,
                            "crossing_y_estimate": round(crossing_y, 2),
                            "line_x": line_x,
                            "line_y_min": line_y_min,
                            "line_y_max": line_y_max,
                        })
                        counted.add(track_id)

            previous[track_id] = (x, y, time_seconds)

    output = args.source.parent / "crossing_events_near.csv"
    fields = [
        "track_id",
        "class",
        "direction",
        "previous_observation_seconds",
        "first_observed_after_crossing_seconds",
        "crossing_y_estimate",
        "line_x",
        "line_y_min",
        "line_y_max",
    ]

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    for direction in ("right_to_left", "left_to_right"):
        total = sum(event["direction"] == direction for event in events)
        print(f"{direction}: {total}")
    print(f"Saved: {output}")
if __name__ == "__main__":
    main()