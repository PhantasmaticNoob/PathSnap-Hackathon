
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "Dataset"
OUTPUT = BASE / "preprocessing" / "results"

OUTPUT.mkdir(parents=True, exist_ok=True)

GAP_THRESHOLD_SECONDS = 60


def load_track(path):
    files = list(path.glob("*.track"))

    if len(files) != 1:
        raise ValueError(f"Expected one .track file in {path}")

    return np.loadtxt(files[0], dtype=float, ndmin=2)


def inspect_trip(partition, folder):
    trip_id = int(folder.name)
    track = load_track(folder)

    issues = []

    if track.shape[1] != 3:
        raise ValueError(
            f"Trip {trip_id}: expected 3 columns, found {track.shape[1]}"
        )

    longitude = track[:, 0]
    latitude = track[:, 1]
    timestamp = track[:, 2]

    finite_rows = np.isfinite(track).all(axis=1)

    invalid_coordinates = (
        (~np.isfinite(longitude))
        | (~np.isfinite(latitude))
        | (longitude < -180)
        | (longitude > 180)
        | (latitude < -90)
        | (latitude > 90)
    )

    duplicate_rows = int(
        pd.DataFrame(track).duplicated().sum()
    )

    time_differences = np.diff(timestamp)

    non_increasing = int(
        np.sum(time_differences <= 0)
    )

    large_gaps = int(
        np.sum(time_differences > GAP_THRESHOLD_SECONDS)
    )

    for i in np.where(~finite_rows)[0]:
        issues.append({
            "partition": partition,
            "trip_id": trip_id,
            "observation_id": int(i),
            "issue": "non_finite_value"
        })

    for i in np.where(invalid_coordinates)[0]:
        issues.append({
            "partition": partition,
            "trip_id": trip_id,
            "observation_id": int(i),
            "issue": "invalid_coordinate"
        })

    for i in np.where(time_differences <= 0)[0]:
        issues.append({
            "partition": partition,
            "trip_id": trip_id,
            "observation_id": int(i + 1),
            "issue": "non_increasing_timestamp"
        })

    for i in np.where(time_differences > GAP_THRESHOLD_SECONDS)[0]:
        issues.append({
            "partition": partition,
            "trip_id": trip_id,
            "observation_id": int(i + 1),
            "issue": "gps_gap_over_60_seconds"
        })

    summary = {
        "partition": partition,
        "trip_id": trip_id,
        "observations": len(track),
        "non_finite_rows": int(np.sum(~finite_rows)),
        "invalid_coordinates": int(np.sum(invalid_coordinates)),
        "duplicate_observations": duplicate_rows,
        "non_increasing_timestamps": non_increasing,
        "gps_gaps_over_60_seconds": large_gaps,
        "first_timestamp": timestamp[0],
        "last_timestamp": timestamp[-1]
    }

    return summary, issues


def main():
    summaries = []
    all_issues = []

    for partition in ["train", "student_test"]:
        partition_folder = DATASET / partition

        if not partition_folder.exists():
            print(f"Skipping missing partition: {partition}")
            continue

        trip_folders = sorted(
            p for p in partition_folder.iterdir()
            if p.is_dir()
        )

        for folder in trip_folders:
            print(f"Checking {partition} trip {folder.name}")

            summary, issues = inspect_trip(partition, folder)

            summaries.append(summary)
            all_issues.extend(issues)

    summary_df = pd.DataFrame(summaries)
    issues_df = pd.DataFrame(
        all_issues,
        columns=[
            "partition",
            "trip_id",
            "observation_id",
            "issue"
        ]
    )

    summary_file = OUTPUT / "gps_validation_summary.csv"
    issues_file = OUTPUT / "gps_validation_issues.csv"

    summary_df.to_csv(summary_file, index=False)
    issues_df.to_csv(issues_file, index=False)

    print("\nGPS VALIDATION COMPLETED")
    print("--------------------------------")
    print("Trips checked:", len(summary_df))
    print(
        "Total observations:",
        summary_df["observations"].sum()
    )
    print(
        "Non-finite rows:",
        summary_df["non_finite_rows"].sum()
    )
    print(
        "Invalid coordinates:",
        summary_df["invalid_coordinates"].sum()
    )
    print(
        "Duplicate observations:",
        summary_df["duplicate_observations"].sum()
    )
    print(
        "Non-increasing timestamps:",
        summary_df["non_increasing_timestamps"].sum()
    )
    print(
        "GPS gaps over 60 seconds:",
        summary_df["gps_gaps_over_60_seconds"].sum()
    )

    print("\nSaved reports:")
    print(summary_file)
    print(issues_file)


if __name__ == "__main__":
    main()
