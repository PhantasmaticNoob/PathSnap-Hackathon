
from pathlib import Path
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent


def find_dataset():
    candidates = [
        PROJECT_DIR / "dataset",
        PROJECT_DIR / "Student folder",
        PROJECT_DIR / "Student folder" / "Student folder",
        PROJECT_DIR,
    ]

    for path in candidates:
        if (path / "train").is_dir() and (path / "student_test").is_dir():
            return path

    for current, dirs, files in __import__("os").walk(PROJECT_DIR):
        path = Path(current)
        if len(path.relative_to(PROJECT_DIR).parts) > 3:
            dirs[:] = []
            continue
        if (path / "train").is_dir() and (path / "student_test").is_dir():
            return path

    raise FileNotFoundError("Dataset folder not found.")


def load_file(folder, extension):
    matches = list(folder.glob(f"*{extension}"))
    if len(matches) != 1:
        raise ValueError(
            f"Expected one {extension} file in {folder}, found {len(matches)}"
        )
    return np.loadtxt(matches[0], dtype=float, ndmin=2)


def haversine(lon1, lat1, lon2, lat2):
    """Calculate great-circle distance in metres."""

    earth_radius = 6371000.0

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * earth_radius * np.arcsin(np.sqrt(a))


def main():
    dataset = find_dataset()

    output_dir = SCRIPT_DIR / "eda_results"
    output_dir.mkdir(exist_ok=True)

    route_breaks = []
    gps_gaps = []

    for partition in ["train", "student_test"]:
        partition_dir = dataset / partition

        for trip_folder in sorted(partition_dir.iterdir()):
            if not trip_folder.is_dir():
                continue

            trip_id = trip_folder.name

            track = load_file(trip_folder, ".track")
            nodes = load_file(trip_folder, ".nodes")
            arcs = load_file(trip_folder, ".arcs")
            route = load_file(trip_folder, ".route")[:, 0].astype(int)

            # --------------------------------------------
            # 1. ROUTE DISCONTINUITY INVESTIGATION
            # --------------------------------------------

            if len(route) > 1:
                route_arcs = arcs[route].astype(int)

                for i in range(len(route_arcs) - 1):
                    previous_to = route_arcs[i, 1]
                    next_from = route_arcs[i + 1, 0]

                    if previous_to != next_from:
                        lon1, lat1 = nodes[previous_to]
                        lon2, lat2 = nodes[next_from]

                        distance = haversine(
                            lon1, lat1, lon2, lat2
                        )

                        route_breaks.append({
                            "partition": partition,
                            "trip_id": trip_id,
                            "route_position": i,
                            "previous_arc_id": int(route[i]),
                            "next_arc_id": int(route[i + 1]),
                            "previous_end_node": int(previous_to),
                            "next_start_node": int(next_from),
                            "endpoint_gap_metres": float(distance),
                        })

            # --------------------------------------------
            # 2. GPS TIME GAP INVESTIGATION
            # --------------------------------------------

            timestamps = track[:, 2]
            time_differences = np.diff(timestamps)

            for i, gap in enumerate(time_differences):
                if gap > 60:
                    distance = haversine(
                        track[i, 0],
                        track[i, 1],
                        track[i + 1, 0],
                        track[i + 1, 1],
                    )

                    speed = (distance / gap) * 3.6

                    gps_gaps.append({
                        "partition": partition,
                        "trip_id": trip_id,
                        "start_observation": i,
                        "end_observation": i + 1,
                        "start_timestamp": timestamps[i],
                        "end_timestamp": timestamps[i + 1],
                        "gap_seconds": float(gap),
                        "gps_displacement_metres": float(distance),
                        "implied_speed_kmh": float(speed),
                    })

    # --------------------------------------------
    # SAVE DETAILED RESULTS
    # --------------------------------------------

    route_df = pd.DataFrame(route_breaks)
    gps_df = pd.DataFrame(gps_gaps)

    route_df.to_csv(
        output_dir / "route_break_details.csv",
        index=False
    )

    gps_df.to_csv(
        output_dir / "gps_gap_details.csv",
        index=False
    )

    print("\n" + "=" * 65)
    print("ROUTE DISCONTINUITY DETAILS")
    print("=" * 65)

    if route_df.empty:
        print("No route discontinuities found.")
    else:
        print(route_df.to_string(index=False))
        print("\nEndpoint gap distance statistics (metres):")
        print(route_df["endpoint_gap_metres"].describe())

    print("\n" + "=" * 65)
    print("GPS GAP DETAILS")
    print("=" * 65)

    if gps_df.empty:
        print("No GPS gaps exceeding 60 seconds found.")
    else:
        print(gps_df.to_string(index=False))
        print("\nGPS gap duration statistics (seconds):")
        print(gps_df["gap_seconds"].describe())

        print("\nGPS displacement statistics (metres):")
        print(gps_df["gps_displacement_metres"].describe())

        print("\nImplied speed statistics (km/h):")
        print(gps_df["implied_speed_kmh"].describe())

    print("\nDetailed files saved in:", output_dir)


if __name__ == "__main__":
    main()

