
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# ============================================================
# PATHSNAP - DATASET EXPLORATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

def find_dataset_root():
    """Find the folder containing both train and student_test."""

    possible_paths = [
        PROJECT_DIR / "dataset",
        PROJECT_DIR / "Student folder",
        PROJECT_DIR / "Student folder" / "Student folder",
        PROJECT_DIR,
    ]

    for path in possible_paths:
        if (path / "train").is_dir() and (path / "student_test").is_dir():
            return path

    # Search a few folder levels below the project directory.
    for current, dirs, files in os.walk(PROJECT_DIR):
        current_path = Path(current)
        depth = len(current_path.relative_to(PROJECT_DIR).parts)

        if depth > 3:
            dirs[:] = []
            continue

        if (current_path / "train").is_dir() and (
            current_path / "student_test"
        ).is_dir():
            return current_path

    raise FileNotFoundError(
        "Could not find a folder containing both 'train' and "
        "'student_test'. Check your project folder structure."
    )


def find_data_file(trip_folder, extension):
    """Find a file with the required extension inside a trip folder."""

    matches = list(trip_folder.glob(f"*{extension}"))

    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {extension} file in {trip_folder}, "
            f"but found {len(matches)}."
        )

    return matches[0]


def load_data(path):
    """Load whitespace-separated numerical data."""

    data = np.loadtxt(path, dtype=float, ndmin=2)

    if data.size == 0:
        raise ValueError(f"Empty file: {path}")

    return data


def explore_trip(trip_folder, partition):
    """Explore all four files belonging to one trip."""

    trip_id = trip_folder.name

    track_file = find_data_file(trip_folder, ".track")
    nodes_file = find_data_file(trip_folder, ".nodes")
    arcs_file = find_data_file(trip_folder, ".arcs")
    route_file = find_data_file(trip_folder, ".route")

    track = load_data(track_file)
    nodes = load_data(nodes_file)
    arcs = load_data(arcs_file)
    route = load_data(route_file)

    if track.shape[1] != 3:
        raise ValueError(f"{trip_id}: .track should have 3 columns.")

    if nodes.shape[1] != 2:
        raise ValueError(f"{trip_id}: .nodes should have 2 columns.")

    if arcs.shape[1] != 2:
        raise ValueError(f"{trip_id}: .arcs should have 2 columns.")

    if route.shape[1] != 1:
        raise ValueError(f"{trip_id}: .route should have 1 column.")

    # GPS data: longitude, latitude, timestamp
    longitude = track[:, 0]
    latitude = track[:, 1]
    timestamps = track[:, 2]

    # Road network data
    arc_from = arcs[:, 0].astype(np.int64)
    arc_to = arcs[:, 1].astype(np.int64)

    # Reference route
    route_ids = route[:, 0].astype(np.int64)

    # --------------------------------------------------------
    # GPS DATA QUALITY
    # --------------------------------------------------------

    gps_finite = np.isfinite(track).all(axis=1)

    valid_coordinates = (
        np.isfinite(longitude)
        & np.isfinite(latitude)
        & (longitude >= -180)
        & (longitude <= 180)
        & (latitude >= -90)
        & (latitude <= 90)
    )

    timestamp_differences = np.diff(timestamps)
    negative_time_steps = int(np.sum(timestamp_differences < 0))
    zero_time_steps = int(np.sum(timestamp_differences == 0))
    large_time_gaps = int(np.sum(timestamp_differences > 60))

    # --------------------------------------------------------
    # ROAD NETWORK VALIDATION
    # --------------------------------------------------------

    node_count = len(nodes)
    arc_count = len(arcs)

    valid_arc_endpoints = (
        (arc_from >= 0)
        & (arc_from < node_count)
        & (arc_to >= 0)
        & (arc_to < node_count)
    )

    valid_arc_count = int(np.sum(valid_arc_endpoints))
    invalid_arc_count = arc_count - valid_arc_count

    # Count isolated nodes (nodes with no incoming or outgoing arcs).
    degree = np.zeros(node_count, dtype=np.int64)

    valid_from = arc_from[valid_arc_endpoints]
    valid_to = arc_to[valid_arc_endpoints]

    np.add.at(degree, valid_from, 1)
    np.add.at(degree, valid_to, 1)

    isolated_nodes = int(np.sum(degree == 0))

    # Duplicate directed arcs
    unique_arcs = np.unique(arcs, axis=0)
    duplicate_arcs = arc_count - len(unique_arcs)

    # --------------------------------------------------------
    # REFERENCE ROUTE VALIDATION
    # --------------------------------------------------------

    valid_route_ids = (
        (route_ids >= 0)
        & (route_ids < arc_count)
    )

    invalid_route_ids = int(np.sum(~valid_route_ids))

    route_continuity_breaks = 0

    if len(route_ids) > 1 and np.all(valid_route_ids):
        route_arcs = arcs[route_ids].astype(np.int64)

        route_continuity_breaks = int(
            np.sum(route_arcs[:-1, 1] != route_arcs[1:, 0])
        )

    # --------------------------------------------------------
    # GEOGRAPHICAL EXTENT
    # --------------------------------------------------------

    valid_gps = valid_coordinates & gps_finite
    valid_nodes = (
        np.isfinite(nodes).all(axis=1)
        & (nodes[:, 0] >= -180)
        & (nodes[:, 0] <= 180)
        & (nodes[:, 1] >= -90)
        & (nodes[:, 1] <= 90)
    )

    if np.any(valid_gps):
        gps_lon_min = float(np.min(longitude[valid_gps]))
        gps_lon_max = float(np.max(longitude[valid_gps]))
        gps_lat_min = float(np.min(latitude[valid_gps]))
        gps_lat_max = float(np.max(latitude[valid_gps]))
    else:
        gps_lon_min = gps_lon_max = np.nan
        gps_lat_min = gps_lat_max = np.nan

    # --------------------------------------------------------
    # RETURN TRIP RESULTS
    # --------------------------------------------------------

    return {
        "partition": partition,
        "trip_id": trip_id,
        "gps_observations": len(track),
        "road_nodes": node_count,
        "road_arcs": arc_count,
        "route_length": len(route_ids),
        "non_finite_gps_rows": int(np.sum(~gps_finite)),
        "invalid_gps_coordinates": int(np.sum(~valid_coordinates)),
        "invalid_node_coordinates": int(np.sum(~valid_nodes)),
        "negative_timestamp_steps": negative_time_steps,
        "zero_timestamp_steps": zero_time_steps,
        "gaps_over_60_seconds": large_time_gaps,
        "invalid_arc_endpoints": invalid_arc_count,
        "isolated_nodes": isolated_nodes,
        "duplicate_directed_arcs": duplicate_arcs,
        "invalid_route_ids": invalid_route_ids,
        "route_continuity_breaks": route_continuity_breaks,
        "gps_longitude_min": gps_lon_min,
        "gps_longitude_max": gps_lon_max,
        "gps_latitude_min": gps_lat_min,
        "gps_latitude_max": gps_lat_max,
    }


def main():

    print("=" * 65)
    print("PATHSNAP DATASET EXPLORATION")
    print("=" * 65)

    dataset_root = find_dataset_root()

    print(f"\nDataset found at: {dataset_root}")

    all_results = []

    for partition in ["train", "student_test"]:

        partition_path = dataset_root / partition

        trip_folders = sorted(
            [folder for folder in partition_path.iterdir() if folder.is_dir()]
        )

        print(f"\nExploring {partition}: {len(trip_folders)} trips")

        for index, trip_folder in enumerate(trip_folders, start=1):

            try:
                result = explore_trip(trip_folder, partition)
                all_results.append(result)

                print(
                    f"  [{index}/{len(trip_folders)}] "
                    f"Trip {trip_folder.name}: OK"
                )

            except Exception as error:
                print(
                    f"  ERROR in trip {trip_folder.name}: {error}"
                )

    if not all_results:
        print("\nNo trips could be analysed.")
        return

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    results = pd.DataFrame(all_results)

    output_dir = SCRIPT_DIR / "eda_results"
    output_dir.mkdir(exist_ok=True)

    results.to_csv(
        output_dir / "trip_inventory.csv",
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary_lines = []

    summary_lines.append("PATHSNAP DATASET EXPLORATION SUMMARY")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Dataset location: {dataset_root}")
    summary_lines.append(f"Total trips analysed: {len(results)}")

    for partition in ["train", "student_test"]:

        subset = results[results["partition"] == partition]

        summary_lines.append("")
        summary_lines.append(f"{partition.upper()} SUMMARY")
        summary_lines.append("-" * 40)
        summary_lines.append(f"Trips analysed: {len(subset)}")

        if len(subset) == 0:
            continue

        columns = [
            "gps_observations",
            "road_nodes",
            "road_arcs",
            "route_length",
        ]

        for column in columns:
            summary_lines.append(
                f"Total {column}: {int(subset[column].sum()):,}"
            )

        summary_lines.append(
            f"Trips with GPS gaps over 60 seconds: "
            f"{int((subset['gaps_over_60_seconds'] > 0).sum())}"
        )

        summary_lines.append(
            f"Trips with isolated nodes: "
            f"{int((subset['isolated_nodes'] > 0).sum())}"
        )

        summary_lines.append(
            f"Trips with duplicate arcs: "
            f"{int((subset['duplicate_directed_arcs'] > 0).sum())}"
        )

        summary_lines.append(
            f"Trips with route continuity breaks: "
            f"{int((subset['route_continuity_breaks'] > 0).sum())}"
        )

    summary_text = "\n".join(summary_lines)

    with open(
        output_dir / "eda_summary.txt",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(summary_text)

    # --------------------------------------------------------
    # GPS OBSERVATION DISTRIBUTION
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for partition in ["train", "student_test"]:

        subset = results[results["partition"] == partition]

        if len(subset) > 0:
            plt.hist(
                subset["gps_observations"],
                bins=15,
                alpha=0.6,
                label=partition
            )

    plt.xlabel("Number of GPS observations per trip")
    plt.ylabel("Number of trips")
    plt.title("GPS Observation Distribution")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "gps_observation_distribution.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # ROAD NETWORK SIZE DISTRIBUTION
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    for partition in ["train", "student_test"]:

        subset = results[results["partition"] == partition]

        if len(subset) > 0:
            plt.scatter(
                subset["road_nodes"],
                subset["road_arcs"],
                label=partition,
                alpha=0.7
            )

    plt.xlabel("Number of road nodes")
    plt.ylabel("Number of road arcs")
    plt.title("Road Network Size per Trip")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output_dir / "road_network_sizes.png",
        dpi=200
    )

    plt.close()

    print("\n" + "=" * 65)
    print(summary_text)
    print("=" * 65)

    print("\nExploration completed.")
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
