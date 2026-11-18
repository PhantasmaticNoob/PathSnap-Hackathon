
from pathlib import Path
import csv
import sys
import time

import numpy as np
import shapely
from pyproj import CRS, Transformer
from shapely import STRtree


# =========================================================
# CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = BASE_DIR / "Dataset"
RESULTS_DIR = BASE_DIR / "preprocessing" / "results"

PARTITIONS = ["train", "student_test"]

TOP_K = 5
BATCH_SIZE = 2000

INITIAL_RADIUS = 25.0
MAX_RADIUS = 100000.0

ZERO_LENGTH_THRESHOLD = 0.01

CANDIDATE_FILE = RESULTS_DIR / "road_candidates.csv"
SUMMARY_FILE = RESULTS_DIR / "candidate_generation_summary.csv"


# =========================================================
# LOAD DATA
# =========================================================

def find_data_file(trip_dir, extension):
    matches = list(trip_dir.glob(f"*{extension}"))

    if not matches:
        raise FileNotFoundError(
            f"Missing {extension} file in {trip_dir}"
        )

    for path in matches:
        if path.stem == trip_dir.name:
            return path

    if len(matches) == 1:
        return matches[0]

    raise RuntimeError(
        f"Multiple {extension} files found in {trip_dir}"
    )


def load_table(path, columns):
    data = np.loadtxt(path, dtype=float, ndmin=2)

    if data.shape[1] != columns:
        raise ValueError(
            f"{path}: expected {columns} columns, "
            f"found {data.shape[1]}"
        )

    if not np.all(np.isfinite(data)):
        raise ValueError(f"Non-finite values in {path}")

    return data


# =========================================================
# PROJECTION
# =========================================================

def create_projector(nodes):
    center_lon = float(np.mean(nodes[:, 0]))
    center_lat = float(np.mean(nodes[:, 1]))

    local_crs = CRS.from_proj4(
        f"+proj=aeqd +lat_0={center_lat} "
        f"+lon_0={center_lon} +datum=WGS84 "
        f"+units=m +no_defs"
    )

    return Transformer.from_crs(
        "EPSG:4326",
        local_crs,
        always_xy=True
    )


# =========================================================
# ROAD SEGMENTS
# =========================================================

def build_segments(nodes, arcs, transformer):
    node_x, node_y = transformer.transform(
        nodes[:, 0],
        nodes[:, 1]
    )

    projected_nodes = np.column_stack((node_x, node_y))

    from_ids = arcs[:, 0]
    to_ids = arcs[:, 1]

    integer_mask = (
        (from_ids == np.floor(from_ids))
        & (to_ids == np.floor(to_ids))
    )

    valid_reference_mask = (
        integer_mask
        & (from_ids >= 0)
        & (to_ids >= 0)
        & (from_ids < len(nodes))
        & (to_ids < len(nodes))
    )

    valid_arc_ids = np.flatnonzero(valid_reference_mask)

    valid_from = from_ids[valid_reference_mask].astype(np.int64)
    valid_to = to_ids[valid_reference_mask].astype(np.int64)

    starts = projected_nodes[valid_from]
    ends = projected_nodes[valid_to]

    lengths = np.hypot(
        ends[:, 0] - starts[:, 0],
        ends[:, 1] - starts[:, 1]
    )

    usable_mask = lengths >= ZERO_LENGTH_THRESHOLD

    usable_arc_ids = valid_arc_ids[usable_mask]

    starts = starts[usable_mask]
    ends = ends[usable_mask]

    segments = shapely.linestrings(
        np.stack((starts, ends), axis=1)
    )

    invalid_arcs = len(arcs) - len(valid_arc_ids)
    zero_length_arcs = int(np.count_nonzero(~usable_mask))
    self_loops = int(np.count_nonzero(
        valid_from[usable_mask] == valid_to[usable_mask]
    ))

    return (
        segments,
        usable_arc_ids.astype(np.int64),
        invalid_arcs,
        zero_length_arcs,
        self_loops
    )


# =========================================================
# TOP-K SEARCH FOR ONE BATCH
# =========================================================

def search_batch(points, tree, segments, arc_ids):
    batch_size = len(points)

    output = [[] for _ in range(batch_size)]

    unresolved = np.arange(batch_size, dtype=np.int64)
    radius = INITIAL_RADIUS

    while len(unresolved) > 0 and radius <= MAX_RADIUS:

        pairs = tree.query(
            points[unresolved],
            predicate="dwithin",
            distance=radius
        )

        if pairs.size == 0:
            radius *= 2
            continue

        point_indices = pairs[0]
        segment_indices = pairs[1]

        candidate_distances = shapely.distance(
            points[unresolved[point_indices]],
            segments[segment_indices]
        )

        counts = np.bincount(
            point_indices,
            minlength=len(unresolved)
        )

        resolved_mask = counts >= TOP_K

        # Process only observations that now have at least
        # TOP_K candidates inside the search radius.
        for local_point_index in np.flatnonzero(resolved_mask):

            mask = point_indices == local_point_index

            selected_segments = segment_indices[mask]
            selected_distances = candidate_distances[mask]

            order = np.lexsort((
                arc_ids[selected_segments],
                selected_distances
            ))

            chosen = order[:TOP_K]

            output_index = int(unresolved[local_point_index])

            output[output_index] = [
                (
                    int(arc_ids[selected_segments[position]]),
                    float(selected_distances[position]),
                    rank
                )
                for rank, position in enumerate(chosen, start=1)
            ]

        unresolved = unresolved[~resolved_mask]
        radius *= 2

    # Rare fallback: if a road network is so sparse that
    # an observation still has fewer than five candidates
    # after the maximum radius, search all segments for it.
    for point_index in unresolved:

        point = points[point_index]

        distances = shapely.distance(segments, point)

        order = np.lexsort((arc_ids, distances))
        chosen = order[:TOP_K]

        output[point_index] = [
            (
                int(arc_ids[position]),
                float(distances[position]),
                rank
            )
            for rank, position in enumerate(chosen, start=1)
        ]

    return output


# =========================================================
# PROCESS ONE TRIP
# =========================================================

def process_trip(partition, trip_dir, writer):
    start_time = time.perf_counter()

    track = load_table(
        find_data_file(trip_dir, ".track"), 3
    )

    nodes = load_table(
        find_data_file(trip_dir, ".nodes"), 2
    )

    arcs = load_table(
        find_data_file(trip_dir, ".arcs"), 2
    )

    transformer = create_projector(nodes)

    (
        segments,
        arc_ids,
        invalid_arcs,
        zero_length_arcs,
        self_loops
    ) = build_segments(nodes, arcs, transformer)

    tree = STRtree(segments)

    gps_x, gps_y = transformer.transform(
        track[:, 0],
        track[:, 1]
    )

    gps_points = shapely.points(gps_x, gps_y)

    candidate_count = 0
    observations_with_candidates = 0
    distance_sum = 0.0

    for start in range(0, len(track), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(track))

        batch_points = gps_points[start:end]

        batch_results = search_batch(
            batch_points,
            tree,
            segments,
            arc_ids
        )

        for offset, candidates in enumerate(batch_results):

            observation_id = start + offset
            timestamp = float(track[observation_id, 2])

            if candidates:
                observations_with_candidates += 1

            for arc_id, distance_metres, rank in candidates:

                writer.writerow([
                    partition,
                    trip_dir.name,
                    observation_id,
                    timestamp,
                    arc_id,
                    distance_metres,
                    rank
                ])

                candidate_count += 1
                distance_sum += distance_metres

    elapsed = time.perf_counter() - start_time

    return {
        "partition": partition,
        "trip_id": trip_dir.name,
        "gps_observations": len(track),
        "road_nodes": len(nodes),
        "road_arcs": len(arcs),
        "usable_road_segments": len(segments),
        "invalid_arcs_skipped": invalid_arcs,
        "zero_length_arcs_skipped": zero_length_arcs,
        "self_loop_arcs": self_loops,
        "observations_with_candidates": observations_with_candidates,
        "candidate_rows": candidate_count,
        "mean_candidate_distance_metres": (
            distance_sum / candidate_count
            if candidate_count else 0.0
        ),
        "processing_seconds": round(elapsed, 3)
    }


# =========================================================
# MAIN
# =========================================================

def main():
    overall_start = time.perf_counter()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    trips = []

    for partition in PARTITIONS:

        partition_dir = DATASET_DIR / partition

        if not partition_dir.exists():
            raise FileNotFoundError(partition_dir)

        for trip_dir in sorted(partition_dir.iterdir()):

            if trip_dir.is_dir():
                trips.append((partition, trip_dir))

    if not trips:
        raise RuntimeError("No trips found.")

    summaries = []

    print("OPTIMISED CANDIDATE GENERATION")
    print("--------------------------------")
    print(f"Trips: {len(trips)}")
    print(f"Top-K: {TOP_K}")
    print(f"Batch size: {BATCH_SIZE}")
    print()

    with open(
        CANDIDATE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as candidate_handle:

        writer = csv.writer(candidate_handle)

        writer.writerow([
            "partition",
            "trip_id",
            "observation_id",
            "timestamp",
            "arc_id",
            "distance_metres",
            "candidate_rank"
        ])

        for index, (partition, trip_dir) in enumerate(
            trips, start=1
        ):

            print(
                f"[{index}/{len(trips)}] "
                f"{partition}/{trip_dir.name}",
                flush=True
            )

            summary = process_trip(
                partition,
                trip_dir,
                writer
            )

            summaries.append(summary)

            candidate_handle.flush()

            print(
                f"  GPS: {summary['gps_observations']} | "
                f"Candidates: {summary['candidate_rows']} | "
                f"Time: {summary['processing_seconds']} sec",
                flush=True
            )

    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as summary_handle:

        writer = csv.DictWriter(
            summary_handle,
            fieldnames=list(summaries[0].keys())
        )

        writer.writeheader()
        writer.writerows(summaries)

    total_observations = sum(
        row["gps_observations"] for row in summaries
    )

    total_candidates = sum(
        row["candidate_rows"] for row in summaries
    )

    total_segments = sum(
        row["usable_road_segments"] for row in summaries
    )

    total_time = time.perf_counter() - overall_start

    print()
    print("CANDIDATE GENERATION COMPLETED")
    print("--------------------------------")
    print(f"Trips processed: {len(summaries)}")
    print(f"GPS observations: {total_observations}")
    print(f"Usable road segments: {total_segments}")
    print(f"Candidate rows: {total_candidates}")
    print(f"Total elapsed time: {total_time:.2f} seconds")

    print()
    print("Files saved:")
    print(CANDIDATE_FILE)
    print(SUMMARY_FILE)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print()
        print("CANDIDATE GENERATION FAILED")
        print("--------------------------------")
        print(f"{type(error).__name__}: {error}")
        sys.exit(1)
