
from pathlib import Path
import csv
import numpy as np

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "Dataset"
RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

EARTH_RADIUS_M = 6371000.0


def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(
        np.radians, [lon1, lat1, lon2, lat2]
    )

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )

    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def load_array(path, columns):
    data = np.loadtxt(path, dtype=float, ndmin=2)

    if data.shape[1] != columns:
        raise ValueError(
            f"{path} expected {columns} columns, got {data.shape[1]}"
        )

    return data


summary_rows = []
duplicate_rows = []

total_arcs = 0
total_duplicates = 0
total_zero_length = 0
total_self_loops = 0

for partition in ["train", "student_test"]:
    partition_dir = DATASET / partition

    if not partition_dir.exists():
        continue

    for trip_dir in sorted(partition_dir.iterdir()):
        if not trip_dir.is_dir():
            continue

        nodes_path = trip_dir / f"{trip_dir.name}.nodes"
        arcs_path = trip_dir / f"{trip_dir.name}.arcs"

        if not nodes_path.exists() or not arcs_path.exists():
            continue

        nodes = load_array(nodes_path, 2)
        arcs = load_array(arcs_path, 2).astype(np.int64)

        from_ids = arcs[:, 0]
        to_ids = arcs[:, 1]

        start = nodes[from_ids]
        end = nodes[to_ids]

        lengths = haversine(
            start[:, 0],
            start[:, 1],
            end[:, 0],
            end[:, 1]
        )

        unique_pairs, counts = np.unique(
            arcs, axis=0, return_counts=True
        )

        duplicate_pairs = unique_pairs[counts > 1]

        duplicate_count = int(np.sum(counts[counts > 1] - 1))
        zero_length_count = int(np.sum(lengths < 0.01))
        self_loop_count = int(np.sum(from_ids == to_ids))

        total_arcs += len(arcs)
        total_duplicates += duplicate_count
        total_zero_length += zero_length_count
        total_self_loops += self_loop_count

        summary_rows.append({
            "partition": partition,
            "trip_id": trip_dir.name,
            "node_count": len(nodes),
            "arc_count": len(arcs),
            "duplicate_directed_arcs": duplicate_count,
            "zero_length_arcs": zero_length_count,
            "self_loop_arcs": self_loop_count,
            "min_length_m": float(np.min(lengths)),
            "median_length_m": float(np.median(lengths)),
            "mean_length_m": float(np.mean(lengths)),
            "max_length_m": float(np.max(lengths)),
        })

        for pair in duplicate_pairs:
            matching_ids = np.where(
                (from_ids == pair[0]) & (to_ids == pair[1])
            )[0]

            for arc_id in matching_ids:
                duplicate_rows.append({
                    "partition": partition,
                    "trip_id": trip_dir.name,
                    "from_node_id": int(pair[0]),
                    "to_node_id": int(pair[1]),
                    "arc_id": int(arc_id),
                    "duplicate_count_for_pair": len(matching_ids),
                    "segment_length_m": float(lengths[arc_id]),
                })


summary_path = RESULTS / "road_geometry_summary.csv"
duplicates_path = RESULTS / "duplicate_arc_details.csv"

with open(summary_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
    writer.writeheader()
    writer.writerows(summary_rows)

duplicate_fields = [
    "partition",
    "trip_id",
    "from_node_id",
    "to_node_id",
    "arc_id",
    "duplicate_count_for_pair",
    "segment_length_m",
]

with open(duplicates_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=duplicate_fields)
    writer.writeheader()
    writer.writerows(duplicate_rows)


all_lengths = [
    row["mean_length_m"] for row in summary_rows
]

print("\nROAD GEOMETRY INSPECTION COMPLETED")
print("----------------------------------")
print(f"Trips inspected: {len(summary_rows)}")
print(f"Total arcs: {total_arcs}")
print(f"Duplicate directed arcs: {total_duplicates}")
print(f"Zero-length arcs (< 1 cm): {total_zero_length}")
print(f"Self-loop arcs: {total_self_loops}")

print("\nFiles saved:")
print(summary_path)
print(duplicates_path)

print("\nThis inspection does not modify the original dataset.")

