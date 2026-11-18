
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent.parent
DATASET = BASE / "Dataset"
OUTPUT = BASE / "preprocessing" / "results"

OUTPUT.mkdir(parents=True, exist_ok=True)


def load_file(folder, extension):
    files = list(folder.glob(f"*{extension}"))

    if len(files) != 1:
        raise ValueError(
            f"Expected one {extension} file in {folder}"
        )

    return np.loadtxt(files[0], dtype=float, ndmin=2)


def inspect_trip(partition, folder):
    trip_id = int(folder.name)

    nodes = load_file(folder, ".nodes")
    arcs = load_file(folder, ".arcs")

    if nodes.shape[1] != 2:
        raise ValueError(f"Trip {trip_id}: invalid node format")

    if arcs.shape[1] != 2:
        raise ValueError(f"Trip {trip_id}: invalid arc format")

    node_finite = np.isfinite(nodes).all(axis=1)

    invalid_coordinates = (
        (~node_finite)
        | (nodes[:, 0] < -180)
        | (nodes[:, 0] > 180)
        | (nodes[:, 1] < -90)
        | (nodes[:, 1] > 90)
    )

    arc_finite = np.isfinite(arcs).all(axis=1)
    integer_arcs = np.zeros(len(arcs), dtype=bool)

    integer_arcs[arc_finite] = np.equal(
        arcs[arc_finite],
        np.floor(arcs[arc_finite])
    ).all(axis=1)

    valid_arc_format = arc_finite & integer_arcs

    safe_arcs = np.zeros_like(arcs, dtype=np.int64)
    safe_arcs[valid_arc_format] = arcs[
        valid_arc_format
    ].astype(np.int64)

    invalid_node_references = np.zeros(len(arcs), dtype=bool)

    invalid_node_references[valid_arc_format] = (
        (safe_arcs[valid_arc_format] < 0)
        | (safe_arcs[valid_arc_format] >= len(nodes))
    ).any(axis=1)

    invalid_arcs = (~valid_arc_format) | invalid_node_references

    valid = ~invalid_arcs

    duplicate_connections = 0

    if valid.any():
        valid_pairs = safe_arcs[valid]
        duplicate_connections = int(
            pd.DataFrame(valid_pairs).duplicated().sum()
        )

    summary = {
        "partition": partition,
        "trip_id": trip_id,
        "node_count": len(nodes),
        "arc_count": len(arcs),
        "invalid_node_coordinates": int(
            invalid_coordinates.sum()
        ),
        "invalid_arc_format": int(
            (~valid_arc_format).sum()
        ),
        "arcs_with_invalid_node_references": int(
            invalid_node_references.sum()
        ),
        "duplicate_directed_connections": duplicate_connections
    }

    return summary


def main():
    summaries = []

    for partition in ["train", "student_test"]:
        partition_folder = DATASET / partition

        if not partition_folder.exists():
            continue

        for folder in sorted(partition_folder.iterdir()):
            if not folder.is_dir():
                continue

            print(
                f"Checking road network: "
                f"{partition} trip {folder.name}"
            )

            summaries.append(
                inspect_trip(partition, folder)
            )

    df = pd.DataFrame(summaries)

    output_file = OUTPUT / "road_validation_summary.csv"
    df.to_csv(output_file, index=False)

    print("\nROAD NETWORK VALIDATION COMPLETED")
    print("--------------------------------")

    for column in [
        "node_count",
        "arc_count",
        "invalid_node_coordinates",
        "invalid_arc_format",
        "arcs_with_invalid_node_references",
        "duplicate_directed_connections"
    ]:
        print(f"{column}: {df[column].sum()}")

    print("\nTrips checked:", len(df))
    print("Report saved:", output_file)


if __name__ == "__main__":
    main()
