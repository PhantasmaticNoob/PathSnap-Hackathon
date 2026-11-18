
from pathlib import Path
import csv
import math
from collections import defaultdict

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "preprocessing" / "results"

CANDIDATE_FILE = RESULTS_DIR / "road_candidates.csv"
SUMMARY_FILE = RESULTS_DIR / "candidate_generation_summary.csv"
VALIDATION_FILE = RESULTS_DIR / "candidate_validation_summary.csv"

EXPECTED_K = 5


def main():
    expected = {}

    with open(SUMMARY_FILE, newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            key = (row["partition"], row["trip_id"])
            expected[key] = int(row["gps_observations"])

    stats = defaultdict(lambda: {
        "observations": 0,
        "candidate_rows": 0,
        "incorrect_counts": 0,
        "incorrect_ranks": 0,
        "unsorted_distances": 0,
        "invalid_distances": 0,
        "duplicate_arc_ids": 0,
        "distance_values": []
    })

    total_rows = 0
    current_observation = None
    current_candidates = []

    def finish_observation(observation_key, candidates):
        if observation_key is None:
            return

        partition, trip_id, observation_id = observation_key
        trip_key = (partition, trip_id)

        result = stats[trip_key]
        result["observations"] += 1

        if len(candidates) != EXPECTED_K:
            result["incorrect_counts"] += 1

        ranks = [item[0] for item in candidates]
        distances = [item[1] for item in candidates]
        arc_ids = [item[2] for item in candidates]

        if sorted(ranks) != list(range(1, EXPECTED_K + 1)):
            result["incorrect_ranks"] += 1

        if len(set(arc_ids)) != len(arc_ids):
            result["duplicate_arc_ids"] += 1

        if any(
            not math.isfinite(value) or value < 0
            for value in distances
        ):
            result["invalid_distances"] += 1

        elif distances != sorted(distances):
            result["unsorted_distances"] += 1

    print("CANDIDATE VALIDATION STARTED")
    print("--------------------------------")

    with open(
        CANDIDATE_FILE,
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            total_rows += 1

            partition = row["partition"]
            trip_id = row["trip_id"]
            observation_id = int(row["observation_id"])

            trip_key = (partition, trip_id)
            observation_key = (
                partition,
                trip_id,
                observation_id
            )

            if trip_key not in expected:
                raise ValueError(
                    f"Unexpected trip: {trip_key}"
                )

            rank = int(row["candidate_rank"])
            distance_metres = float(row["distance_metres"])
            arc_id = int(row["arc_id"])

            stats[trip_key]["candidate_rows"] += 1
            stats[trip_key]["distance_values"].append(
                distance_metres
            )

            if observation_key != current_observation:

                finish_observation(
                    current_observation,
                    current_candidates
                )

                current_observation = observation_key
                current_candidates = []

            current_candidates.append((
                rank,
                distance_metres,
                arc_id
            ))

        finish_observation(
            current_observation,
            current_candidates
        )

    total_expected = sum(expected.values())

    validation_rows = []

    total_found = 0
    total_bad_counts = 0
    total_bad_ranks = 0
    total_bad_order = 0
    total_bad_distances = 0
    total_duplicate_arcs = 0
    total_missing = 0
    all_distances = []

    for key, expected_count in sorted(expected.items()):

        partition, trip_id = key
        result = stats[key]

        found = result["observations"]
        missing = max(0, expected_count - found)
        extra = max(0, found - expected_count)

        distances = np.asarray(
            result["distance_values"],
            dtype=float
        )

        finite = distances[
            np.isfinite(distances) & (distances >= 0)
        ]

        all_distances.extend(finite.tolist())

        total_found += found
        total_missing += missing
        total_bad_counts += result["incorrect_counts"]
        total_bad_ranks += result["incorrect_ranks"]
        total_bad_order += result["unsorted_distances"]
        total_bad_distances += result["invalid_distances"]
        total_duplicate_arcs += result["duplicate_arc_ids"]

        validation_rows.append({
            "partition": partition,
            "trip_id": trip_id,
            "expected_observations": expected_count,
            "observations_found": found,
            "missing_observations": missing,
            "extra_observations": extra,
            "candidate_rows": result["candidate_rows"],
            "incorrect_candidate_counts": result["incorrect_counts"],
            "incorrect_ranks": result["incorrect_ranks"],
            "unsorted_distances": result["unsorted_distances"],
            "invalid_distances": result["invalid_distances"],
            "duplicate_arc_ids": result["duplicate_arc_ids"],
            "mean_candidate_distance_metres": (
                float(np.mean(finite)) if len(finite) else 0.0
            )
        })

    with open(
        VALIDATION_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=list(validation_rows[0].keys())
        )

        writer.writeheader()
        writer.writerows(validation_rows)

    print(f"Candidate rows inspected: {total_rows}")
    print(f"Expected GPS observations: {total_expected}")
    print(f"Observations found: {total_found}")
    print(f"Missing observations: {total_missing}")
    print(f"Incorrect candidate counts: {total_bad_counts}")
    print(f"Incorrect ranks: {total_bad_ranks}")
    print(f"Unsorted distances: {total_bad_order}")
    print(f"Invalid distances: {total_bad_distances}")
    print(f"Observations with duplicate arc IDs: {total_duplicate_arcs}")

    if all_distances:
        values = np.asarray(all_distances)

        print()
        print("CANDIDATE DISTANCE DISTRIBUTION")
        print("--------------------------------")
        print(f"Minimum: {np.min(values):.3f} m")
        print(f"Median: {np.median(values):.3f} m")
        print(f"Mean: {np.mean(values):.3f} m")
        print(f"90th percentile: {np.percentile(values, 90):.3f} m")
        print(f"99th percentile: {np.percentile(values, 99):.3f} m")
        print(f"Maximum: {np.max(values):.3f} m")

    passed = (
        total_rows == EXPECTED_K * total_expected
        and total_found == total_expected
        and total_missing == 0
        and total_bad_counts == 0
        and total_bad_ranks == 0
        and total_bad_order == 0
        and total_bad_distances == 0
        and total_duplicate_arcs == 0
    )

    print()
    print(f"Report saved: {VALIDATION_FILE}")

    if passed:
        print()
        print("CANDIDATE STRUCTURAL VALIDATION PASSED")
    else:
        print()
        print("CANDIDATE STRUCTURAL VALIDATION FAILED")


if __name__ == "__main__":
    main()
