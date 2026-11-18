
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET = SCRIPT_DIR.parent / "dataset"


def trip_folder(partition, trip_id):
    return DATASET / partition / f"{int(trip_id):08d}"


def load_file(folder, extension):
    files = list(folder.glob(f"*{extension}"))

    if len(files) != 1:
        raise ValueError(
            f"Expected one {extension} file in {folder}, found {len(files)}"
        )

    return np.loadtxt(files[0], dtype=float, ndmin=2)


def haversine(lon1, lat1, lon2, lat2):
    radius = 6371000

    lon1, lat1, lon2, lat2 = np.radians(
        [lon1, lat1, lon2, lat2]
    )

    a = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2)
        * np.sin((lon2 - lon1) / 2) ** 2
    )

    return 2 * radius * np.arcsin(np.sqrt(a))


def nearest_gps(nodes, node_id, track):
    lon, lat = nodes[node_id]

    distances = np.array([
        haversine(lon, lat, point[0], point[1])
        for point in track
    ])

    index = int(np.argmin(distances))

    return index, distances[index]


def inspect_gps_gap(trip_id, observation):
    folder = trip_folder("train", trip_id)
    track = load_file(folder, ".track")

    print("\n" + "=" * 70)
    print(f"GPS GAP: TRAIN TRIP {trip_id}")
    print(f"Between observations {observation} and {observation + 1}")
    print("=" * 70)

    print("Index          Longitude       Latitude       Timestamp")

    start = max(0, observation - 2)
    end = min(len(track), observation + 4)

    for i in range(start, end):
        marker = ""

        if i == observation:
            marker = " <-- GAP START"
        elif i == observation + 1:
            marker = " <-- GAP END"

        print(
            f"{i:<14} {track[i, 0]:<15.7f} "
            f"{track[i, 1]:<15.7f} "
            f"{track[i, 2]:.1f}{marker}"
        )


def inspect_route_break(trip_id, position):
    folder = trip_folder("train", trip_id)

    nodes = load_file(folder, ".nodes")
    arcs = load_file(folder, ".arcs").astype(int)
    route = load_file(folder, ".route")[:, 0].astype(int)
    track = load_file(folder, ".track")

    print("\n" + "=" * 70)
    print(f"ROUTE BREAK: TRAIN TRIP {trip_id}, POSITION {position}")
    print("=" * 70)

    start = max(0, position - 2)
    end = min(len(route), position + 4)

    print("\nNearby route entries:")
    print("Position    Arc ID       From node    To node")

    for i in range(start, end):
        arc_id = route[i]
        from_node, to_node = arcs[arc_id]

        marker = ""

        if i == position:
            marker = " <-- PREVIOUS ARC"
        elif i == position + 1:
            marker = " <-- NEXT ARC"

        print(
            f"{i:<12} {arc_id:<12} "
            f"{from_node:<12} {to_node}{marker}"
        )

    previous_end = int(arcs[route[position], 1])
    next_start = int(arcs[route[position + 1], 0])

    lon1, lat1 = nodes[previous_end]
    lon2, lat2 = nodes[next_start]

    gap = haversine(lon1, lat1, lon2, lat2)

    print("\nDiscontinuity endpoints:")
    print(f"Previous endpoint node: {previous_end}")
    print(f"Next endpoint node: {next_start}")
    print(f"Distance between endpoints: {gap:.2f} metres")

    gps1, distance1 = nearest_gps(nodes, previous_end, track)
    gps2, distance2 = nearest_gps(nodes, next_start, track)

    print("\nNearest GPS observations:")
    print(
        f"Previous endpoint: observation {gps1}, "
        f"distance {distance1:.2f} metres"
    )
    print(
        f"Next endpoint: observation {gps2}, "
        f"distance {distance2:.2f} metres"
    )


def main():
    if not DATASET.is_dir():
        raise FileNotFoundError(f"Dataset not found: {DATASET}")

    print("INSPECTING EXTREME GPS GAPS")

    inspect_gps_gap(90, 534)
    inspect_gps_gap(78, 1)

    print("\nINSPECTING LARGEST ROUTE DISCONTINUITIES")

    inspect_route_break(19, 300)
    inspect_route_break(2, 53)
    inspect_route_break(19, 151)
    inspect_route_break(19, 94)
    inspect_route_break(80, 118)

    print("\nInspection complete.")


if __name__ == "__main__":
    main()
