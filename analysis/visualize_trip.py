
from pathlib import Path
import numpy as np
import pandas as pd
import folium
from folium import FeatureGroup, LayerControl

BASE = Path(__file__).resolve().parent
DATASET = BASE.parent / "dataset"
RESULTS = BASE / "eda_results"

GPS_FILE = RESULTS / "gps_gap_details.csv"
ROUTE_FILE = RESULTS / "route_break_details.csv"

OUTPUT = RESULTS / "anomaly_trips_map.html"


def load_data(folder, extension):
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


def trip_path(partition, trip_id):
    return DATASET / partition / f"{int(trip_id):08d}"


def add_local_network(group, nodes, arcs, track):
    # Display a limited local sample of the road network.
    min_lon = track[:, 0].min() - 0.002
    max_lon = track[:, 0].max() + 0.002
    min_lat = track[:, 1].min() - 0.002
    max_lat = track[:, 1].max() + 0.002

    start_nodes = arcs[:, 0]
    end_nodes = arcs[:, 1]

    start_coords = nodes[start_nodes]
    end_coords = nodes[end_nodes]

    mask = (
        (start_coords[:, 0] >= min_lon)
        & (start_coords[:, 0] <= max_lon)
        & (start_coords[:, 1] >= min_lat)
        & (start_coords[:, 1] <= max_lat)
        & (end_coords[:, 0] >= min_lon)
        & (end_coords[:, 0] <= max_lon)
        & (end_coords[:, 1] >= min_lat)
        & (end_coords[:, 1] <= max_lat)
    )

    selected = np.flatnonzero(mask)

    # Limit map size while retaining a representative network sample.
    if len(selected) > 350:
        selected = selected[
            np.linspace(0, len(selected) - 1, 350, dtype=int)
        ]

    for arc_id in selected:
        a = nodes[start_nodes[arc_id]]
        b = nodes[end_nodes[arc_id]]

        folium.PolyLine(
            locations=[
                [a[1], a[0]],
                [b[1], b[0]]
            ],
            color="gray",
            weight=1,
            opacity=0.35
        ).add_to(group)


def add_gps(group, track, gaps):
    # Downsample the trajectory for browser performance.
    step = max(1, len(track) // 1500)
    indices = list(range(0, len(track), step))

    if len(track) - 1 not in indices:
        indices.append(len(track) - 1)

    points = [
        [track[i, 1], track[i, 0]]
        for i in indices
    ]

    folium.PolyLine(
        points,
        color="blue",
        weight=3,
        opacity=0.85,
        tooltip="GPS trajectory"
    ).add_to(group)

    # Mark every GPS gap identified during exploration.
    for _, gap in gaps.iterrows():
        i = int(gap["start_observation"])

        if i + 1 >= len(track):
            continue

        p1 = track[i]
        p2 = track[i + 1]

        popup = (
            f"GPS recording gap<br>"
            f"Duration: {gap['gap_seconds']:.0f} seconds<br>"
            f"Displacement: "
            f"{gap['gps_displacement_metres']:.2f} metres"
        )

        folium.PolyLine(
            [
                [p1[1], p1[0]],
                [p2[1], p2[0]]
            ],
            color="orange",
            weight=4,
            dash_array="6, 8",
            tooltip=popup
        ).add_to(group)

        folium.CircleMarker(
            [p1[1], p1[0]],
            radius=5,
            color="orange",
            fill=True,
            fill_opacity=1,
            tooltip=f"GPS gap starts: observation {i}"
        ).add_to(group)

        folium.CircleMarker(
            [p2[1], p2[0]],
            radius=5,
            color="darkred",
            fill=True,
            fill_opacity=1,
            tooltip=f"GPS gap ends: observation {i + 1}"
        ).add_to(group)


def add_reference_route(group, nodes, arcs, route, breaks):
    # Draw continuous route portions separately so that
    # disconnected reference segments are not falsely connected.
    runs = []
    current_run = []

    for arc_id in route:
        start, end = arcs[int(arc_id)]

        start_point = nodes[start]
        end_point = nodes[end]

        if not current_run:
            current_run = [
                [start_point[1], start_point[0]],
                [end_point[1], end_point[0]]
            ]

        elif (
            abs(current_run[-1][0] - start_point[1]) < 1e-10
            and abs(current_run[-1][1] - start_point[0]) < 1e-10
        ):
            current_run.append([end_point[1], end_point[0]])

        else:
            runs.append(current_run)
            current_run = [
                [start_point[1], start_point[0]],
                [end_point[1], end_point[0]]
            ]

    if current_run:
        runs.append(current_run)

    for run in runs:
        folium.PolyLine(
            run,
            color="green",
            weight=4,
            opacity=0.8,
            tooltip="Reference route"
        ).add_to(group)

    # Highlight each reference-route discontinuity.
    for _, row in breaks.iterrows():
        previous_node = int(row["previous_end_node"])
        next_node = int(row["next_start_node"])

        a = nodes[previous_node]
        b = nodes[next_node]

        distance = row["endpoint_gap_metres"]

        folium.PolyLine(
            [
                [a[1], a[0]],
                [b[1], b[0]]
            ],
            color="red",
            weight=4,
            dash_array="8, 8",
            tooltip=f"Reference-route break: {distance:.2f} m"
        ).add_to(group)

        folium.Marker(
            [a[1], a[0]],
            tooltip=f"Previous arc endpoint: node {previous_node}",
            icon=folium.Icon(color="red", icon="arrow-down")
        ).add_to(group)

        folium.Marker(
            [b[1], b[0]],
            tooltip=f"Next arc start: node {next_node}",
            icon=folium.Icon(color="darkred", icon="arrow-up")
        ).add_to(group)


def main():
    gps_df = pd.read_csv(GPS_FILE)
    route_df = pd.read_csv(ROUTE_FILE)

    # Identify every trip with at least one detected anomaly.
    gps_keys = set(
        zip(gps_df["partition"], gps_df["trip_id"].astype(int))
    )

    route_keys = set(
        zip(route_df["partition"], route_df["trip_id"].astype(int))
    )

    anomalous_trips = sorted(
        gps_keys | route_keys,
        key=lambda x: (x[0], x[1])
    )

    if not anomalous_trips:
        print("No anomalous trips found.")
        return

    first_folder = trip_path(*anomalous_trips[0])
    first_track = load_data(first_folder, ".track")

    center = [
        float(np.median(first_track[:, 1])),
        float(np.median(first_track[:, 0]))
    ]

    map_object = folium.Map(
        location=center,
        zoom_start=14,
        tiles=None,
        control_scale=True
    )

    print(f"Found {len(anomalous_trips)} anomalous trips.")

    for trip_number, (partition, trip_id) in enumerate(anomalous_trips):
        print(
            f"Creating map layer {trip_number + 1}/"
            f"{len(anomalous_trips)}: {partition} trip {trip_id}"
        )

        folder = trip_path(partition, trip_id)

        track = load_data(folder, ".track")
        nodes = load_data(folder, ".nodes")
        arcs = load_data(folder, ".arcs").astype(int)
        route = load_data(folder, ".route")[:, 0].astype(int)

        gaps = gps_df[
            (gps_df["partition"] == partition)
            & (gps_df["trip_id"].astype(int) == trip_id)
        ]

        breaks = route_df[
            (route_df["partition"] == partition)
            & (route_df["trip_id"].astype(int) == trip_id)
        ]

        group = FeatureGroup(
            name=f"{partition} | Trip {trip_id:02d} "
                 f"| GPS gaps: {len(gaps)} "
                 f"| Route breaks: {len(breaks)}",
            show=(trip_number == 0)
        )

        add_local_network(group, nodes, arcs, track)
        add_gps(group, track, gaps)
        add_reference_route(group, nodes, arcs, route, breaks)

        group.add_to(map_object)

    LayerControl(collapsed=False).add_to(map_object)

    legend = """
    <div style="
        position: fixed;
        bottom: 25px;
        left: 25px;
        z-index: 9999;
        background: white;
        padding: 12px;
        border: 2px solid gray;
        border-radius: 6px;
        font-size: 13px;">
        <b>PathSnap - Data Anomaly Exploration</b><br>
        <span style="color:blue;">━━</span> GPS trajectory<br>
        <span style="color:gray;">━━</span> Nearby road network<br>
        <span style="color:green;">━━</span> Reference route<br>
        <span style="color:red;">┅┅</span> Route discontinuity<br>
        <span style="color:orange;">┅┅</span> GPS recording gap
    </div>
    """

    map_object.get_root().html.add_child(
        folium.Element(legend)
    )

    map_object.save(str(OUTPUT))

    print("\nMAP CREATED SUCCESSFULLY")
    print("Output:", OUTPUT)
    print("Total anomalous trips:", len(anomalous_trips))
    print("GPS gap records:", len(gps_df))
    print("Route discontinuities:", len(route_df))


if __name__ == "__main__":
    main()
