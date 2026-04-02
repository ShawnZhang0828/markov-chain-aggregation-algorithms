import numpy as np
from pathlib import Path
from sklearn.cluster import MiniBatchKMeans
from math import radians, sin, cos, sqrt, atan2


def haversine_xy_meters(lat1, lon1, lat2, lon2):
    # Convert two GPS points into approximate local planar displacement in meters
    R = 6371000.0

    lat1_r = radians(lat1)
    lon1_r = radians(lon1)
    lat2_r = radians(lat2)
    lon2_r = radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    mean_lat = 0.5 * (lat1_r + lat2_r)
    dx = R * dlon * cos(mean_lat)
    dy = R * dlat
    return dx, dy


def read_geolife_plt(path):
    # Read one GeoLife .plt trajectory file
    pts = []

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[6:]:
        parts = line.strip().split(",")
        if len(parts) < 7:
            continue

        lat = float(parts[0])
        lon = float(parts[1])
        date_str = parts[5]
        time_str = parts[6]

        # Build a simple timestamp string first
        ts = np.datetime64(f"{date_str}T{time_str}")

        pts.append((lat, lon, ts))

    return pts


def trajectory_to_step_features(points, max_speed_mps=60.0):
    # Convert one trajectory into step-level motion features
    feats = []

    if len(points) < 2:
        return np.empty((0, 3), dtype=float)

    for i in range(len(points) - 1):
        lat1, lon1, t1 = points[i]
        lat2, lon2, t2 = points[i + 1]

        dt = (t2 - t1) / np.timedelta64(1, "s")
        if dt <= 0:
            continue

        dx, dy = haversine_xy_meters(lat1, lon1, lat2, lon2)
        speed = np.sqrt(dx * dx + dy * dy) / dt

        # Drop obvious outliers
        if speed > max_speed_mps:
            continue

        feats.append([dx, dy, speed])

    if len(feats) == 0:
        return np.empty((0, 3), dtype=float)

    return np.asarray(feats, dtype=float)


def load_geolife_markov_chain(
    root_dir,
    n_states=30,
    max_users=None,
    max_files_per_user=None,
    smoothing=1e-8,
    random_state=0,
):
    # Collect all trajectory files
    root = Path(root_dir)
    user_dirs = sorted([p for p in root.iterdir() if p.is_dir()])

    if max_users is not None:
        user_dirs = user_dirs[:max_users]

    all_sequences = []
    all_features = []

    for user_dir in user_dirs:
        traj_dir = user_dir / "Trajectory"
        if not traj_dir.exists():
            continue

        files = sorted(traj_dir.glob("*.plt"))
        if max_files_per_user is not None:
            files = files[:max_files_per_user]

        for fp in files:
            points = read_geolife_plt(fp)
            feats = trajectory_to_step_features(points)

            if len(feats) >= 2:
                all_sequences.append(feats)
                all_features.append(feats)

    if len(all_features) == 0:
        raise ValueError("No valid trajectory features were extracted.")

    all_features = np.vstack(all_features)

    # Normalize features before clustering
    mean = all_features.mean(axis=0, keepdims=True)
    std = all_features.std(axis=0, keepdims=True)
    std[std < 1e-12] = 1.0
    all_features_norm = (all_features - mean) / std

    # Fit clustering model
    kmeans = MiniBatchKMeans(
        n_clusters=n_states,
        random_state=random_state,
        batch_size=10000,
        n_init=10,
    )
    kmeans.fit(all_features_norm)

    # Count transitions between consecutive motion states
    C = np.zeros((n_states, n_states), dtype=float)

    for seq in all_sequences:
        seq_norm = (seq - mean) / std
        labels = kmeans.predict(seq_norm)

        for a, b in zip(labels[:-1], labels[1:]):
            C[a, b] += 1.0

    # Fix zero rows
    for i in range(n_states):
        if C[i].sum() == 0:
            C[i, i] = 1.0

    C += smoothing
    P = C / C.sum(axis=1, keepdims=True)

    return P
