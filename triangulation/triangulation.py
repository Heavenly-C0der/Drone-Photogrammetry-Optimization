import os
import cv2
import pickle
import numpy as np
import pandas as pd
from glob import glob
import re
import gc
from tqdm import tqdm
from scipy.spatial import KDTree

# ============================
# CONFIG
# ============================
FEATURES_DIR = r"D:\SB Project\features_filtered"
MATCHES_DIR = r"D:\SB Project\matches_filtered"
CSV_PATH = r"D:\SB Project\image_metadata.csv"
OUTPUT_DIR = r"D:\SB Project\triangulated_results_for_BA"

MIN_MATCHES = 4
MAX_PTS_PER_PAIR = 6000
FOV_ANGLE_THRESH = 60     # max allowed yaw difference for FOV overlap (deg)
GPS_WEIGHT = 1.0          # weight for GPS translation prior

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================
# Load metadata
# ============================
metadata = pd.read_csv(CSV_PATH)
metadata["filename_norm"] = metadata["filename"].apply(lambda x: os.path.splitext(x.lower())[0])
metadata["yaw"] = metadata["yaw_deg"].fillna(0)
metadata["pitch"] = metadata["pitch_deg"].fillna(0)
metadata["roll"] = metadata["roll_deg"].fillna(0)
metadata_dict = metadata.set_index("filename_norm").T.to_dict()

coords = metadata[['easting','northing','altitude']].values
filenames = metadata['filename_norm'].tolist()
tree = KDTree(coords)

print(f"🔹 Loaded metadata for {len(metadata)} images")
print(f"🔹 KDTree built for GPS neighbor queries\n")

# ============================
# Camera intrinsics
# ============================
def get_intrinsics(img_key):
    intr = metadata_dict.get(img_key)
    if intr is None:
        raise ValueError(f"Missing intrinsics for {img_key}")
    fx, fy, cx, cy = float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0, 1]], dtype=np.float64)
    return K

# ============================
# Yaw/Pitch/Roll → Rotation
# ============================
def euler_to_rot(yaw, pitch, roll, deg=True):
    if deg:
        yaw = np.deg2rad(yaw)
        pitch = np.deg2rad(pitch)
        roll = np.deg2rad(roll)
    Rz = np.array([[ np.cos(yaw), -np.sin(yaw), 0],
                   [ np.sin(yaw),  np.cos(yaw), 0],
                   [0,0,1]])
    Ry = np.array([[ np.cos(pitch), 0, np.sin(pitch)],
                   [0,1,0],
                   [-np.sin(pitch),0, np.cos(pitch)]])
    Rx = np.array([[1,0,0],
                   [0,np.cos(roll), -np.sin(roll)],
                   [0,np.sin(roll),  np.cos(roll)]])
    return Rz @ Ry @ Rx

def get_camera_rotation(img_key):
    intr = metadata_dict.get(img_key)
    yaw = float(intr.get("yaw", 0))
    pitch = float(intr.get("pitch", 0))
    roll = float(intr.get("roll", 0))
    return euler_to_rot(yaw, pitch, roll)

def fov_overlap_ok(img1, img2, max_angle_diff=FOV_ANGLE_THRESH):
    R1 = get_camera_rotation(img1)
    R2 = get_camera_rotation(img2)
    f1 = R1 @ np.array([0,0,1])
    f2 = R2 @ np.array([0,0,1])
    angle = np.rad2deg(np.arccos(np.clip(f1 @ f2, -1,1)))
    return angle < max_angle_diff

# ============================
# Helpers
# ============================
def load_features_pkl(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data.get("keypoints"), data.get("descriptors")

def reconstruct_points_array(kp_serialized):
    if kp_serialized is None:
        return None
    return np.array([(float(pt[0][0]), float(pt[0][1])) for pt in kp_serialized], dtype=np.float32)

def triangulate_points(pts1, pts2, K1, K2, R, t):
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3,1))))
    P2 = K2 @ np.hstack((R, t.reshape(3,1)))
    pts4 = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4[:3] / pts4[3]).T
    Z1 = pts3d[:,2]
    Z2 = (R @ pts3d.T + t.reshape(3,1)).T[:,2]
    mask = (Z1 > 0) & (Z2 > 0)
    return pts3d[mask], mask

# ============================
# Build candidate pairs
# ============================
pairs = set()
for i, name in enumerate(filenames):
    dists, idxs = tree.query(coords[i], k=6)  # 5 nearest neighbors + itself
    for j in idxs[1:]:
        pair = tuple(sorted([filenames[i], filenames[j]]))
        pairs.add(pair)
pairs = sorted(list(pairs))
print(f"🔹 Candidate image pairs after GPS neighbors: {len(pairs)}\n")

# ============================
# Main loop
# ============================
camera_poses = {}
triangulated_points = []
observations = []
point_id_counter = 0

for img1_key, img2_key in tqdm(pairs, desc="Triangulating pairs"):

    # --- FOV / yaw/pitch filter ---
    if not fov_overlap_ok(img1_key, img2_key):
        continue

    # --- Intrinsics ---
    try:
        K1 = get_intrinsics(img1_key)
        K2 = get_intrinsics(img2_key)
    except ValueError:
        continue

    # --- Features ---
    feat1_file = os.path.join(FEATURES_DIR, f"{img1_key}_features.pkl")
    feat2_file = os.path.join(FEATURES_DIR, f"{img2_key}_features.pkl")
    if not (os.path.exists(feat1_file) and os.path.exists(feat2_file)):
        continue

    kp1_ser, _ = load_features_pkl(feat1_file)
    kp2_ser, _ = load_features_pkl(feat2_file)
    if kp1_ser is None or kp2_ser is None:
        continue

    pts_all1 = reconstruct_points_array(kp1_ser)
    pts_all2 = reconstruct_points_array(kp2_ser)

    match_file = os.path.join(MATCHES_DIR, f"{img1_key}_{img2_key}_matches.pkl")
    if not os.path.exists(match_file):
        continue

    with open(match_file, "rb") as f:
        match_data = pickle.load(f)

    matches_idx = match_data.get("matches", [])
    if len(matches_idx) < MIN_MATCHES:
        continue

    query_idx = np.array([m[0] for m in matches_idx])
    train_idx = np.array([m[1] for m in matches_idx])
    if len(query_idx) > MAX_PTS_PER_PAIR:
        query_idx = query_idx[:MAX_PTS_PER_PAIR]
        train_idx = train_idx[:MAX_PTS_PER_PAIR]

    pts1 = pts_all1[query_idx]
    pts2 = pts_all2[train_idx]

    # --- Initialize rotation from yaw/pitch/roll and translation from GPS ---
    R1 = get_camera_rotation(img1_key)
    R2 = get_camera_rotation(img2_key)
    R_rel = R2 @ R1.T
            
    t_rel = np.array([
        metadata_dict[img2_key]['easting'],
        metadata_dict[img2_key]['northing'],
        metadata_dict[img2_key]['altitude']
    ], dtype=np.float64) - np.array([
        metadata_dict[img1_key]['easting'],
        metadata_dict[img1_key]['northing'],
        metadata_dict[img1_key]['altitude']
    ], dtype=np.float64)


    pts3d, mask3d = triangulate_points(pts1, pts2, K1, K2, R_rel, t_rel)
    if pts3d.shape[0] == 0:
        continue

    # --- Store camera pose ---
    camera_poses[(img1_key, img2_key)] = {"R": R_rel, "t": t_rel}

    # --- Store 3D points and observations ---
    for i, p3 in enumerate(pts3d):
        pid = point_id_counter
        point_id_counter += 1
        triangulated_points.append({"point_id": pid, "pair": (img1_key, img2_key), "X": p3.tolist()})
        observations.append((img1_key, pid, pts1[i][0], pts1[i][1]))
        observations.append((img2_key, pid, pts2[i][0], pts2[i][1]))

    del kp1_ser, kp2_ser, pts_all1, pts_all2, pts1, pts2
    gc.collect()

    print(f"✅ Triangulated {len(pts3d)} points for pair {img1_key} ↔ {img2_key}")

# ============================
# Save results
# ============================
with open(os.path.join(OUTPUT_DIR, "camera_poses.pkl"), "wb") as f:
    pickle.dump(camera_poses, f)
with open(os.path.join(OUTPUT_DIR, "triangulated_points.pkl"), "wb") as f:
    pickle.dump(triangulated_points, f)
with open(os.path.join(OUTPUT_DIR, "observations.pkl"), "wb") as f:
    pickle.dump(observations, f)

print("\n🎯 Triangulation complete for Bundle Adjustment")
print(f"Cameras: {len(camera_poses)}, Points: {len(triangulated_points)}, Observations: {len(observations)}")
