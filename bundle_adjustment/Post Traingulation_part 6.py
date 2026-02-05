# reprojection_error_filtering.py
# Compute per-point reprojection errors, remove outliers (max_reproj_err > 55px),
# and compute global per-observation reprojection stats.

import os
import pickle
import numpy as np
import cv2
from tqdm import tqdm

# ---------- CONFIG ----------
TRIANG_FILE = r"D:\SB Project\triangulation_output_distortion_new\triangulated_points_ransac_distortion.pkl"
POSE_FILE   = r"D:\SB Project\triangulation_output_distortion_new\camera_poses_ransac_distortion.pkl"
OUTPUT_FILTERED = r"D:\SB Project\triangulation_output_distortion_new\triangulated_points_filtered_final.pkl"
OUTPUT_CSV  = r"D:\SB Project\triangulation_output_distortion_new\reprojection_errors_per_point.csv"

REPROJ_THRESHOLD_REMOVE = 45.0  # pixels
dist_coeffs_default = np.array([-0.1, 0.02, 0.0, 0.0, 0.0])

# ---------- Load data ----------
with open(TRIANG_FILE, "rb") as f:
    triang_points = pickle.load(f)
with open(POSE_FILE, "rb") as f:
    camera_poses = pickle.load(f)

print(f"Loaded {len(triang_points)} triangulated points and {len(camera_poses)} camera poses.")

# ---------- Helper ----------
def undistort_points(pts, K, dist_coeffs):
    pts = np.array(pts, dtype=np.float32).reshape(-1,1,2)
    undist = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
    return undist.reshape(-1,2)

# ---------- Step 1: Compute per-point reprojection errors ----------
filtered_points = []
num_removed = 0

for (X, obs) in tqdm(triang_points, desc="Filtering points by reprojection error"):
    X_h = np.hstack([X, 1])
    errs = []

    for img, (u, v) in obs:
        cam = camera_poses[img]
        K, P = cam["K"], cam["P"]

        (u_undist, v_undist) = undistort_points([(u, v)], K, dist_coeffs_default)[0]
        uvh = P @ X_h
        if uvh[2] <= 1e-6:
            continue
        uv_proj = uvh[:2] / uvh[2]
        err = np.linalg.norm(uv_proj - np.array([u_undist, v_undist]))
        errs.append(err)

    if not errs:
        continue

    max_err = np.max(errs)
    if max_err <= REPROJ_THRESHOLD_REMOVE:
        filtered_points.append((X, obs))
    else:
        num_removed += 1

print(f"\nRemoved {num_removed} points with max_reproj_err > {REPROJ_THRESHOLD_REMOVE}px")
print(f"Remaining points: {len(filtered_points)}")

# ---------- Save filtered points ----------
with open(OUTPUT_FILTERED, "wb") as f:
    pickle.dump(filtered_points, f)

print(f"✅ Saved filtered triangulated points to:\n{OUTPUT_FILTERED}")

# ---------- Step 2: Compute per-observation reprojection errors ----------
all_errors = []

for X, obs in tqdm(filtered_points, desc="Computing per-observation reprojection errors"):
    X_h = np.hstack([X, 1])
    for img, (u, v) in obs:
        cam = camera_poses[img]
        K, P = cam["K"], cam["P"]

        (u_undist, v_undist) = undistort_points([(u, v)], K, dist_coeffs_default)[0]
        uvh = P @ X_h
        if uvh[2] <= 1e-6:
            continue
        uv_proj = uvh[:2] / uvh[2]
        err = np.linalg.norm(uv_proj - np.array([u_undist, v_undist]))
        all_errors.append(err)

all_errors = np.array(all_errors)

# ---------- Step 3: Global reprojection statistics ----------
global_mean = np.mean(all_errors)
global_rms  = np.sqrt(np.mean(np.square(all_errors)))
global_max  = np.max(all_errors)

print("\n------ GLOBAL REPROJECTION ERROR STATS ------")
print(f"Total observations considered: {len(all_errors)}")
print(f"Mean reprojection error: {global_mean:.2f}px")
print(f"RMS reprojection error:  {global_rms:.2f}px")
print(f"Max reprojection error:  {global_max:.2f}px")
print("---------------------------------------------")
