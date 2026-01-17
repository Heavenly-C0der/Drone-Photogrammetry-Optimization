# full_triangulation_ransac_distortion.py
## latest _ 02/11 - 03:45am
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import cv2
import random

# ---------- CONFIG ----------
MATCHES_DIR   = r"D:\SB Project\matches_filtered"
CSV_PATH      = r"D:\SB Project\image_metadata.csv"
REF_CENTER    = True
REPROJ_THRESH = 50.0   # px
RANSAC_ITERS  = 50     # increase for better outlier rejection
OUTPUT_DIR    = r"D:\SB Project\triangulation_output_distortion"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- Load metadata ----------
df = pd.read_csv(CSV_PATH)
df["filename_norm"] = df["filename"].apply(lambda x: os.path.splitext(str(x).upper())[0])
metadata_dict = df.set_index("filename_norm").T.to_dict()

# ---------- Compute reference center ----------
all_positions = np.array([
    [float(meta['easting']), float(meta['northing']), float(meta['altitude'])]
    for meta in metadata_dict.values()
])
REF_C = all_positions.mean(axis=0) if REF_CENTER else np.zeros(3)
print(f"Reference center: {REF_C}")

# ---------- Camera utilities ----------
def euler_to_rot_autel(yaw_deg, pitch_deg, roll_deg, yaw_sign=-1.0, apply_ned2enu=True, transpose_R=True):
    yaw, pitch, roll = np.deg2rad([yaw_sign*yaw_deg, pitch_deg, roll_deg])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw),  np.cos(yaw), 0],
                    [0,0,1]])
    R_y = np.array([[ np.cos(pitch),0,np.sin(pitch)],
                    [0,1,0],
                    [-np.sin(pitch),0,np.cos(pitch)]])
    R_x = np.array([[1,0,0],
                    [0,np.cos(roll),-np.sin(roll)],
                    [0,np.sin(roll), np.cos(roll)]])
    R_c2w_ned = R_z @ R_y @ R_x
    if apply_ned2enu:
        R_ned2enu = np.array([[0,1,0],[1,0,0],[0,0,-1]])
        R_c2w = R_ned2enu @ R_c2w_ned
    else:
        R_c2w = R_c2w_ned
    return R_c2w.T if transpose_R else R_c2w

def get_camera_pose(img):
    meta = metadata_dict[img]
    C_world = np.array([float(meta['easting']), float(meta['northing']), float(meta['altitude'])])
    C_centered = C_world - REF_C
    pitch = float(meta['pitch_deg'])
    roll  = float(meta['roll_deg'])
    yaw   = float(meta['yaw_deg'])
    R_c2w = euler_to_rot_autel(yaw, pitch, roll, yaw_sign=-1.0)
    R_w2c = R_c2w.T
    t = -R_w2c @ C_centered.reshape(3,1)
    fx, fy = float(meta['fx']), float(meta['fy'])
    cx, cy = float(meta['cx']), float(meta['cy'])
    K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    P = K @ np.hstack([R_w2c, t])
    return R_w2c, C_centered, K, P

def undistort_points(pts, K, dist_coeffs):
    pts = np.array(pts, dtype=np.float32).reshape(-1,1,2)
    undist = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
    return undist.reshape(-1,2)

# ---------- Default distortion coefficients ----------
dist_coeffs_default = np.array([-0.1, 0.02, 0.0, 0.0, 0.0])  # k1, k2, p1, p2, k3
dist_dict = {img: dist_coeffs_default for img in metadata_dict.keys()}

# ---------- Build initial tracks from matches ----------
track_obs = defaultdict(list)
point_to_track = dict()
next_track_id = 0

match_files = [f for f in os.listdir(MATCHES_DIR) if f.endswith("_matches.pkl")]
print(f"Found {len(match_files)} match files.")

for mf in tqdm(match_files, desc="Building tracks"):
    path = os.path.join(MATCHES_DIR, mf)
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    base = os.path.splitext(mf)[0].rsplit("_matches", 1)[0]
    parts = base.split("_")
    img1 = "_".join(parts[0:2])
    img2 = "_".join(parts[2:4])
    pts1, pts2 = data["pts1"], data["pts2"]

    for idx, (pt1, pt2) in enumerate(zip(pts1, pts2)):
        key1, key2 = (img1, idx), (img2, idx)
        t1 = point_to_track.get(key1)
        t2 = point_to_track.get(key2)

        if t1 is not None and t2 is not None:
            # Merge only if no overlapping images
            if t1 != t2:
                imgs_t1 = set(img for img,_ in track_obs[t1])
                imgs_t2 = set(img for img,_ in track_obs[t2])
                if len(imgs_t1 & imgs_t2) == 0:
                    track_obs[t1].extend(track_obs[t2])
                    for k in track_obs[t2]:
                        point_to_track[k] = t1
                    del track_obs[t2]
            track_id = t1
        elif t1 is not None:
            track_id = t1
            track_obs[track_id].append((img2, pt2))
            point_to_track[key2] = track_id
        elif t2 is not None:
            track_id = t2
            track_obs[track_id].append((img1, pt1))
            point_to_track[key1] = track_id
        else:
            track_id = next_track_id
            track_obs[track_id] = [(img1, pt1), (img2, pt2)]
            point_to_track[key1] = track_id
            point_to_track[key2] = track_id
            next_track_id += 1

print(f"Total raw tracks built: {len(track_obs)}")

# ---------- Triangulate a single track with RANSAC ----------
def triangulate_track_ransac(obs, iters=RANSAC_ITERS, reproj_thresh=REPROJ_THRESH):
    best_X = None
    best_err = float('inf')
    if len(obs) < 2:
        return None, float('inf')
    for _ in range(iters):
        samples = random.sample(obs, 2)
        A = []
        for img, (u,v) in samples:
            R_w2c, C_centered, K, P = get_camera_pose(img)
            # Undistort
            (u,v) = undistort_points([(u,v)], K, dist_dict[img])[0]
            A.append(u*P[2,:]-P[0,:])
            A.append(v*P[2,:]-P[1,:])
        A = np.vstack(A)
        try:
            _,_,Vt = np.linalg.svd(A)
            X = Vt[-1,:3]/Vt[-1,3]
        except np.linalg.LinAlgError:
            continue
        # Reprojection error for all observations
        errs = []
        for img, (u,v) in obs:
            R_w2c, C_centered, K, P = get_camera_pose(img)
            (u,v) = undistort_points([(u,v)], K, dist_dict[img])[0]
            uvh = P @ np.hstack([X,1])
            uv = uvh[:2]/uvh[2]
            errs.append(np.linalg.norm(uv-[u,v]))
        mean_err = np.mean(errs)
        if mean_err < best_err:
            best_err = mean_err
            best_X = X
    return best_X, best_err

# ---------- Run RANSAC triangulation ----------
triang_points = []
reproj_errors = []
worst_tracks = []

for tid, obs in tqdm(track_obs.items(), desc="Triangulating tracks with RANSAC + distortion"):
    X, err = triangulate_track_ransac(obs)
    if X is not None:
        triang_points.append((X, obs))
        reproj_errors.append(err)
        worst_tracks.append((err, tid, X, obs))

reproj_errors = np.array(reproj_errors)
print(f"Total triangulated points: {len(triang_points)}")
print(f"Mean reprojection error: {np.mean(reproj_errors):.2f}px")
print(f"Median reprojection error: {np.median(reproj_errors):.2f}px")
print(f"Max reprojection error: {np.max(reproj_errors):.2f}px")

# ---------- Filter points ----------
filtered_points = [(X, obs) for (X, obs), e in zip(triang_points, reproj_errors) if e <= REPROJ_THRESH]
print(f"Filtered points (<= {REPROJ_THRESH}px): {len(filtered_points)}")

# ---------- Save results ----------
with open(os.path.join(OUTPUT_DIR, "triangulated_points_ransac_distortion.pkl"), "wb") as f:
    pickle.dump(filtered_points, f)

camera_poses = {}
for img in metadata_dict.keys():
    R,C,K,P = get_camera_pose(img)
    camera_poses[img] = {"R": R, "C": C, "K": K, "P": P}

with open(os.path.join(OUTPUT_DIR, "camera_poses_ransac_distortion.pkl"), "wb") as f:
    pickle.dump(camera_poses, f)

print(f"Saved triangulated points and camera poses to {OUTPUT_DIR}")
