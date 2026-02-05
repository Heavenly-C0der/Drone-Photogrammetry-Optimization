import os
import cv2
import pickle
import gc
import numpy as np
import pandas as pd
from tqdm import tqdm

# ========== CONFIG ==========
FEATURES_DIR = r"D:\SB Project\features_filtered"
MATCHES_DIR = r"D:\SB Project\matches_filtered"
CSV_PATH = r"D:\SB Project\image_metadata.csv"
os.makedirs(MATCHES_DIR, exist_ok=True)

USE_SIFT = True
SLIDING_WINDOW = 20      # max consecutive images to match forward
MIN_MATCHES = 10
MIN_INLIERS = 6
LOWE_RATIO = 0.68
DIST_THRESH = 165.0       # max descriptor distance for SIFT matches
MIN_BASELINE = 5        # meters, skip pairs too close
MAX_FOV_ANGLE = 120.0     # degrees, skip pairs looking too parallel

# ========== LOAD METADATA ==========
df = pd.read_csv(CSV_PATH)
filenames = df['filename'].tolist()
positions = df[['easting', 'northing', 'altitude']].values
n_images = len(filenames)

# ========== SETUP MATCHER ==========
if USE_SIFT:
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
else:
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

# ========== HELPER: convert serialized keypoints to cv2.KeyPoint ==========
def to_cv_keypoints(kp_serialized):
    kp_cv = []
    for pt in kp_serialized:
        if isinstance(pt[0], (tuple, list)):
            x, y = float(pt[0][0]), float(pt[0][1])
            size = float(pt[1])
            angle = float(pt[2])
            response = float(pt[3])
            octave = int(pt[4])
            class_id = int(pt[5])
        elif len(pt) >= 7:
            x, y = float(pt[0]), float(pt[1])
            size = float(pt[2])
            angle = float(pt[3])
            response = float(pt[4])
            octave = int(pt[5])
            class_id = int(pt[6])
        else:
            continue
        kp_cv.append(cv2.KeyPoint(x, y, size, angle, response, octave, class_id))
    return kp_cv

# ========== HELPER: compute baseline distance ==========
def baseline_dist(idx1, idx2):
    return np.linalg.norm(positions[idx1] - positions[idx2])

# ========== HELPER: compute FOV angle ==========
def fov_angle(idx1, idx2):
    # simple vector from camera 1 to camera 2
    vec = positions[idx2] - positions[idx1]
    vec /= np.linalg.norm(vec)
    # assume forward vector along +Z in camera frame
    fwd = np.array([0,0,1], dtype=float)
    cosang = np.clip(np.dot(vec, fwd), -1.0, 1.0)
    return np.rad2deg(np.arccos(cosang))

# ========== SLIDING WINDOW MATCHING ==========
for i in tqdm(range(n_images), desc="Matching images"):
    f1 = filenames[i]
    base1 = os.path.splitext(f1)[0]
    feat1_file = os.path.join(FEATURES_DIR, f"{base1}_features.pkl")
    if not os.path.exists(feat1_file):
        continue
    with open(feat1_file, "rb") as fh:
        data1 = pickle.load(fh)
    kp1_ser, des1 = data1["keypoints"], data1["descriptors"]
    if des1 is None or len(des1) == 0:
        continue
    des1 = np.asarray(des1).astype(np.float32)
    kp1_cv = to_cv_keypoints(kp1_ser)

    for j in range(i+1, min(i+1+SLIDING_WINDOW, n_images)):
        # geometric pre-filter: baseline + FOV
        if baseline_dist(i,j) < MIN_BASELINE:
            continue
        if fov_angle(i,j) > MAX_FOV_ANGLE:
            continue

        f2 = filenames[j]
        base2 = os.path.splitext(f2)[0]
        feat2_file = os.path.join(FEATURES_DIR, f"{base2}_features.pkl")
        if not os.path.exists(feat2_file):
            continue
        with open(feat2_file, "rb") as fh:
            data2 = pickle.load(fh)
        kp2_ser, des2 = data2["keypoints"], data2["descriptors"]
        if des2 is None or len(des2) == 0:
            continue
        des2 = np.asarray(des2).astype(np.float32)
        kp2_cv = to_cv_keypoints(kp2_ser)

        # ----- KNN + Lowe ratio -----
        good_matches = []
        if USE_SIFT:
            try:
                matches_knn = matcher.knnMatch(des1, des2, k=2)
            except:
                continue
            for m_n in matches_knn:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < LOWE_RATIO * n.distance and m.distance < DIST_THRESH:
                    good_matches.append((m.queryIdx, m.trainIdx, m.distance))
        else:
            matches = matcher.match(des1, des2)
            for m in matches:
                if m.distance < DIST_THRESH:
                    good_matches.append((m.queryIdx, m.trainIdx, m.distance))

        # ----- Mutual check -----
        matches12 = {q: t for q, t, d in good_matches}
        matches21 = {}
        if USE_SIFT:
            try:
                matches_knn_rev = matcher.knnMatch(des2, des1, k=2)
            except:
                continue
            for m_n in matches_knn_rev:
                if len(m_n) < 2:
                    continue
                m, n = m_n
                if m.distance < LOWE_RATIO * n.distance and m.distance < DIST_THRESH:
                    matches21[m.queryIdx] = m.trainIdx
        else:
            matches_rev = matcher.match(des2, des1)
            for m in matches_rev:
                if m.distance < DIST_THRESH:
                    matches21[m.queryIdx] = m.trainIdx

        mutual_matches = []
        for q, t, d in good_matches:
            if matches21.get(t, -1) == q:
                mutual_matches.append((q, t, d))

        if len(mutual_matches) < MIN_MATCHES:
            continue

        # ----- RANSAC for inliers -----
        pts1 = np.float32([kp1_cv[q].pt for q, t, _ in mutual_matches])
        pts2 = np.float32([kp2_cv[t].pt for q, t, _ in mutual_matches])
        inliers = []
        if len(pts1) >= 8:
            F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)
            mask_bool = mask.ravel().astype(bool) if mask is not None else np.ones(len(pts1), dtype=bool)
            inliers = [mutual_matches[k] for k, m in enumerate(mask_bool) if m]
        else:
            inliers = mutual_matches

        if len(inliers) < MIN_INLIERS:
            continue

        # ----- Save matches -----
        match_file = os.path.join(MATCHES_DIR, f"{base1}_{base2}_matches.pkl")
        matched_pts1 = [tuple(kp1_cv[q].pt) for q, t, _ in inliers]
        matched_pts2 = [tuple(kp2_cv[t].pt) for q, t, _ in inliers]
        with open(match_file, "wb") as fh:
            pickle.dump({
                "image1": f1,
                "image2": f2,
                "matches": inliers,
                "pts1": matched_pts1,
                "pts2": matched_pts2
            }, fh)

        print(f"✅ {f1} ↔ {f2}: baseline={baseline_dist(i,j):.2f}m, FOV={fov_angle(i,j):.2f}°, {len(inliers)} inliers saved")

        # free memory
        del data2, kp2_cv, des2, matches_knn, mutual_matches, inliers
        gc.collect()

    # free memory
    del data1, kp1_cv, des1
    gc.collect()

print("\n🎯 All filtered matches saved in:", MATCHES_DIR)

