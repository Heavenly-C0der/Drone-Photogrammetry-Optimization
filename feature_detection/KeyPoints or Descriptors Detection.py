# KeyPoints and Descriptors detections, 
# A spatial grid-based filter is a method used in keypoint detection to ensure that the selected keypoints are distributed evenly across 
# an image or 3D point cloud. This technique is used to prevent keypoints from clustering in a single high-texture
# region, which is a common issue when simply selecting the strongest keypoint responses
import cv2
import numpy as np
import os
import pandas as pd
import pickle
import gc
import matplotlib.pyplot as plt

# ============================
# CONFIG
# ============================
IMAGE_DIR = r"D:\SB Project\odm_data_helenenschacht-main\odm_data_helenenschacht-main\images"
CSV_PATH = r"D:\SB Project\image_metadata.csv"
FEATURES_DIR = r"D:\SB Project\features_filtered"
os.makedirs(FEATURES_DIR, exist_ok=True)

# Load metadata
df = pd.read_csv(CSV_PATH)
image_coords = {row['filename']: (row['easting'], row['northing'], row['altitude']) for _, row in df.iterrows()}

# ============================
# SIFT Configuration
# ============================
MAX_FEATURES = 50000   # per image
GRID_SIZE = 18         # pixels per cell for spatial deduplication (3 keypoints per grid cell)
MAX_PER_CELL = 5

sift = cv2.SIFT_create(nfeatures=MAX_FEATURES)

print(f"🔧 Using SIFT with max {MAX_FEATURES} features per image")
print(f"📏 Spatial grid suppression cell size: {GRID_SIZE} pixels, max {MAX_PER_CELL} per cell\n")

# ============================
# Helper: Spatial Suppression
# ============================
def spatially_filter_keypoints(kp, des, grid_size=GRID_SIZE, max_per_cell=MAX_PER_CELL):
    if not kp:
        return [], None, []

    pts = np.array([k.pt for k in kp], dtype=np.float32)
    x_bins = (pts[:, 0] // grid_size).astype(int)
    y_bins = (pts[:, 1] // grid_size).astype(int)
    grid_dict = {}

    for i, (xb, yb) in enumerate(zip(x_bins, y_bins)):
        key = (xb, yb)
        if key not in grid_dict:
            grid_dict[key] = [(kp[i], des[i])]
        else:
            grid_dict[key].append((kp[i], des[i]))
            grid_dict[key] = sorted(grid_dict[key], key=lambda x: -x[0].response)[:max_per_cell]

    filtered_kp = []
    filtered_des = []
    grid_cells = []

    for (xb, yb), lst in grid_dict.items():
        for k, d in lst:
            filtered_kp.append(k)
            filtered_des.append(d)
            grid_cells.append((xb, yb))

    filtered_des = np.array(filtered_des)
    return filtered_kp, filtered_des, grid_cells

# ============================
# Feature Extraction Loop
# ============================

for fname in sorted(os.listdir(IMAGE_DIR)):
    if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    img_path = os.path.join(IMAGE_DIR, fname)
    print(f"🔍 Processing {fname} ...")

    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ Skipping {fname} (failed to load)")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = sift.detectAndCompute(gray, None)
    if des is None or len(kp) == 0:
        print(f"⚠️ No descriptors for {fname}")
        continue

    print(f"  ➤ Detected {len(kp)} raw keypoints")

    # Spatial suppression
    kp, des, grid_cells = spatially_filter_keypoints(kp, des, grid_size=GRID_SIZE, max_per_cell=MAX_PER_CELL)
    print(f"  ➤ After spatial filtering: {len(kp)} keypoints")

    # Limit to top MAX_FEATURES strongest
    if len(kp) > MAX_FEATURES:
        idx = np.argsort([-k.response for k in kp])[:MAX_FEATURES]
        kp = [kp[i] for i in idx]
        des = des[idx]
        print(f"  ➤ Truncated to {MAX_FEATURES} strongest features")

    # Serialize
    kp_serialized = [
        (float(k.pt[0]), float(k.pt[1]), k.size, k.angle, k.response, k.octave, k.class_id, gc)
        for k, gc in zip(kp, grid_cells)
    ]
    out_path = os.path.join(FEATURES_DIR, f"{os.path.splitext(fname)[0]}_features.pkl")

    with open(out_path, "wb") as f:
        pickle.dump({"keypoints": kp_serialized, "descriptors": des.astype(np.float32)}, f)

    print(f"✅ Saved {len(kp)} filtered features for {fname}\n")

    # Optional visualize
    #img_vis = cv2.drawKeypoints(gray, kp, None, color=(0,255,0))
    #plt.imshow(img_vis[..., ::-1]); plt.show()

    del img, gray, kp, des
    gc.collect()


print("\n🎯 Feature extraction + filtering complete.")
print(f"All features saved in: {FEATURES_DIR}")
