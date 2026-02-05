import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- CONFIG ----------
OUTPUT_DIR = r"D:\SB Project\triangulation_output_distortion_new"
CSV_PATH   = r"D:\SB Project\image_metadata.csv"


# ---------- Load filtered points ----------
with open(f"{OUTPUT_DIR}/triangulated_points_ransac_distortion.pkl", "rb") as f:
    filtered_points = pickle.load(f)  # list of (X, obs)

# ---------- Load camera poses ----------
with open(f"{OUTPUT_DIR}/camera_poses_ransac_distortion.pkl", "rb") as f:
    camera_poses = pickle.load(f)

# ---------- Compute RMS reprojection error ----------
all_errors = []

def undistort_points(pts, K, dist_coeffs):
    import cv2
    pts = np.array(pts, dtype=np.float32).reshape(-1,1,2)
    undist = cv2.undistortPoints(pts, K, dist_coeffs, P=K)
    return undist.reshape(-1,2)

dist_coeffs_default = np.array([-0.1, 0.02, 0.0, 0.0, 0.0])

for X, obs in filtered_points:
    for img, (u,v) in obs:
        R = camera_poses[img]["R"]
        C = camera_poses[img]["C"]
        K = camera_poses[img]["K"]
        P = camera_poses[img]["P"]
        (u_dist, v_dist) = undistort_points([(u,v)], K, dist_coeffs_default)[0]
        uvh = P @ np.hstack([X,1])
        uv_proj = uvh[:2]/uvh[2]
        all_errors.append(np.linalg.norm(uv_proj - [u_dist, v_dist]))

all_errors = np.array(all_errors)
rms_error = np.sqrt(np.mean(all_errors**2))
print(f"Filtered points: {len(filtered_points)}")
print(f"Global RMS reprojection error (filtered): {rms_error:.2f}px")

# ---------- Plot 3D points and cameras ----------
# Prepare points and cameras
points_xyz = np.array([X for X,_ in filtered_points])
cam_centers = np.array([camera_poses[img]["C"] for img in camera_poses])

# Static 3D plot
fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(points_xyz[:,0], points_xyz[:,1], points_xyz[:,2], c='blue', s=1, alpha=0.5, label='3D points')
ax.scatter(cam_centers[:,0], cam_centers[:,1], cam_centers[:,2], c='red', s=30, marker='^', label='Cameras')

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
ax.set_title("Filtered 3D points + Camera Positions")
ax.legend()

# Save static plot
plt.savefig(f"{OUTPUT_DIR}/3D_points_and_cameras.png", dpi=300)
plt.show()

"""
####### Interactive 3D points ######
import plotly.graph_objects as go
fig = go.Figure()

# 3D points
fig.add_trace(go.Scatter3d(
    x=points_xyz[:,0], y=points_xyz[:,1], z=points_xyz[:,2],
    mode='markers',
    marker=dict(size=2, color='blue', opacity=0.6),
    name='3D points'
))

# Camera positions
fig.add_trace(go.Scatter3d(
    x=cam_centers[:,0], y=cam_centers[:,1], z=cam_centers[:,2],
    mode='markers',
    marker=dict(size=5, color='red', symbol='diamond'),
    name='Cameras'
))

fig.update_layout(
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        aspectmode='data'  # preserves scale
    ),
    title="Interactive 3D Point Cloud with Cameras",
    width=2000, height=800
)

# Save interactive plot as HTML
fig.write_html(f"{OUTPUT_DIR}/3D_points_and_cameras.html")

# Show in browser
fig.show()

"""
