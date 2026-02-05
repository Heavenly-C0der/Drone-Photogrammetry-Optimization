import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------- CONFIG ----------
TRIANGULATION_DIR = r"D:\SB Project\triangulation_output_distortion_new"
BA_DIR = r"D:\SB Project\Ba_BFGS_jax_output"

# ---------- Load triangulated points ----------
with open(f"{TRIANGULATION_DIR}/triangulated_points_filtered_final.pkl", "rb") as f:
    triangulated_points = pickle.load(f)  # list of (X, obs)

triangulated_xyz = np.array([X for X, _ in triangulated_points])

# ---------- Load BA refined points ----------
with open(f"{BA_DIR}/ba_refined_points_jax.pkl", "rb") as f:
    ba_refined_points = pickle.load(f)  # numpy array of shape (N, 3)

ba_refined_xyz = np.asarray(ba_refined_points)
assert ba_refined_xyz.shape[1] == 3, "Each BA point must have 3 coordinates"

# ---------- Load camera poses (pick any, since they should match) ----------
with open(f"{BA_DIR}/ba_refined_camera_poses_jax.pkl", "rb") as f:
    camera_poses = pickle.load(f)

cam_centers = np.array([camera_poses[k]["C"] for k in camera_poses])

# ---------- Function for equal aspect ratio ----------
def set_axes_equal(ax):
    """Set 3D plot axes to equal scale."""
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
    ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
    ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

# ---------- Plot triangulated + BA-refined + cameras ----------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Triangulated points (before BA)
ax.scatter(triangulated_xyz[:, 0], triangulated_xyz[:, 1], triangulated_xyz[:, 2],
           c='blue', s=1, alpha=0.4, label='Triangulated Points (Before BA)')

# BA-refined points
ax.scatter(ba_refined_xyz[:, 0], ba_refined_xyz[:, 1], ba_refined_xyz[:, 2],
           c='red', s=1, alpha=0.4, label='Refined Points (After BA)')

# Cameras
ax.scatter(cam_centers[:, 0], cam_centers[:, 1], cam_centers[:, 2],
           c='green', s=40, marker='^', label='Cameras')

# Labels and legend
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Z (m)")
ax.set_title("3D Visualization: Overlay of Triangulated Points (Pre-Optimization) and BA-refined (Post-Optimization) 3D Points with Cameras")
ax.legend(loc='upper right')

# Equalize scales
set_axes_equal(ax)

plt.show()

# Save zoomed plot
plt.savefig(f"D:\SB Project\Final_BA_before_after.png", dpi=300)
plt.show()