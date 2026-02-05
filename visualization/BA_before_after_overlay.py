import pickle
import numpy as np
import plotly.graph_objs as go

# ---------- CONFIG ----------
TRIANGULATION_DIR = r"D:\SB_Project\triangulation_output_distortion_new"
BA_DIR = r"D:\SB_Project\Ba_BFGS_jax_output"
EXPORT_PATH = r"D:\SB_Project\3D_comparison_BA_before_after.html"  # Path to save HTML

# ---------- Load triangulated points ----------
with open(f"{TRIANGULATION_DIR}/triangulated_points_filtered_final.pkl", "rb") as f:
    triangulated_points = pickle.load(f)  # list of (X, obs)

triangulated_xyz = np.array([X for X, _ in triangulated_points])

# ---------- Load BA-refined points ----------
with open(f"{BA_DIR}/ba_refined_points_jax.pkl", "rb") as f:
    ba_refined_points = pickle.load(f)  # numpy array of shape (N, 3)

ba_refined_xyz = np.asarray(ba_refined_points)
assert ba_refined_xyz.shape[1] == 3, "Each BA point must have 3 coordinates"

# ---------- Load BA camera poses ----------
with open(f"{BA_DIR}/ba_refined_camera_poses_jax.pkl", "rb") as f:
    camera_poses = pickle.load(f)

cam_centers = np.array([camera_poses[k]["C"] for k in camera_poses])

# ---------- Build Plotly 3D scatter plot ----------
fig = go.Figure()

# Triangulated points (before BA)
fig.add_trace(go.Scatter3d(
    x=triangulated_xyz[:, 0], y=triangulated_xyz[:, 1], z=triangulated_xyz[:, 2],
    mode='markers',
    marker=dict(size=2, color='blue', opacity=0.5),
    name='Triangulated Points (Before BA)'
))

# Refined points (after BA)
fig.add_trace(go.Scatter3d(
    x=ba_refined_xyz[:, 0], y=ba_refined_xyz[:, 1], z=ba_refined_xyz[:, 2],
    mode='markers',
    marker=dict(size=2, color='red', opacity=0.5),
    name='Refined Points (After BA)'
))

# Cameras
fig.add_trace(go.Scatter3d(
    x=cam_centers[:, 0], y=cam_centers[:, 1], z=cam_centers[:, 2],
    mode='markers',
    marker=dict(size=6, symbol='diamond', color='green'),
    name='Cameras'
))

# ---------- Configure layout ----------
fig.update_layout(
    width=1100, height=750,
    title="Interactive 3D Visualization: Triangulated Points (Pre-Optimization) vs BA-refined Points (Post-Optimization) with Cameras",
    scene=dict(
        xaxis_title="X (m)",
        yaxis_title="Y (m)",
        zaxis_title="Z (m)",
        aspectmode="data"   # Equal axis scaling
    )
)

# ---------- Show in browser ----------
fig.show()

# ---------- Auto-export to HTML ----------
# ---------- Auto-export to HTML (works on GitHub Pages) ----------
fig.write_html(EXPORT_PATH, include_plotlyjs='inline', full_html=True)
print(f"✅ Interactive 3D visualization saved to:\n{EXPORT_PATH}")

