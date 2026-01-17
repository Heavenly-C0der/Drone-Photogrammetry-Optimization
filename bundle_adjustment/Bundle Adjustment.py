import jax
import jax.numpy as jnp
import numpy as np
import pickle
import pandas as pd
import cv2
import time

# ============================
# PATHS
# ============================
OUTPUT_DIR = r"D:\SB Project\triangulated_results_for_BA"
CSV_PATH = r"D:\SB Project\image_metadata.csv"

# ============================
# LOAD METADATA
# ============================
metadata = pd.read_csv(CSV_PATH)
metadata["filename_norm"] = metadata["filename"].apply(lambda x: str(x).lower().split('.')[0].strip())
metadata_dict = metadata.set_index("filename_norm").T.to_dict()

# ============================
# LOAD PICKLES
# ============================
with open(f"{OUTPUT_DIR}/camera_poses.pkl", "rb") as f:
    camera_poses_dict = pickle.load(f)

with open(f"{OUTPUT_DIR}/triangulated_points.pkl", "rb") as f:
    triangulated_points_list = pickle.load(f)

with open(f"{OUTPUT_DIR}/observations.pkl", "rb") as f:
    observations = pickle.load(f)

# ============================
# CAMERA KEYS AND INDICES
# ============================
camera_keys = [str(k[0]).lower().strip() for k in camera_poses_dict.keys()]  # first cam in tuple
camera_indices = {k: i for i, k in enumerate(camera_keys)}
N_cams = len(camera_keys)

# Camera parameters: rvec + t
camera_params = []
for k in camera_keys:
    # Find the matching tuple key in camera_poses_dict
    tuple_key = next(tk for tk in camera_poses_dict.keys() if str(tk[0]).lower().strip() == k)
    R = camera_poses_dict[tuple_key]["R"]
    t = camera_poses_dict[tuple_key]["t"]
    rvec, _ = cv2.Rodrigues(np.array(R))
    camera_params.append(np.hstack([rvec.ravel(), t]))
camera_params = jnp.array(camera_params)  # shape (N_cams, 6)

# ============================
# TRIANGULATED POINTS
# ============================

points3D_list = [np.array(pair["X"], dtype=np.float64) for pair in triangulated_points_list]
points3D = jnp.array(np.vstack(points3D_list))  # shape (N_points, 3)
#points3D_mean = points3D.mean(axis=0)
#points3D = (points3D - points3D_mean) * 0.01  # scale down
N_points = points3D.shape[0]
print(f"Loaded {N_points} 3D points and {N_cams} cameras.")

# ============================
# FILTER OBSERVATIONS
# ============================
obs_filtered = []
for obs in observations:
    cam_key = str(obs[0]).lower().strip()
    if cam_key in camera_indices:
        obs_filtered.append(obs)

print(f"Filtered observations: {len(obs_filtered)} / {len(observations)} kept.")
observations = obs_filtered

# Build JAX arrays with correct integer dtype
obs_cam_idx = jnp.array([camera_indices[str(obs[0]).lower().strip()] for obs in observations], dtype=int)
obs_point_idx = jnp.array([int(obs[1]) for obs in observations], dtype=int)
obs_uv = jnp.array([[obs[2], obs[3]] for obs in observations], dtype=jnp.float32)

# ============================
# CAMERA INTRINSICS (first camera)
# ============================
img_key = camera_keys[0]
intr = metadata_dict[img_key]
fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
K = jnp.array([[fx, 0, cx],
               [0, fy, cy],
               [0, 0, 1]], dtype=jnp.float32)

print(f"Loaded {N_points} 3D points and {N_cams} cameras.")
print(f"Filtered observations: {len(observations)}")
print("Data loading complete. Ready for Gauss-Newton optimization.")


# ============================
# NORMALIZE POINTS AND CAMERA TRANSLATIONS
# ============================
points_mean = points3D.mean(axis=0)
scale_factor = 0.01  # reduce large coordinates
params_x = (points3D - points_mean) * scale_factor
params_c = camera_params.at[:, 3:].multiply(scale_factor)  # only scale translations

# ============================
# GAUSS-NEWTON STEP
# ============================
def rodrigues(r):
    theta = jnp.linalg.norm(r) + 1e-12
    r_hat = r / theta
    K_mat = jnp.array([[0, -r_hat[2], r_hat[1]],
                       [r_hat[2], 0, -r_hat[0]],
                       [-r_hat[1], r_hat[0], 0]])
    R = jnp.eye(3) + jnp.sin(theta)*K_mat + (1 - jnp.cos(theta))*(K_mat @ K_mat)
    return R

def project(point3D, cam_params):
    r = cam_params[:3]
    t = cam_params[3:]
    R = rodrigues(r)
    Xc = R @ point3D + t
    uv = Xc[:2] / Xc[2]
    uv = uv @ K[:2,:2].T + K[:2,2]
    return uv

def gauss_newton_step(params_c, params_x, alpha=1.0, lambda_damp=1e-3):
    # Observed points and cameras
    pts_obs = params_x[obs_point_idx]   # (N_obs, 3)
    cam_obs = params_c[obs_cam_idx]     # (N_obs, 6)

    # Residuals
    uv_proj = jax.vmap(project)(pts_obs, cam_obs)
    res = uv_proj - obs_uv              # (N_obs, 2)
    cost = 0.5 * jnp.sum(res**2)

    # Jacobians
    def jac_fn(c, p):
        J_c = jax.jacobian(lambda c_: project(p, c_))(c)  # (2,6)
        J_x = jax.jacobian(lambda p_: project(p_, c))(p)  # (2,3)
        return J_c, J_x

    J_c_list, J_x_list = jax.vmap(jac_fn)(cam_obs, pts_obs)  # (N_obs,2,6), (N_obs,2,3)

    # Gradients
    g_C = jax.ops.segment_sum(jnp.einsum('nij,ni->nj', J_c_list, res), obs_cam_idx).reshape(-1,6)
    g_X = jax.ops.segment_sum(jnp.einsum('nij,ni->nj', J_x_list, res), obs_point_idx)

    # Apply simple damping (like Levenberg-Marquardt)
    delta_C = - alpha * g_C / (jnp.linalg.norm(g_C, axis=1, keepdims=True) + lambda_damp)
    delta_X = - alpha * g_X / (jnp.linalg.norm(g_X, axis=1, keepdims=True) + lambda_damp)

    return delta_C, delta_X, cost

# ============================
# MAIN LOOP
# ============================
max_iter = 20
tol = 1e-6
alpha = 1.0
lambda_damp = 1e-3

cost_history = []

for itr in range(max_iter):
    t0 = time.time()
    delta_C, delta_X, cost = gauss_newton_step(params_c, params_x, alpha, lambda_damp)
    
    # Update parameters
    params_c = params_c + delta_C
    params_x = params_x + delta_X

    max_update = jnp.maximum(jnp.linalg.norm(delta_C), jnp.linalg.norm(delta_X))
    cost_history.append(cost.item())
    print(f"[Itr {itr+1}] Cost={cost:.6f}, MaxUpdate={max_update:.6e}, Time={time.time()-t0:.2f}s")

    if max_update < tol:
        print(f"Converged at iteration {itr+1}")
        break

