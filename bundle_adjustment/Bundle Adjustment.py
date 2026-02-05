# ba_jax_lbfgs.py
"""
Bundle Adjustment using JAX autodiff + SciPy BFGS.

- Minimize mean squared reprojection error (RMS-equivalent).
- Uses angle-axis (Rodrigues) for camera rotations and camera centers (meters).
- Intrinsics read from CSV and kept fixed. Distortion kept fixed.
- Prints per-iteration RMS and absolute gradient statistics.
"""
import csv
import os
import sys
import time
import pickle
import numpy as np
import pandas as pd
from functools import partial
from scipy.optimize import minimize
from tqdm import tqdm

# ---- JAX setup ----
import jax
import jax.numpy as jnp
# enable float64 in JAX for better numeric stability
jax.config.update("jax_enable_x64", True)

# ---- CONFIG ----
TRIANG_FILE = r"D:\SB Project\triangulation_output_distortion_new\triangulated_points_filtered_final.pkl"
POSES_FILE  = r"D:\SB Project\triangulation_output_distortion_new\camera_poses_ransac_distortion.pkl"
CSV_PATH    = r"D:\SB Project\image_metadata.csv"   # to read intrinsics
OUTPUT_DIR  = r"D:\SB Project\Ba_BFGS_jax_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optimization settings
MAX_ITER = 30
REG_PRIOR_WEIGHT = 1e-3    # small L2 prior on parameter deltas to stabilize gradients
GRAD_CLIP_MAX = 1e6        # clip gradient components beyond this (to avoid inf/nan propagation)
PRINT_EVERY = 1

# Distortion (keep fixed)
DIST_COEFFS = jnp.array([-0.1, 0.02, 0.0, 0.0, 0.0], dtype=jnp.float64)

# ---- Utilities: Rodrigues (angle-axis) -> rotation matrix (JAX) ----
def skew(v):
    return jnp.array([[0.0, -v[2], v[1]],
                      [v[2], 0.0, -v[0]],
                      [-v[1], v[0], 0.0]], dtype=jnp.float64)

@jax.jit
def rodrigues_matrix(rotvec):
    """Convert 3-vector (angle-axis) to 3x3 rotation matrix. Handles near-zero angle with series."""
    theta = jnp.linalg.norm(rotvec)
    # safeguard small theta
    def small():
        # R ≈ I + [r]_x  (first-order)
        return jnp.eye(3, dtype=jnp.float64) + skew(rotvec)
    def large():
        k = rotvec / theta
        K = skew(k)
        return jnp.eye(3, dtype=jnp.float64) + jnp.sin(theta) * K + (1.0 - jnp.cos(theta)) * (K @ K)
    return jax.lax.cond(theta < 1e-8, small, large)

# ---- Distortion & projection (JAX) ----
@jax.jit
def distort_and_project_point(X_world, rotvec, C_world, K, dist_coeffs):
    """
    Project a single 3D point X_world (3,) into a camera parameterized by rotvec (angle-axis) and C_world (3,).
    K is (4,) = [fx, fy, cx, cy]
    dist_coeffs: (5,) -> k1,k2,p1,p2,k3
    Returns projected pixel (u,v).
    """
    R = rodrigues_matrix(rotvec)  # R_w2c
    # transform: X_cam = R @ (X_world - C_world)
    Xc = R @ (X_world - C_world)
    x = Xc[0] / (Xc[2] + 1e-12)
    y = Xc[1] / (Xc[2] + 1e-12)
    r2 = x*x + y*y
    k1, k2, p1, p2, k3 = dist_coeffs
    radial = 1.0 + k1*r2 + k2*r2**2 + k3*r2**3
    x_dist = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    y_dist = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    fx, fy, cx, cy = K
    u = fx * x_dist + cx
    v = fy * y_dist + cy
    return jnp.stack([u, v])

# ---- Load data (triangulated points + camera poses + intrinsics CSV) ----
print("Loading data...")
with open(TRIANG_FILE, "rb") as fh:
    triangulated = pickle.load(fh)  # list of (X (3,), obs list)
with open(POSES_FILE, "rb") as fh:
    camera_poses = pickle.load(fh)   # dict img -> {"R":R_w2c, "C":C_centered, "K":K, "P":P}

# read intrinsics from CSV (assumes filename_norm logic uses uppercase no extension)
df = pd.read_csv(CSV_PATH)
df["filename_norm"] = df["filename"].apply(lambda x: os.path.splitext(str(x).upper())[0])
meta_dict = df.set_index("filename_norm").T.to_dict()

# Build camera list (only those present in camera_poses)
camera_list = sorted(list(camera_poses.keys()))
cam_idx = {img: i for i, img in enumerate(camera_list)}
n_cams = len(camera_list)
n_points = len(triangulated)

print(f"n_cams={n_cams}, n_points={n_points}")

# Build observations list: each observation is (cam_idx, point_idx, u_obs, v_obs)
obs_records = []
for p_idx, (X, obs) in enumerate(triangulated):
    for img, (u, v) in obs:
        if img not in cam_idx:
            continue
        obs_records.append((cam_idx[img], p_idx, float(u), float(v)))
n_obs = len(obs_records)
print(f"Total observations: {n_obs}")

# Build per-camera intrinsics vector K (fx,fy,cx,cy) from CSV; fallback to camera_poses K if missing
K_camera = []
for img in camera_list:
    fname_norm = os.path.splitext(img)[0].upper()
    if fname_norm in meta_dict:
        m = meta_dict[fname_norm]
        fx = float(m["fx"])
        fy = float(m["fy"])
        cx = float(m["cx"])
        cy = float(m["cy"])
    else:
        Km = camera_poses[img]["K"]
        fx, fy, cx, cy = float(Km[0,0]), float(Km[1,1]), float(Km[0,2]), float(Km[1,2])
    K_camera.append(jnp.array([fx, fy, cx, cy], dtype=jnp.float64))
K_camera = jnp.stack(K_camera)  # (n_cams,4)

# Prepare initial parameters
# Camera rotations: derive angle-axis from initial R_w2c via inverse Rodrigues (we have R_w2c matrices)
def rotmat_to_rotvec_np(Rm):
    # use cv2 if available; else use numpy fallback (we'll try numpy)
    try:
        import cv2
        rvec, _ = cv2.Rodrigues(Rm)
        return rvec.reshape(3)
    except Exception:
        # numpy fallback: axis-angle from rotation matrix
        theta = np.arccos((np.trace(Rm) - 1.0) / 2.0)
        if np.isclose(theta, 0.0):
            return np.zeros(3)
        rx = (Rm[2,1] - Rm[1,2]) / (2*np.sin(theta))
        ry = (Rm[0,2] - Rm[2,0]) / (2*np.sin(theta))
        rz = (Rm[1,0] - Rm[0,1]) / (2*np.sin(theta))
        axis = np.array([rx, ry, rz])
        return axis * theta

cam_rot0 = np.zeros((n_cams, 3), dtype=np.float64)
cam_C0   = np.zeros((n_cams, 3), dtype=np.float64)
for i, img in enumerate(camera_list):
    cam = camera_poses[img]
    Rmat = np.asarray(cam["R"], dtype=np.float64)
    cam_rot0[i] = rotmat_to_rotvec_np(Rmat)
    cam_C0[i]   = np.asarray(cam["C"], dtype=np.float64)  # centered meters

points0 = np.vstack([X for X, obs in triangulated]).astype(np.float64)  # (n_points,3)

# Pack parameter vector: [cams_rot (n_cams*3), cams_C (n_cams*3), points (n_points*3)]
def pack_to_1d(cam_rot, cam_C, points):
    return np.hstack([cam_rot.reshape(-1), cam_C.reshape(-1), points.reshape(-1)])

def unpack_from_1d(vec):
    idx = 0
    cr = vec[idx: idx + 3*n_cams].reshape((n_cams,3)); idx += 3*n_cams
    cC = vec[idx: idx + 3*n_cams].reshape((n_cams,3)); idx += 3*n_cams
    pts = vec[idx: idx + 3*n_points].reshape((n_points,3)); idx += 3*n_points
    return cr, cC, pts

x0 = pack_to_1d(cam_rot0, cam_C0, points0)

# Keep initial parameters as JAX arrays for regularizer
cam_rot0_j = jnp.asarray(cam_rot0)
cam_C0_j   = jnp.asarray(cam_C0)
points0_j  = jnp.asarray(points0)

# Convert obs_records and K_camera to jax arrays for fast use
obs_cam_idx = jnp.asarray([int(r[0]) for r in obs_records], dtype=jnp.int32)
obs_pt_idx  = jnp.asarray([int(r[1]) for r in obs_records], dtype=jnp.int32)
obs_u       = jnp.asarray([r[2] for r in obs_records], dtype=jnp.float64)
obs_v       = jnp.asarray([r[3] for r in obs_records], dtype=jnp.float64)

# Vectorized reprojection for all observations
@partial(jax.jit, static_argnums=())
def objective_and_residuals(params_vec):
    """
    Returns (mean_squared_error, residuals_vector (2*n_obs,))
    params_vec is a flat 1D array (numpy -> converted to jax when called via jax)
    """
    # unpack params
    cr_flat = params_vec[0:3*n_cams].reshape((n_cams,3))
    cc_flat = params_vec[3*n_cams:6*n_cams].reshape((n_cams,3))
    pts_flat = params_vec[6*n_cams:6*n_cams + 3*n_points].reshape((n_points,3))

    # gather per-observation values
    cam_rots = cr_flat[obs_cam_idx]   # (n_obs,3)
    cam_Cs   = cc_flat[obs_cam_idx]   # (n_obs,3)
    Ks       = K_camera[obs_cam_idx]  # (n_obs,4)
    # points for each observation
    Pts_obs  = pts_flat[obs_pt_idx]   # (n_obs,3)

    # vectorized projection: map along leading axis
    def proj_one(pX, rvec, C, K):
        return distort_and_project_point(pX, rvec, C, K, DIST_COEFFS)

    proj = jax.vmap(proj_one)(Pts_obs, cam_rots, cam_Cs, Ks)  # (n_obs,2)
    res_u = proj[:,0] - obs_u
    res_v = proj[:,1] - obs_v
    residuals = jnp.stack([res_u, res_v], axis=1).reshape(-1)  # (2*n_obs,)
    rms = jnp.sqrt(jnp.mean(residuals**2) + 1e-12)  # RMS reprojection error (px)
    loss = rms + REG_PRIOR_WEIGHT * (
        jnp.sum((cr_flat - cam_rot0_j)**2) +
        jnp.sum((cc_flat - cam_C0_j)**2) +
        jnp.sum((pts_flat - points0_j)**2)
    )
    return loss, residuals


def rms_only(params_vec):
    cr_flat = params_vec[0:3*n_cams].reshape((n_cams,3))
    cc_flat = params_vec[3*n_cams:6*n_cams].reshape((n_cams,3))
    pts_flat = params_vec[6*n_cams:6*n_cams + 3*n_points].reshape((n_points,3))
    cam_rots = cr_flat[obs_cam_idx]
    cam_Cs   = cc_flat[obs_cam_idx]
    Ks       = K_camera[obs_cam_idx]
    Pts_obs  = pts_flat[obs_pt_idx]

    def proj_one(pX, rvec, C, K):
        return distort_and_project_point(pX, rvec, C, K, DIST_COEFFS)

    proj = jax.vmap(proj_one)(Pts_obs, cam_rots, cam_Cs, Ks)
    res_u = proj[:,0] - obs_u
    res_v = proj[:,1] - obs_v
    residuals = jnp.stack([res_u, res_v], axis=1).reshape(-1)
    rms = jnp.sqrt(jnp.mean(residuals**2) + 1e-12)
    return float(rms)



# JIT-compiled objective function and gradient
jax_obj = jax.jit(lambda pv: objective_and_residuals(pv)[0])
jax_obj_and_grad = jax.jit(jax.value_and_grad(jax_obj))

# wrapper to give SciPy objective (numpy -> jax -> numpy)
def scipy_obj_and_grad(x_numpy):
    xj = jnp.asarray(x_numpy, dtype=jnp.float64)
    val, grad = jax_obj_and_grad(xj)
    val_np = np.asarray(val, dtype=np.float64)
    grad_np = np.asarray(grad, dtype=np.float64)
    # clip huge gradient values to avoid blow-ups communicated to SciPy
    grad_np = np.clip(grad_np, -GRAD_CLIP_MAX, GRAD_CLIP_MAX)
    return val_np, grad_np

# ---- Prepare CSV Writer for Iteration Logs ----
iter_log_path = os.path.join(OUTPUT_DIR, "ba_iteration_log.csv")
with open(iter_log_path, "w", newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Iteration", "RMS_px", "Parameters"])
    # store file handler to append data later
    
def format_params(params, n_head=5, n_tail=5):
    """
    Format the parameter array to show only the first few and last few values.
    Adds '...' in between if the array is large.
    """
    if params.size <= n_head + n_tail:
        return np.array2string(params, precision=4)
    else:
        head = np.array2string(params[:n_head], precision=10)
        tail = np.array2string(params[-n_tail:], precision=10)
        return f"{head[:-1]}, ... , {tail[1:]}"


# For printing per-iteration statistics we create callback that computes RMS and grad stats
iteration = {"k": 0}
last_grad = None

def callback_scipy(xk):
    iteration["k"] += 1
    val, grad = scipy_obj_and_grad(xk)

    # Compute RMS based on residuals
    xj = jnp.asarray(xk, dtype=jnp.float64)
    mse, residuals = objective_and_residuals(xj)
    #rms = float(mse)  # RMS already implemented as objective
    rms = rms_only(xj)
    grad_abs_max = np.max(np.abs(grad))
    grad_norm = np.linalg.norm(grad)
    compact_params = format_params(xk)  # format the parameters

    # Print progress
    print(f"Iter {iteration['k']:4d}: RMS={rms:.6f} px, |grad|_2={grad_norm:.6e}")
    sys.stdout.flush()

    # Write to CSV
    with open(iter_log_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([iteration["k"], rms, compact_params])

    # Sanity checks
    if not np.isfinite(rms) or not np.isfinite(grad_abs_max):
        raise RuntimeError("Non-finite RMS or gradient encountered — aborting optimization.")

# ---- Run SciPy BFGS with JAX gradient ----
print("Warming up JAX compilation (first call can take a bit)...")
v0 = jnp.asarray(x0, dtype=jnp.float64)
# run one evaluation to JIT compile
_val, _grad = jax_obj_and_grad(v0)
print("JAX compiled.")

# ---- Compute and print initial RMS before optimization ----
print("Computing initial RMS reprojection error...")

xj0 = jnp.asarray(x0, dtype=jnp.float64)
init_loss, _ = objective_and_residuals(xj0)
#init_rms = float(init_loss)
init_rms = rms_only(x0)
print(f"\nInitial RMS before BA: {init_rms:.4f} px\n")

with open(iter_log_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([0, init_rms, x0])
        
print("Starting SciPy BFGS optimization...")
t0 = time.time()
res = minimize(fun=lambda x: scipy_obj_and_grad(x)[0],
               x0=x0,
               jac=lambda x: scipy_obj_and_grad(x)[1],
               method="L-BFGS-B",
               options={"maxiter": 20000, "disp": True,
                    "gtol" : 2e-1,
                },
               callback=callback_scipy)
t1 = time.time()
print(f"Optimization finished in {t1-t0:.1f}s with message: {res.message}")


import pandas as pd
import numpy as np
import os
import pickle

# ---- Load camera metadata ----
metadata_path = os.path.join("D:\SB Project", "image_metadata.csv")  # adjust if needed
df_meta = pd.read_csv(metadata_path)
df_meta["filename"] = df_meta["filename"].str.strip()

meta_dict = {row["filename"]: row for _, row in df_meta.iterrows()}

# ---- Unpack optimized params ----
cam_rot_opt, cam_C_opt, pts_opt = unpack_from_1d(res.x)
cam_rot_opt = cam_rot_opt.astype(np.float64)
cam_C_opt   = cam_C_opt.astype(np.float64)
pts_opt     = pts_opt.astype(np.float64)

# ---- Build refined camera poses with metadata ----
refined_camera_poses = {}

for i, img in enumerate(camera_list):
    # get rotation matrix from Rodrigues
    rotvec = cam_rot_opt[i]
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        Rm = np.eye(3)
    else:
        k = rotvec / theta
        K_skew = np.array([
            [0, -k[2], k[1]],
            [k[2], 0, -k[0]],
            [-k[1], k[0], 0]
        ], dtype=np.float64)
        Rm = np.eye(3) + np.sin(theta)*K_skew + (1.0 - np.cos(theta))*(K_skew @ K_skew)

    C = cam_C_opt[i]
    Kmat = np.array([
        [float(K_camera[i, 0]), 0.0, float(K_camera[i, 2])],
        [0.0, float(K_camera[i, 1]), float(K_camera[i, 3])],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    t = (-Rm @ C.reshape(3, 1)).reshape(3)
    P = Kmat @ np.hstack([Rm, t.reshape(3, 1)])

    # --- Attach image metadata from CSV ---
    meta = meta_dict.get(img, {})
    fx = float(meta.get("fx", Kmat[0, 0]))
    fy = float(meta.get("fy", Kmat[1, 1]))
    cx = float(meta.get("cx", Kmat[0, 2]))
    cy = float(meta.get("cy", Kmat[1, 2]))
    width = int(meta.get("image_width_px", 5472))
    height = int(meta.get("image_height_px", 3648))
    focal_length_mm = float(meta.get("focal_length_mm", np.nan))
    sensor_width_mm = float(meta.get("sensor_width_mm", np.nan))
    sensor_height_mm = float(meta.get("sensor_height_mm", np.nan))
    yaw_deg = float(meta.get("yaw_deg", np.nan))
    pitch_deg = float(meta.get("pitch_deg", np.nan))
    roll_deg = float(meta.get("roll_deg", np.nan))

    refined_camera_poses[img] = {
        "image_name": img,
        "R": Rm,
        "C": C,
        "t": t,
        "K": Kmat,
        "P": P,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "width": width,
        "height": height,
        "focal_length_mm": focal_length_mm,
        "sensor_width_mm": sensor_width_mm,
        "sensor_height_mm": sensor_height_mm,
        "yaw_deg": yaw_deg,
        "pitch_deg": pitch_deg,
        "roll_deg": roll_deg,
        "dist": np.asarray(DIST_COEFFS, dtype=np.float64),
    }

# ---- Save results ----
OUTPUT_DIR = os.path.join("D:\SB Project", "Ba_BFGS_jax_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "ba_refined_camera_poses_jax.pkl"), "wb") as fh:
    pickle.dump(refined_camera_poses, fh)
with open(os.path.join(OUTPUT_DIR, "ba_refined_points_jax.pkl"), "wb") as fh:
    pickle.dump(pts_opt, fh)

print(f"Saved refined poses for {len(refined_camera_poses)} images and {len(pts_opt)} points → {OUTPUT_DIR}")

# ---- Compute final per-observation reprojection errors and print global stats ----
all_errs = []
for (ci, pi, uo, vo) in obs_records:
    uo = float(uo); vo = float(vo)
    X = pts_opt[pi]
    rotvec = cam_rot_opt[ci]
    Cw = cam_C_opt[ci]
    Kvec = np.array(K_camera[ci], dtype=np.float64)
    # compute projected pixel using same math as in JAX (numpy implementation)
    theta = np.linalg.norm(rotvec)
    if theta < 1e-8:
        Rm = np.eye(3)
    else:
        k = rotvec / theta
        Kx = np.array([[0, -k[2], k[1]],[k[2], 0, -k[0]],[-k[1], k[0], 0]], dtype=np.float64)
        Rm = np.eye(3) + np.sin(theta)*Kx + (1.0 - np.cos(theta))*(Kx @ Kx)
    Xc = Rm @ (X - Cw)
    x = Xc[0] / (Xc[2] + 1e-12); y = Xc[1] / (Xc[2] + 1e-12)
    r2 = x*x + y*y
    k1,k2,p1,p2,k3 = np.asarray(DIST_COEFFS, dtype=np.float64)
    radial = 1.0 + k1*r2 + k2*r2**2 + k3*r2**3
    x_dist = x*radial + 2*p1*x*y + p2*(r2 + 2*x*x)
    y_dist = y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y
    fx, fy, cx, cy = Kvec
    u_proj = fx * x_dist + cx
    v_proj = fy * y_dist + cy
    err = np.sqrt((u_proj - uo)**2 + (v_proj - vo)**2)
    all_errs.append(err)

all_errs = np.array(all_errs, dtype=np.float64)
global_mean = np.mean(all_errs)
global_rms = np.sqrt(np.mean(all_errs**2))
global_max = np.max(all_errs)

print("\nFINAL GLOBAL REPROJECTION STATS (post-BA):")
print(f"Total observations considered: {len(all_errs)}")
print(f"Mean reprojection error: {global_mean:.6f} px")
print(f"RMS reprojection error:  {global_rms:.6f} px")
print(f"Max reprojection error:  {global_max:.6f} px")

# Save per-observation errors CSV
import csv
csv_path = os.path.join(OUTPUT_DIR, "ba_per_obs_errors_jax.csv")
with open(csv_path, "w", newline='') as fh:
    w = csv.writer(fh)
    w.writerow(["camera","point_idx","u_obs","v_obs","err_px"])
    for (ci, pi, uo, vo), err in zip(obs_records, all_errs):
        w.writerow([camera_list[int(ci)], int(pi), float(uo), float(vo), float(err)])
print(f"Saved per-observation errors to {csv_path}")

print("All done. Refined camera poses and points saved in:", OUTPUT_DIR)
