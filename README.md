# Application of Optimization in Drone-based Photogrammetry

## Overview

This repository presents an academic implementation and analysis of optimization techniques applied to **drone-based photogrammetry**. The project demonstrates how classical numerical optimization methods can significantly improve the accuracy of 3D reconstruction, camera pose estimation, and ground control point (GCP) alignment in UAV-based mapping workflows.

The work was developed as part of the course **MDTS4311** and focuses on minimizing reprojection errors, correcting RTK-based drone trajectories, and jointly optimizing overall photogrammetric cost functions.

---

## Objectives

* Understand the photogrammetric reconstruction pipeline
* Minimize **reprojection error** using Bundle Adjustment
* Correct **drone RTK misalignment** using Ground Control Points (GCPs)
* Compare and justify the use of different optimization techniques
* Reduce the **overall cost function** by combining multiple error sources

---

## Photogrammetry Pipeline

1. **Image Acquisition** using UAV-mounted cameras
2. **Feature Extraction & Matching**
3. **Initial Triangulation** of 3D points
4. **Bundle Adjustment Optimization**
5. **GCP Alignment Optimization**
6. **Overall Cost Optimization**
7. **Visualization and Evaluation**

---

## Bundle Adjustment Optimization

### Purpose

Refine camera parameters and 3D point locations by minimizing reprojection error.

### Optimization Method

**BFGS (Broyden–Fletcher–Goldfarb–Shanno)**

**Rationale:**

* Newton’s method requires computing and storing a very large Hessian matrix (≈ 6132 × 6132)
* BFGS provides a quasi-Newton approximation with significantly lower memory and computational cost
* More stable for large-scale photogrammetric problems

### Optimization Objective

Minimize the reprojection error:

[\sum_{i,j} | x_{ij} - \Pi(R_i, t_i, X_j) |^2]

Where:

* (R_i, t_i): Rotation and translation of camera *i*
* (X_j): 3D coordinates of point *j*
* (x_{ij}): Observed 2D image point
* (\Pi(·)): 3D → 2D projection function

### Results

* Converged in **10,704 iterations**
* Significant reduction in reprojection error
* Interactive 3D visualization supported

---

## GCP Alignment Optimization

### Purpose

Correct drone trajectory errors caused by RTK inaccuracies using known Ground Control Points.

### Optimization Method

**Newton’s Method**

### Preprocessing Steps

1. **Nearest Neighbor Matching**

   * Each GCP is matched with the nearest drone position using 3D geodesic distance (`pyproj.Geod.inv`)

2. **Baseline Error Calculation**

   * Initial 3D RMSE computed before correction

3. **Coordinate Transformation**

   * Convert WGS84 (EPSG:4326) latitude/longitude to local **ENU (East, North, Up)** coordinates in meters

### Optimization Outcome

* Drone positions translated closer to true GCP locations
* Substantial reduction in RMSE
* Clear improvement visible in ENU-plane visualizations

---

## Overall Cost Optimization

### Purpose

Jointly reduce reprojection error and GCP alignment error.

### Optimization Method

**Fibonacci Search**

### Combined Cost Function

A weighted linear combination of:

* Bundle Adjustment reprojection error
* GCP alignment error

This approach balances internal geometric consistency with external geospatial accuracy.

---

## Results & Performance

### GCP Misalignment Optimization (Newton's Method)

**Estimated Translation Offsets (ENU):**

* dx (East): **+0.367 m**
* dy (North): **−1.124 m**
* dz (Up): **−51.300 m**

**Estimated Orientation Offsets:**

* Roll: **+0.568°**
* Pitch: **−0.297°**
* Yaw: **+1.144°**

**3D RMSE Improvement:**

* Before optimization: **67.635 m**
* After Newton optimization: **3.639 m**
* **Error reduction: 94.6%**

The Newton-based georeferencing optimization converged to the solution in **2 iterations**, indicating a well-conditioned cost function and accurate Jacobian formulation.

---

### Bundle Adjustment Optimization (BFGS)

* Initial reprojection RMS error: **18.71 px**
* Final reprojection RMS error: **2.91 px**
* Total iterations to convergence: **10,704**

The gradual but stable convergence validates the use of BFGS over full Newton optimization for high-dimensional parameter spaces involving camera poses and 3D points.

---

### Overall Cost Optimization (Fibonacci Search)

* Optimization variable: **λ (weighting factor)**
* Optimal λ: **0.000455**
* Minimum combined cost: **0.23433 m**

The Fibonacci search efficiently narrowed the interval and converged within **18 iterations**, yielding a balanced trade-off between reprojection accuracy and geospatial alignment.

---

## Visualizations

* ENU plane alignment (Before vs After)
* Reprojection error convergence curves
* Fibonacci interval contraction plots

---

## Technologies & Libraries

* Python
* NumPy
* SciPy
* OpenCV
* PyProj
* Matplotlib / Plotly

---

## Repository Structure
```
├── data/
│   └── [odm_data_helenenschacht-main](https://github.com/OpenDroneMap/odm_data_helenenschacht/tree/main)  # Input dataset (images, metadata)
│
├── metadata_extraction/                # EXIF & camera metadata parsing
│
├── feature_detection/                  # Keypoints & descriptors detection
│
├── feature_matching/                   # Descriptor matching (KNN / ratio test)
│
├── triangulation/                      # 3D point triangulation
│   └── tests/                          # Triangulation error handling tests
│
├── bundle_adjustment/                  # Reprojection error minimization (BFGS)
│
├── gcp_misalignment/                   # GCP-based RTK correction (Newton)
│
├── overall_optimization/               # Fibonacci search for joint cost
│
├── visualization/                      # 2D / 3D plots and alignment visuals
│
├── results/                            # Metrics, RMSE, convergence outputs
│
├── requirements.txt
├── README.md
```

---

## References

- [DJI Enterprise – Ground Control Points](https://enterprise.dji.com)
- [MDPI Remote Sensing (2018)](https://www.mdpi.com/journal/remotesensing)
- [MDPI Journal of Imaging (2023)](https://www.mdpi.com/journal/jimaging)
- [Agisoft Photogrammetry Forum](https://www.agisoft.com/forum/)
- [Baeldung – Bundle Adjustment](https://www.baeldung.com/cs/bundle-adjustment)
- [Artec3D Learning Center](https://www.artec3d.com/learning-center)
- [Helenenschacht ODM Dataset](https://github.com/OpenDroneMap/odm_data_helenenschacht/tree/main)

---


## Notes
This project is academic in nature and intended for learning and demonstration purposes. The optimization techniques used here can be extended to real-world large-scale photogrammetric systems with appropriate computational resources.

```
