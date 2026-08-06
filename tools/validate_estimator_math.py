#!/usr/bin/env python3
"""Numerical validation of the C++ estimator maths.

This mirrors, formula for formula, the code in
``ros_ws/src/oak_vio/src/motion_estimator.cpp`` so that the residual, the
analytic Jacobian, the weighting and the Gauss-Newton update can be checked
against numerical derivatives and known ground truth on a workstation with no
C++ toolchain, no ROS and no camera.

It is a correctness oracle, not a reimplementation: if a formula changes in the
C++, change it here too and re-run. Any divergence between the two is a bug in
one of them.

Run:  python tools/validate_estimator_math.py
"""

from __future__ import annotations

import sys

import numpy as np

# Camera matching the plan's recommended rectified configuration for the
# OAK 4 D W: ~100 deg HFoV pinhole after alpha scaling, 75 mm baseline.
FX = FY = 537.0
CX, CY = 640.0, 400.0
BASELINE = 0.075
FX_B = FX * BASELINE

WIDTH, HEIGHT = 1280, 800

# Measurement noise. PIXEL_SIGMA is corner localisation from the HW tracker.
# DISPARITY_SIGMA is block-matcher noise -- note this is the *noise*, not the
# 1/16 px subpixel quantisation step. Conflating the two understates depth
# error by roughly an order of magnitude.
PIXEL_SIGMA = 0.3
DISPARITY_SIGMA = 0.5

# Matches the damping in MotionEstimator::refine.
DAMPING = 1e-7

# Matches VioParams::covarianceInflation.
COVARIANCE_INFLATION = 3.0

RNG = np.random.default_rng(20260806)


# --------------------------------------------------------------------------
# Lie group helpers -- mirrors types.hpp
# --------------------------------------------------------------------------
def skew(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def exp_so3(omega):
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        return np.eye(3) + skew(omega)
    axis = omega / theta
    K = skew(axis)
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * K @ K


def log_so3(R):
    cos_theta = np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if theta < 1e-10:
        return w * 0.5
    return w * (theta / (2.0 * np.sin(theta)))


# --------------------------------------------------------------------------
# Camera model -- mirrors rectified_camera.hpp
# --------------------------------------------------------------------------
def project(p):
    """3D point -> (u, v, disparity)."""
    inv_z = 1.0 / p[2]
    return np.array([FX * p[0] * inv_z + CX, FY * p[1] * inv_z + CY, FX_B * inv_z])


def unproject(u, v, d):
    z = FX_B / d
    return np.array([(u - CX) * z / FX, (v - CY) * z / FY, z])


def projection_jacobian(p):
    inv_z = 1.0 / p[2]
    inv_z2 = inv_z * inv_z
    j = np.zeros((3, 3))
    j[0, 0] = FX * inv_z
    j[0, 2] = -FX * p[0] * inv_z2
    j[1, 1] = FY * inv_z
    j[1, 2] = -FY * p[1] * inv_z2
    j[2, 2] = -FX_B * inv_z2
    return j


# --------------------------------------------------------------------------
# Estimator -- mirrors motion_estimator.cpp
# --------------------------------------------------------------------------
def apply_pose(R, t, p):
    return R @ p + t


def perturb(R, t, delta):
    """Left perturbation: R <- Exp(phi) R, t <- Exp(phi) t + rho."""
    dR = exp_so3(delta[3:])
    return dR @ R, dR @ t + delta[:3]


def evaluate(point_kf, z_cur, R, t, disparity_kf=None,
             pixel_sigma=PIXEL_SIGMA, disparity_sigma=DISPARITY_SIGMA):
    """Mirrors MotionEstimator::evaluate: residual, Jacobian, variance."""
    transformed = apply_pose(R, t, point_kf)
    if transformed[2] <= 1e-3:
        return None
    proj_jac = projection_jacobian(transformed)
    residual = project(transformed) - z_cur
    point_jac = np.hstack([np.eye(3), -skew(transformed)])
    jacobian = proj_jac @ point_jac

    variance = np.array([pixel_sigma**2, pixel_sigma**2, disparity_sigma**2])
    if disparity_kf is not None and disparity_kf > 0.0:
        # sigma_Z / Z == sigma_d / d, rank-1 along the keyframe viewing ray.
        depth_kf = point_kf[2]
        sigma_z = depth_kf * disparity_sigma / disparity_kf
        ray_kf = point_kf / np.linalg.norm(point_kf)
        sensitivity = proj_jac @ (R @ ray_kf)
        variance = variance + (sigma_z**2) * (sensitivity**2)

    return residual, jacobian, variance


def residual_and_jacobian(point_kf, z_cur, R, t):
    ev = evaluate(point_kf, z_cur, R, t, disparity_kf=None)
    if ev is None:
        return None, None
    return ev[0], ev[1]


def rigid_align(a, b):
    """Umeyama without scale. Mirrors rigidAlign() in motion_estimator.cpp."""
    ca, cb = a.mean(axis=0), b.mean(axis=0)
    cov = (b - cb).T @ (a - ca)
    U, _, Vt = np.linalg.svd(cov)
    correction = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        correction[2, 2] = -1.0
    R = U @ correction @ Vt
    return R, cb - R @ ca


def gauss_newton(points_kf, measurements, R0, t0, disparities_kf=None, max_iters=15,
                 huber=1.5, pixel_sigma=PIXEL_SIGMA, disparity_sigma=DISPARITY_SIGMA,
                 eps=1e-9):
    """Mirrors MotionEstimator::refine. Returns (R, t, H, iterations)."""
    R, t = R0.copy(), t0.copy()
    H = np.eye(6)
    iterations = 0
    for iterations in range(1, max_iters + 1):
        H = np.zeros((6, 6))
        g = np.zeros(6)
        used = 0
        for i, (pk, z) in enumerate(zip(points_kf, measurements)):
            dkf = None if disparities_kf is None else disparities_kf[i]
            ev = evaluate(pk, z, R, t, dkf, pixel_sigma, disparity_sigma)
            if ev is None:
                continue
            r, J, var = ev
            for row in range(3):
                w = (1.0 if abs(r[row]) <= huber else huber / abs(r[row])) / var[row]
                jr = J[row]
                H += w * np.outer(jr, jr)
                g -= w * r[row] * jr
            used += 1
        if used < 3:
            break
        H = H + np.eye(6) * (DAMPING * np.trace(H) / 6.0)
        try:
            delta = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        if not np.all(np.isfinite(delta)):
            break
        R, t = perturb(R, t, delta)
        if np.linalg.norm(delta) < eps:
            break
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt, t, H, iterations


# --------------------------------------------------------------------------
# Synthetic scene
# --------------------------------------------------------------------------
def make_scene(n, min_depth=1.0, max_depth=25.0, triangulate_from_noisy=False):
    """Random visible 3D points with usable disparity.

    With triangulate_from_noisy=True the returned keyframe points are
    triangulated from *noisy* disparity, as they are on real hardware. That is
    what exercises the depth-uncertainty weighting.
    """
    true_pts, kf_pts, kf_disparities = [], [], []
    while len(true_pts) < n:
        u = RNG.uniform(0.05 * WIDTH, 0.95 * WIDTH)
        v = RNG.uniform(0.05 * HEIGHT, 0.95 * HEIGHT)
        z = np.exp(RNG.uniform(np.log(min_depth), np.log(max_depth)))
        d_true = FX_B / z
        if d_true < 1.0:
            continue
        true_pts.append(unproject(u, v, d_true))
        if triangulate_from_noisy:
            d_meas = d_true + RNG.normal(0, DISPARITY_SIGMA)
            if d_meas < 1.0:
                true_pts.pop()
                continue
            kf_pts.append(unproject(u + RNG.normal(0, PIXEL_SIGMA),
                                    v + RNG.normal(0, PIXEL_SIGMA), d_meas))
            kf_disparities.append(d_meas)
        else:
            kf_pts.append(true_pts[-1])
            kf_disparities.append(d_true)
    return np.array(true_pts), np.array(kf_pts), np.array(kf_disparities)


def observe(true_pts, kf_pts, kf_disparities, R, t, pixel_noise=0.0, disparity_noise=0.0):
    """Project the true points into the current frame; keep the visible ones."""
    measurements, kept = [], []
    for i, p in enumerate(true_pts):
        q = apply_pose(R, t, p)
        if q[2] <= 0.1:
            continue
        z = project(q)
        if not (0 <= z[0] < WIDTH and 0 <= z[1] < HEIGHT) or z[2] < 1.0:
            continue
        measurements.append(z + np.array([RNG.normal(0, pixel_noise),
                                          RNG.normal(0, pixel_noise),
                                          RNG.normal(0, disparity_noise)]))
        kept.append(i)
    return kf_pts[kept], np.array(measurements), kf_disparities[kept]


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
FAILURES = []


def check(name, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def test_jacobian_against_numerical():
    """The analytic 3x6 Jacobian must match a central finite difference."""
    print("\nAnalytic Jacobian vs numerical derivative")
    worst = 0.0
    tested = 0
    while tested < 200:
        point_kf = np.array([RNG.uniform(-4, 4), RNG.uniform(-3, 3), RNG.uniform(1.0, 20.0)])
        R = exp_so3(RNG.normal(0, 0.3, 3))
        t = RNG.normal(0, 0.5, 3)
        if apply_pose(R, t, point_kf)[2] <= 0.5:
            continue  # behind or too near the camera; not a valid observation
        tested += 1
        z = project(apply_pose(R, t, point_kf)) + RNG.normal(0, 0.3, 3)

        _, J_analytic = residual_and_jacobian(point_kf, z, R, t)

        J_numeric = np.zeros((3, 6))
        h = 1e-7
        for k in range(6):
            step = np.zeros(6)
            step[k] = h
            Rp, tp = perturb(R, t, step)
            Rm, tm = perturb(R, t, -step)
            rp = project(apply_pose(Rp, tp, point_kf)) - z
            rm = project(apply_pose(Rm, tm, point_kf)) - z
            J_numeric[:, k] = (rp - rm) / (2 * h)

        scale = max(1.0, np.abs(J_analytic).max())
        worst = max(worst, np.abs(J_analytic - J_numeric).max() / scale)

    check("relative error < 1e-6", worst < 1e-6, f"worst = {worst:.3e}")


def test_closed_form_solve():
    """Umeyama must be exact on clean data, from any non-degenerate triple."""
    print("\nClosed-form (Umeyama) solve")
    worst_all, worst_minimal = 0.0, 0.0
    for _ in range(50):
        R_true = exp_so3(RNG.normal(0, 0.15, 3))
        t_true = RNG.normal(0, 0.3, 3)
        true_pts, kf_pts, disp = make_scene(60)
        kf, meas, _ = observe(true_pts, kf_pts, disp, R_true, t_true)
        if len(kf) < 10:
            continue
        cur = np.array([unproject(*m) for m in meas])

        R_e, t_e = rigid_align(kf, cur)
        worst_all = max(worst_all, np.linalg.norm(log_so3(R_true.T @ R_e)) + np.linalg.norm(t_true - t_e))

        idx = RNG.choice(len(kf), 3, replace=False)
        R_m, t_m = rigid_align(kf[idx], cur[idx])
        worst_minimal = max(worst_minimal,
                            np.linalg.norm(log_so3(R_true.T @ R_m)) + np.linalg.norm(t_true - t_m))

    check("exact over all points", worst_all < 1e-9, f"worst = {worst_all:.3e}")
    check("exact from a 3-point minimal set", worst_minimal < 1e-9, f"worst = {worst_minimal:.3e}")

    base = np.array([1.0, 0.5, 6.0])
    direction = np.array([1.0, 0.3, 0.2])
    collinear = np.array([base + s * direction for s in (-1.0, 0.0, 1.0)])
    centred = collinear - collinear.mean(axis=0)
    s = np.linalg.svd(centred.T @ centred, compute_uv=False)
    check("collinear triple detectable via singular values", s[1] < 1e-9 * s[0],
          f"s1/s0 = {s[1] / s[0]:.3e}")


def test_noise_free_recovery():
    """Noise-free data must recover the pose to numerical precision."""
    print("\nNoise-free pose recovery")
    worst_rot, worst_trans = 0.0, 0.0
    for _ in range(30):
        R_true = exp_so3(RNG.normal(0, 0.15, 3))
        t_true = RNG.normal(0, 0.3, 3)
        true_pts, kf_pts, disp = make_scene(150)
        kf, meas, dkf = observe(true_pts, kf_pts, disp, R_true, t_true)
        if len(kf) < 30:
            continue
        R0, t0 = rigid_align(kf, np.array([unproject(*m) for m in meas]))
        R_est, t_est, _, _ = gauss_newton(kf, meas, R0, t0, dkf, max_iters=20)
        worst_rot = max(worst_rot, np.linalg.norm(log_so3(R_true.T @ R_est)))
        worst_trans = max(worst_trans, np.linalg.norm(t_true - t_est))

    check("rotation error < 1e-9 rad", worst_rot < 1e-9, f"worst = {worst_rot:.3e}")
    check("translation error < 1e-9 m", worst_trans < 1e-9, f"worst = {worst_trans:.3e}")


def test_noisy_accuracy():
    """With realistic noise, errors should be small."""
    print(f"\nAccuracy under realistic noise ({PIXEL_SIGMA} px corners, {DISPARITY_SIGMA} px disparity)")
    rot_errors, trans_errors = [], []
    for _ in range(120):
        R_true = exp_so3(RNG.normal(0, 0.05, 3))
        t_true = RNG.normal(0, 0.15, 3)
        true_pts, kf_pts, disp = make_scene(300, triangulate_from_noisy=True)
        kf, meas, dkf = observe(true_pts, kf_pts, disp, R_true, t_true,
                                pixel_noise=PIXEL_SIGMA, disparity_noise=DISPARITY_SIGMA)
        if len(kf) < 50:
            continue
        R0, t0 = rigid_align(kf, np.array([unproject(*m) for m in meas]))
        R_est, t_est, _, _ = gauss_newton(kf, meas, R0, t0, dkf)
        rot_errors.append(np.linalg.norm(log_so3(R_true.T @ R_est)))
        trans_errors.append(np.linalg.norm(t_true - t_est))

    rot_errors, trans_errors = np.array(rot_errors), np.array(trans_errors)
    check("median rotation error < 0.5 mrad", np.median(rot_errors) < 5e-4,
          f"median = {np.median(rot_errors) * 1e3:.4f} mrad")
    check("median translation error < 5 mm", np.median(trans_errors) < 5e-3,
          f"median = {np.median(trans_errors) * 1e3:.3f} mm")
    check("95th pct translation error < 20 mm", np.percentile(trans_errors, 95) < 2e-2,
          f"p95 = {np.percentile(trans_errors, 95) * 1e3:.3f} mm")


def test_depth_uncertainty_weighting():
    """Propagating keyframe-point depth uncertainty must prevent the
    catastrophic-divergence tail on far-dominated scenes.

    This is the regression guard for the error-in-variables bug: treating the
    triangulated keyframe point as exact is fine on near scenes and badly wrong
    on far ones.
    """
    print("\nDepth-uncertainty weighting on a far scene (5-40 m)")
    naive, weighted = [], []
    for _ in range(120):
        R_true = exp_so3(RNG.normal(0, 0.03, 3))
        t_true = RNG.normal(0, 0.10, 3)
        true_pts, kf_pts, disp = make_scene(300, 5.0, 40.0, triangulate_from_noisy=True)
        kf, meas, dkf = observe(true_pts, kf_pts, disp, R_true, t_true,
                                pixel_noise=PIXEL_SIGMA, disparity_noise=DISPARITY_SIGMA)
        if len(kf) < 50:
            continue
        R0, t0 = rigid_align(kf, np.array([unproject(*m) for m in meas]))
        _, t_naive, _, _ = gauss_newton(kf, meas, R0, t0, disparities_kf=None)
        _, t_weighted, _, _ = gauss_newton(kf, meas, R0, t0, disparities_kf=dkf)
        naive.append(np.linalg.norm(t_true - t_naive))
        weighted.append(np.linalg.norm(t_true - t_weighted))

    naive, weighted = np.array(naive), np.array(weighted)
    print(f"     naive (point treated as exact): median {np.median(naive) * 1e3:8.2f} mm  "
          f"p95 {np.percentile(naive, 95) * 1e3:10.2f} mm")
    print(f"     weighted (uncertainty propagated): median {np.median(weighted) * 1e3:6.2f} mm  "
          f"p95 {np.percentile(weighted, 95) * 1e3:10.2f} mm")
    check("weighting improves the median", np.median(weighted) < np.median(naive),
          f"{100 * (np.median(naive) - np.median(weighted)) / np.median(naive):+.1f}%")
    check("weighting kills the divergence tail (p95 < 50 mm)",
          np.percentile(weighted, 95) < 0.05,
          f"p95 = {np.percentile(weighted, 95) * 1e3:.2f} mm")


def test_huber_rejects_outliers():
    """Huber weighting must contain gross outliers that survive RANSAC."""
    print("\nHuber robustness to surviving outliers")
    R_true = exp_so3(np.array([0.01, 0.03, -0.02]))
    t_true = np.array([0.05, 0.0, 0.2])
    true_pts, kf_pts, disp = make_scene(300, triangulate_from_noisy=True)
    kf, meas, dkf = observe(true_pts, kf_pts, disp, R_true, t_true,
                            pixel_noise=PIXEL_SIGMA, disparity_noise=DISPARITY_SIGMA)

    corrupted = meas.copy()
    n_out = max(1, int(0.10 * len(corrupted)))
    for i in RNG.choice(len(corrupted), n_out, replace=False):
        corrupted[i, :2] += RNG.normal(0, 25.0, 2)

    R0, t0 = rigid_align(kf, np.array([unproject(*m) for m in corrupted]))
    _, t_h, _, _ = gauss_newton(kf, corrupted, R0, t0, dkf, max_iters=20, huber=1.5)
    _, t_n, _, _ = gauss_newton(kf, corrupted, R0, t0, dkf, max_iters=20, huber=1e9)

    err_h, err_n = np.linalg.norm(t_true - t_h), np.linalg.norm(t_true - t_n)
    check("Huber beats plain least squares with 10% outliers", err_h < err_n,
          f"huber = {err_h * 1e3:.2f} mm vs plain = {err_n * 1e3:.2f} mm")
    check("Huber translation error stays under 20 mm", err_h < 2e-2, f"{err_h * 1e3:.2f} mm")


def test_covariance_is_calibrated():
    """Reported covariance must not be over-confident.

    Weights are absolute (1/variance), so the formal covariance is just
    inv(H). That formal value came out ~2.7x optimistic here, because the
    keyframe triangulation error is correlated with the current-frame
    measurement through the shared feature. VioParams::covarianceInflation
    corrects for it; this test pins the calibration.
    """
    print("\nCovariance calibration (predicted vs empirical scatter)")
    errors, variances = [], []
    for _ in range(150):
        R_true = exp_so3(RNG.normal(0, 0.05, 3))
        t_true = RNG.normal(0, 0.15, 3)
        true_pts, kf_pts, disp = make_scene(300, triangulate_from_noisy=True)
        kf, meas, dkf = observe(true_pts, kf_pts, disp, R_true, t_true,
                                pixel_noise=PIXEL_SIGMA, disparity_noise=DISPARITY_SIGMA)
        if len(kf) < 50:
            continue
        R0, t0 = rigid_align(kf, np.array([unproject(*m) for m in meas]))
        _, t_est, H, _ = gauss_newton(kf, meas, R0, t0, dkf)
        errors.append(t_true - t_est)
        variances.append(np.diag(np.linalg.inv(H))[:3])

    errors, variances = np.array(errors), np.array(variances)
    formal_ratio = errors.var(axis=0) / variances.mean(axis=0)
    inflated_ratio = formal_ratio / COVARIANCE_INFLATION
    print(f"     formal   empirical/predicted = {np.array2string(formal_ratio, precision=2)}")
    print(f"     inflated (x{COVARIANCE_INFLATION:.1f})            "
          f"= {np.array2string(inflated_ratio, precision=2)}")

    check("formal covariance is over-confident, justifying the inflation",
          bool(np.any(formal_ratio > 1.2)), f"max ratio = {formal_ratio.max():.2f}")
    check("inflated covariance is not over-confident (ratio <= 1.2)",
          bool(np.all(inflated_ratio <= 1.2)), f"max ratio = {inflated_ratio.max():.2f}")
    check("inflated covariance is not absurdly pessimistic (ratio >= 0.05)",
          bool(np.all(inflated_ratio >= 0.05)), f"min ratio = {inflated_ratio.min():.2f}")


def compose(a, b):
    """c = a * b, applying b first."""
    Ra, ta = a
    Rb, tb = b
    return Ra @ Rb, Ra @ tb + ta


def invert(pose):
    R, t = pose
    return R.T, -R.T @ t


def pose_perturb(pose, delta):
    R, t = pose
    dR = exp_so3(delta[3:])
    return dR @ R, dR @ t + delta[:3]


def pose_difference(perturbed, reference):
    """Recover delta such that perturbed == pose_perturb(reference, delta)."""
    Rp, tp = perturbed
    Rr, tr = reference
    dR = Rp @ Rr.T
    return np.concatenate([tp - dR @ tr, log_so3(dR)])


def test_pose_composition_jacobians():
    """Validate the hand-derived Jacobians in pose_integrator.cpp.

    The perturbation convention here is non-standard -- translation is
    decoupled rather than going through the SE(3) left Jacobian -- so the
    textbook SE(3) adjoint does not apply and these were derived by hand.
    """
    print("\nPose composition/inverse Jacobians (pose_integrator.cpp)")
    rng = np.random.default_rng(31337)
    h = 1e-6
    worst_inv, worst_rhs, worst_lhs = 0.0, 0.0, 0.0

    for _ in range(50):
        a = (exp_so3(rng.normal(0, 0.4, 3)), rng.normal(0, 1.0, 3))
        b = (exp_so3(rng.normal(0, 0.4, 3)), rng.normal(0, 1.0, 3))
        c = compose(a, b)

        # --- inverse ---
        Rt = a[0].T
        analytic_inv = np.zeros((6, 6))
        analytic_inv[:3, :3] = -Rt
        analytic_inv[:3, 3:] = skew(Rt @ a[1]) @ Rt
        analytic_inv[3:, 3:] = -Rt

        a_inv = invert(a)
        numeric_inv = np.zeros((6, 6))
        for k in range(6):
            step = np.zeros(6)
            step[k] = h
            plus = pose_difference(invert(pose_perturb(a, step)), a_inv)
            minus = pose_difference(invert(pose_perturb(a, -step)), a_inv)
            numeric_inv[:, k] = (plus - minus) / (2 * h)
        worst_inv = max(worst_inv, np.abs(analytic_inv - numeric_inv).max()
                        / max(1.0, np.abs(analytic_inv).max()))

        # --- compose, right-hand side ---
        analytic_rhs = np.zeros((6, 6))
        analytic_rhs[:3, :3] = a[0]
        analytic_rhs[:3, 3:] = skew(c[1]) @ a[0] - a[0] @ skew(b[1])
        analytic_rhs[3:, 3:] = a[0]

        numeric_rhs = np.zeros((6, 6))
        for k in range(6):
            step = np.zeros(6)
            step[k] = h
            plus = pose_difference(compose(a, pose_perturb(b, step)), c)
            minus = pose_difference(compose(a, pose_perturb(b, -step)), c)
            numeric_rhs[:, k] = (plus - minus) / (2 * h)
        worst_rhs = max(worst_rhs, np.abs(analytic_rhs - numeric_rhs).max()
                        / max(1.0, np.abs(analytic_rhs).max()))

        # --- compose, left-hand side: claimed to be exactly identity ---
        numeric_lhs = np.zeros((6, 6))
        for k in range(6):
            step = np.zeros(6)
            step[k] = h
            plus = pose_difference(compose(pose_perturb(a, step), b), c)
            minus = pose_difference(compose(pose_perturb(a, -step), b), c)
            numeric_lhs[:, k] = (plus - minus) / (2 * h)
        worst_lhs = max(worst_lhs, np.abs(numeric_lhs - np.eye(6)).max())

    check("inverse Jacobian matches finite differences", worst_inv < 1e-6, f"worst = {worst_inv:.3e}")
    check("compose RHS Jacobian matches finite differences", worst_rhs < 1e-6, f"worst = {worst_rhs:.3e}")
    check("compose LHS Jacobian is the identity", worst_lhs < 1e-6, f"worst = {worst_lhs:.3e}")


def test_wide_lens_geometry():
    """Sanity check the fisheye analysis of the OAK 4 D W optics, and state
    the honest depth precision."""
    print("\nOAK 4 D W wide-lens geometry (from published FoV spec)")
    hfov, vfov = np.radians(127.0), np.radians(79.5)

    f_pinhole_h = (WIDTH / 2) / np.tan(hfov / 2)
    f_pinhole_v = (HEIGHT / 2) / np.tan(vfov / 2)
    f_equi_h = (WIDTH / 2) / (hfov / 2)
    f_equi_v = (HEIGHT / 2) / (vfov / 2)

    print(f"     pinhole  f from HFoV = {f_pinhole_h:6.1f} px, from VFoV = {f_pinhole_v:6.1f} px")
    print(f"     equidist f from HFoV = {f_equi_h:6.1f} px, from VFoV = {f_equi_v:6.1f} px")

    check("pinhole model is inconsistent across axes (>20% disagreement)",
          abs(f_pinhole_h - f_pinhole_v) / f_pinhole_v > 0.2,
          f"{100 * abs(f_pinhole_h - f_pinhole_v) / f_pinhole_v:.0f}% apart")
    check("equidistant fisheye model is consistent (<5% disagreement)",
          abs(f_equi_h - f_equi_v) / f_equi_v < 0.05,
          f"{100 * abs(f_equi_h - f_equi_v) / f_equi_v:.1f}% apart")

    print(f"     depth sigma at f={FX:.0f} px, b={BASELINE * 1000:.0f} mm, "
          f"sigma_d={DISPARITY_SIGMA} px:")
    for z in (2.0, 5.0, 10.0, 20.0):
        sigma = z * z * DISPARITY_SIGMA / FX_B
        print(f"       {z:5.1f} m -> {sigma * 100:6.1f} cm  ({100 * sigma / z:4.1f}%)")

    # The honest figure. The plan originally quoted ~4 cm at 5 m by using the
    # 1/16 px subpixel step as if it were the noise; the real number is ~8x
    # worse. This check pins the corrected value so it cannot drift back.
    sigma_5m = 25.0 * DISPARITY_SIGMA / FX_B
    check("depth sigma at 5 m is ~31 cm (not the 4 cm the plan first claimed)",
          0.25 < sigma_5m < 0.40, f"{sigma_5m * 100:.1f} cm")

    # Depth beyond which relative depth error exceeds 10%.
    z_10pct = 0.10 * FX_B / DISPARITY_SIGMA
    print(f"     relative depth error exceeds 10% beyond {z_10pct:.1f} m "
          f"(disparity {FX_B / z_10pct:.1f} px)")


def main():
    print("=" * 74)
    print("OAK 4 D stereo VO -- estimator maths validation")
    print("=" * 74)

    test_jacobian_against_numerical()
    test_closed_form_solve()
    test_noise_free_recovery()
    test_noisy_accuracy()
    test_depth_uncertainty_weighting()
    test_huber_rejects_outliers()
    test_covariance_is_calibrated()
    test_pose_composition_jacobians()
    test_wide_lens_geometry()

    print("\n" + "=" * 74)
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}): " + ", ".join(FAILURES))
        return 1
    print("All checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
