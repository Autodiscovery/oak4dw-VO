#!/usr/bin/env python3
"""Phase 0: measure what the OAK 4 D W optics and stereo block actually give us.

Nothing else in this project is worth tuning until these numbers exist. Three
things come out of it, each feeding a parameter in params/vio.yaml:

  calib   -- distortion model and intrinsics. Confirms or refutes the
             assumption that the lens is an equidistant fisheye, and shows what
             a pinhole rectification would cost in field of view.
  rectify -- rectified field of view and epipolar error across a sweep of alpha
             scaling values. Sets vio.i_alpha_scaling and
             vio.i_mask_border_fraction.
  noise   -- disparity noise against a flat wall. Sets
             vio.i_disparity_sigma_px, which drives both the depth-uncertainty
             weighting and the published covariance. This is the one people
             skip and then wonder why the covariance is wrong.

Runs on the HOST against a networked device, not inside the app container.

    pip install depthai opencv-python numpy
    python tools/phase0_calibration_check.py calib   --device 192.168.1.42
    python tools/phase0_calibration_check.py rectify --device 192.168.1.42
    python tools/phase0_calibration_check.py noise   --device 192.168.1.42
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    import depthai as dai
except ImportError:  # pragma: no cover - host tooling
    print("depthai not installed.  pip install depthai", file=sys.stderr)
    raise

try:
    import cv2
except ImportError:  # pragma: no cover - host tooling
    cv2 = None

LEFT_SOCKET = dai.CameraBoardSocket.CAM_B
RIGHT_SOCKET = dai.CameraBoardSocket.CAM_C
WIDTH, HEIGHT = 1280, 800


def _connect(device_ip: str | None):
    if device_ip:
        return dai.Device(dai.DeviceInfo(device_ip))
    return dai.Device()


# ---------------------------------------------------------------------------
# calib
# ---------------------------------------------------------------------------
def cmd_calib(args) -> int:
    with _connect(args.device) as device:
        print(f"Device: {device.getDeviceName()}  platform={device.getPlatformAsString()}")
        calib = device.readCalibration()

        baseline_cm = calib.getBaselineDistance()
        print(f"\nBaseline: {baseline_cm:.3f} cm ({baseline_cm / 100:.5f} m)")

        for label, socket in (("LEFT  (CAM_B)", LEFT_SOCKET), ("RIGHT (CAM_C)", RIGHT_SOCKET)):
            print(f"\n=== {label} ===")
            try:
                model = calib.getDistortionModel(socket)
                print(f"  distortion model: {model}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  distortion model: unavailable ({exc})")

            coeffs = calib.getDistortionCoefficients(socket)
            print(f"  distortion coeffs ({len(coeffs)}): "
                  + ", ".join(f"{c:+.5f}" for c in coeffs))

            k = np.array(calib.getCameraIntrinsics(socket, WIDTH, HEIGHT))
            fx, fy, cx, cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
            print(f"  intrinsics @ {WIDTH}x{HEIGHT}: fx={fx:.2f} fy={fy:.2f} cx={cx:.2f} cy={cy:.2f}")

            try:
                hfov = calib.getFov(socket)
                print(f"  reported FoV: {hfov:.1f} deg")
            except Exception:  # noqa: BLE001
                pass

            # Which projection model do these intrinsics actually imply?
            # A pinhole lens gives the same implied focal length from both
            # axes. A fisheye does not.
            hfov_pinhole = 2 * np.degrees(np.arctan((WIDTH / 2) / fx))
            vfov_pinhole = 2 * np.degrees(np.arctan((HEIGHT / 2) / fy))
            hfov_equi = 2 * np.degrees((WIDTH / 2) / fx)
            vfov_equi = 2 * np.degrees((HEIGHT / 2) / fy)
            print(f"    if pinhole:     HFoV {hfov_pinhole:6.1f} deg, VFoV {vfov_pinhole:6.1f} deg")
            print(f"    if equidistant: HFoV {hfov_equi:6.1f} deg, VFoV {vfov_equi:6.1f} deg")
            print(f"    fx/fy ratio: {fx / fy:.4f}  (far from 1.0 suggests strong distortion)")

        # Depth precision implied by this calibration.
        k_left = np.array(calib.getCameraIntrinsics(LEFT_SOCKET, WIDTH, HEIGHT))
        fx = k_left[0, 0]
        fx_b = fx * (baseline_cm / 100.0)
        print(f"\nDepth precision (fx*b = {fx_b:.2f} m*px):")
        for sigma_d in (0.25, 0.5, 1.0):
            row = "  ".join(f"{z:>4.0f}m: {z * z * sigma_d / fx_b * 100:6.1f}cm" for z in (2, 5, 10, 20))
            print(f"  sigma_d={sigma_d:.2f} px ->  {row}")
        print("\n  Use the 'noise' subcommand to find which sigma_d is real for this device;")
        print("  do NOT assume the 1/32 px subpixel step -- that is quantisation, not noise.")

    return 0


# ---------------------------------------------------------------------------
# rectify
# ---------------------------------------------------------------------------
def _build_rectify_pipeline(alpha: float | None, fps: float):
    pipeline = dai.Pipeline()
    left = pipeline.create(dai.node.Camera).build(LEFT_SOCKET)
    right = pipeline.create(dai.node.Camera).build(RIGHT_SOCKET)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
    stereo.setRectification(True)
    stereo.setLeftRightCheck(True)
    stereo.setSubpixel(True)
    stereo.setSubpixelFractionalBits(5)
    stereo.setDepthAlign(LEFT_SOCKET)
    if alpha is not None and alpha >= 0.0:
        stereo.setAlphaScaling(alpha)

    left.requestOutput((WIDTH, HEIGHT), fps=fps).link(stereo.left)
    right.requestOutput((WIDTH, HEIGHT), fps=fps).link(stereo.right)

    queues = {
        "rectifiedLeft": stereo.rectifiedLeft.createOutputQueue(4, False),
        "rectifiedRight": stereo.rectifiedRight.createOutputQueue(4, False),
        "disparity": stereo.disparity.createOutputQueue(4, False),
    }
    return pipeline, queues


def _epipolar_error(left_img, right_img) -> tuple[float, float, int]:
    """Median and 95th-pct |dv| between matched corners in a rectified pair.

    In a correctly rectified pair this should be a small fraction of a pixel.
    A large value means rectification is not doing its job and every depth is
    suspect.
    """
    if cv2 is None:
        return float("nan"), float("nan"), 0

    corners = cv2.goodFeaturesToTrack(left_img, maxCorners=600, qualityLevel=0.01, minDistance=12)
    if corners is None or len(corners) < 20:
        return float("nan"), float("nan"), 0

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        left_img, right_img, corners, None,
        winSize=(21, 21), maxLevel=4,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    ok = status.ravel() == 1
    if ok.sum() < 20:
        return float("nan"), float("nan"), 0

    src = corners[ok].reshape(-1, 2)
    dst = tracked[ok].reshape(-1, 2)
    disparity = src[:, 0] - dst[:, 0]
    sane = disparity > 0.5  # matches must lie to the left in the right image
    if sane.sum() < 20:
        return float("nan"), float("nan"), 0

    dv = np.abs(src[sane, 1] - dst[sane, 1])
    return float(np.median(dv)), float(np.percentile(dv, 95)), int(sane.sum())


def cmd_rectify(args) -> int:
    alphas = [None] if args.alpha is None else args.alpha
    print(f"{'alpha':>7} {'valid disp %':>13} {'FoV kept %':>11} "
          f"{'epi med px':>11} {'epi p95 px':>11} {'matches':>8}")
    print("-" * 68)

    for alpha in alphas:
        pipeline, queues = _build_rectify_pipeline(alpha, args.fps)
        with pipeline:
            pipeline.start()
            # Discard the first frames: 3A is still settling.
            deadline = time.time() + args.settle
            while time.time() < deadline:
                for q in queues.values():
                    q.tryGet()
                time.sleep(0.05)

            coverage, kept, epi_med, epi_p95, matches = [], [], [], [], []
            for _ in range(args.frames):
                left = queues["rectifiedLeft"].get()
                right = queues["rectifiedRight"].get()
                disp = queues["disparity"].get()

                left_img = left.getCvFrame() if cv2 is not None else None
                right_img = right.getCvFrame() if cv2 is not None else None

                d = np.asarray(disp.getFrame())
                coverage.append(100.0 * np.count_nonzero(d) / d.size)

                # Rectification pads invalid regions with zeros in the
                # rectified image; the non-black fraction approximates how
                # much of the sensor's field of view survived.
                if left_img is not None:
                    kept.append(100.0 * np.count_nonzero(left_img) / left_img.size)
                    m, p, n = _epipolar_error(left_img, right_img)
                    if not np.isnan(m):
                        epi_med.append(m)
                        epi_p95.append(p)
                        matches.append(n)

            label = "default" if alpha is None else f"{alpha:.2f}"
            print(f"{label:>7} {np.mean(coverage):>13.1f} "
                  f"{(np.mean(kept) if kept else float('nan')):>11.1f} "
                  f"{(np.mean(epi_med) if epi_med else float('nan')):>11.3f} "
                  f"{(np.mean(epi_p95) if epi_p95 else float('nan')):>11.3f} "
                  f"{(int(np.mean(matches)) if matches else 0):>8}")

    print("\nReading these numbers:")
    print("  valid disp %  higher is better; how much of the image the block matcher solved.")
    print("  FoV kept %    higher keeps more of the wide lens; falls as alpha crops harder.")
    print("  epi med px    should be well under 0.5 px. Above ~1 px, rectification is the")
    print("                problem and no amount of estimator tuning will fix the drift.")
    print("\nPick the alpha with acceptable epipolar error and the most retained FoV,")
    print("then put it in params/vio.yaml as vio.i_alpha_scaling.")
    return 0


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------
def cmd_noise(args) -> int:
    """Measure disparity noise against a flat surface.

    Point the camera squarely at a textured flat wall filling the frame, 2-4 m
    away. We fit a plane to the disparity in a central ROI -- a plane in 3D is
    also a plane in disparity space, which is what makes this work -- and the
    residual scatter is the disparity noise.
    """
    print("Point the camera at a flat, textured wall filling the frame (2-4 m).")
    print(f"Sampling {args.frames} frames in {args.settle:.0f} s...\n")

    pipeline, queues = _build_rectify_pipeline(args.alpha[0] if args.alpha else None, args.fps)
    subpixel_scale = 1.0 / 32.0  # matches setSubpixelFractionalBits(5)

    with pipeline:
        pipeline.start()
        deadline = time.time() + args.settle
        while time.time() < deadline:
            queues["disparity"].tryGet()
            time.sleep(0.05)

        residual_stds, plane_disparities = [], []
        for _ in range(args.frames):
            d = np.asarray(queues["disparity"].get().getFrame()).astype(np.float64) * subpixel_scale

            h, w = d.shape
            y0, y1 = int(0.35 * h), int(0.65 * h)
            x0, x1 = int(0.35 * w), int(0.65 * w)
            roi = d[y0:y1, x0:x1]

            ys, xs = np.mgrid[y0:y1, x0:x1]
            valid = roi > 0.5
            if valid.sum() < 500:
                continue

            # Least-squares plane  disparity = a*x + b*y + c
            A = np.column_stack([xs[valid], ys[valid], np.ones(valid.sum())])
            coeffs, *_ = np.linalg.lstsq(A, roi[valid], rcond=None)
            residual = roi[valid] - A @ coeffs

            # Trim the tail before taking the std: mismatches are outliers, not
            # noise, and would inflate the estimate.
            lo, hi = np.percentile(residual, [2, 98])
            trimmed = residual[(residual >= lo) & (residual <= hi)]
            residual_stds.append(float(np.std(trimmed)))
            plane_disparities.append(float(np.median(roi[valid])))

    if not residual_stds:
        print("Not enough valid disparity in the ROI. Is the wall textured and lit?")
        return 1

    sigma = float(np.median(residual_stds))
    disparity = float(np.median(plane_disparities))
    print(f"  frames used:              {len(residual_stds)}")
    print(f"  median disparity in ROI:  {disparity:.2f} px")
    print(f"  disparity noise (sigma):  {sigma:.3f} px")
    print(f"  per-frame spread:         {np.min(residual_stds):.3f} - {np.max(residual_stds):.3f} px")
    print(f"\n  Subpixel quantisation step is {subpixel_scale:.4f} px -- note how far")
    print(f"  above it the real noise sits. That gap is exactly the mistake this")
    print(f"  measurement exists to prevent.")
    print(f"\n  Set in params/vio.yaml:   vio.i_disparity_sigma_px: {sigma:.2f}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="Device IP. Omit to auto-discover.")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--settle", type=float, default=3.0, help="Seconds to let 3A settle.")
    parser.add_argument("--alpha", type=float, nargs="*", default=None,
                        help="Alpha scaling values to sweep, e.g. --alpha 0 0.25 0.5 0.75 1")

    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calib", help="Dump calibration and infer the projection model.")
    sub.add_parser("rectify", help="Sweep alpha scaling; measure FoV and epipolar error.")
    sub.add_parser("noise", help="Measure disparity noise against a flat wall.")

    args = parser.parse_args()
    return {"calib": cmd_calib, "rectify": cmd_rectify, "noise": cmd_noise}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
