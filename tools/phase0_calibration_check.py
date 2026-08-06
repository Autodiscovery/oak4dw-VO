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


def _epipolar_error(left_img, right_img) -> tuple[float, float, int, float]:
    """Median and 95th-pct |dv| between matched corners in a rectified pair.

    In a correctly rectified pair this should be a small fraction of a pixel.
    A large value means rectification is not doing its job and every depth is
    suspect.

    Matching left against right with Lucas-Kanade is the weak link: the two views
    differ by more than a small flow, so plain LK produces a substantial minority
    of confident nonsense. An earlier version of this function reported those
    mismatches as epipolar error, which made rectification look far worse than the
    evidence supported.

    Two corrections. Bad matches are rejected by forward-backward consistency --
    match right, then match back to left, and require the round trip to return to
    where it started -- which is a geometry-agnostic test. And the acceptance rate
    is returned, because a low rate means the remaining statistics rest on a small
    biased sample and should not be trusted on their own.

    Deliberately NOT filtered on |dv| itself: excluding large vertical
    disagreements would be assuming the answer and would bias the estimate
    downward.
    """
    if cv2 is None:
        return float("nan"), float("nan"), 0, float("nan")

    corners = cv2.goodFeaturesToTrack(left_img, maxCorners=600, qualityLevel=0.01, minDistance=12)
    if corners is None or len(corners) < 20:
        return float("nan"), float("nan"), 0, float("nan")

    lk = dict(winSize=(21, 21), maxLevel=4,
              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(left_img, right_img, corners, None, **lk)
    back, status_back, _ = cv2.calcOpticalFlowPyrLK(right_img, left_img, tracked, None, **lk)

    ok = (status.ravel() == 1) & (status_back.ravel() == 1)
    round_trip = np.linalg.norm((back - corners).reshape(-1, 2), axis=1)
    ok &= round_trip < 1.0

    src = corners.reshape(-1, 2)
    dst = tracked.reshape(-1, 2)
    # A match must lie to the left in the right image; that is what disparity is.
    ok &= (src[:, 0] - dst[:, 0]) > 0.5

    acceptance = float(ok.sum()) / float(len(corners))
    if ok.sum() < 20:
        return float("nan"), float("nan"), int(ok.sum()), acceptance

    dv = np.abs(src[ok, 1] - dst[ok, 1])
    return float(np.median(dv)), float(np.percentile(dv, 95)), int(ok.sum()), acceptance


def cmd_rectify(args) -> int:
    alphas = [None] if args.alpha is None else args.alpha
    print(f"{'alpha':>7} {'valid disp %':>13} {'valid bbox %':>13} "
          f"{'epi med px':>11} {'epi p95 px':>11} {'matches':>8} {'accept %':>9}")
    print("-" * 82)

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

            coverage, bbox, epi_med, epi_p95, matches, accept = [], [], [], [], [], []
            for _ in range(args.frames):
                left = queues["rectifiedLeft"].get()
                right = queues["rectifiedRight"].get()
                disp = queues["disparity"].get()

                left_img = left.getCvFrame() if cv2 is not None else None
                right_img = right.getCvFrame() if cv2 is not None else None

                d = np.asarray(disp.getFrame())
                coverage.append(100.0 * np.count_nonzero(d) / d.size)

                # Extent of the region the matcher solved anything in, as a
                # fraction of the frame. An earlier version reported the non-black
                # fraction of the rectified image as "FoV kept", which returned
                # exactly 100.0% for every alpha -- rectification here does not
                # zero-pad, so that metric could only ever say 100%. This at least
                # measures something that varies.
                rows = np.any(d > 0, axis=1)
                cols = np.any(d > 0, axis=0)
                if rows.any() and cols.any():
                    height = np.flatnonzero(rows)[-1] - np.flatnonzero(rows)[0] + 1
                    width = np.flatnonzero(cols)[-1] - np.flatnonzero(cols)[0] + 1
                    bbox.append(100.0 * (height * width) / d.size)

                if left_img is not None:
                    m, p, n, a = _epipolar_error(left_img, right_img)
                    accept.append(100.0 * a)
                    if not np.isnan(m):
                        epi_med.append(m)
                        epi_p95.append(p)
                        matches.append(n)

            mean = lambda xs: np.mean(xs) if len(xs) else float("nan")  # noqa: E731
            label = "default" if alpha is None else f"{alpha:.2f}"
            print(f"{label:>7} {mean(coverage):>13.1f} {mean(bbox):>13.1f} "
                  f"{mean(epi_med):>11.3f} {mean(epi_p95):>11.3f} "
                  f"{(int(mean(matches)) if matches else 0):>8} {mean(accept):>9.1f}")

    print("\nReading these numbers:")
    print("  valid disp %  how much of the image the block matcher solved. Below ~20% on a")
    print("                textured scene suggests the stereo pair is not matching well.")
    print("  valid bbox %  extent of the solved region. Much larger than 'valid disp %' means")
    print("                the solved pixels are scattered rather than a coherent area.")
    print("  epi med px    should be well under 0.5 px. Above ~1 px, rectification is the")
    print("                problem and no estimator tuning will fix the resulting scale error.")
    print("  accept %      fraction of corners that survived the forward-backward match check.")
    print("                Below ~50%, treat the epipolar figures as unreliable: too few")
    print("                trustworthy matches to characterise the rectification.")
    print("\nIf every alpha gives identical numbers, setAlphaScaling is having no effect on")
    print("this platform, and the choice of alpha is not the lever to pull.")
    return 0


# ---------------------------------------------------------------------------
# noise
# ---------------------------------------------------------------------------
def cmd_noise(args) -> int:
    """Measure disparity noise, and separately how pinhole-like the rectified pair is.

    Two different quantities, which an earlier version of this command conflated
    into one misleading figure.

    TEMPORAL NOISE is the per-pixel standard deviation of disparity across frames
    with the camera and scene held still. That is sigma_d, the quantity the
    estimator's weighting and covariance actually need. It assumes nothing about
    the scene or the camera model, which is what makes it trustworthy.

    PLANE-FIT RESIDUAL fits `disparity = a*x + b*y + c` over a central ROI. For a
    flat wall viewed through an ideal PINHOLE, disparity is exactly linear in image
    coordinates, so the residual would be pure noise. Through a lens that is not
    behaving as a rectified pinhole it is not linear at all, and the residual is
    dominated by model error instead.

    The earlier version reported only the plane-fit residual and called it noise.
    On this camera that produced 12.7 px, which would be an absurd noise figure --
    it was measuring the failure of the pinhole assumption, not the sensor. Both
    are reported now, and the gap between them is itself the diagnostic.
    """
    print("Two measurements with DIFFERENT scene requirements:")
    print("  sigma_d      needs texture. Any static, cluttered, well-lit scene will do --")
    print("               a bookshelf, a desk, a room. Planarity is irrelevant.")
    print("  model error  needs texture AND a flat surface filling the frame: a brick wall,")
    print("               a large poster, newspaper taped up. A plain painted wall has no")
    print("               texture, gives almost no valid disparity, and makes both")
    print("               measurements worthless.")
    print("\nHold the camera STILL either way -- the temporal measurement depends on it.")
    print(f"Sampling {args.frames} frames in {args.settle:.0f} s...\n")

    pipeline, queues = _build_rectify_pipeline(args.alpha[0] if args.alpha else None, args.fps)
    subpixel_scale = 1.0 / 32.0  # matches setSubpixelFractionalBits(5)

    with pipeline:
        pipeline.start()
        deadline = time.time() + args.settle
        while time.time() < deadline:
            queues["disparity"].tryGet()
            time.sleep(0.05)

        stack, residual_stds, plane_disparities = [], [], []
        for _ in range(args.frames):
            d = np.asarray(queues["disparity"].get().getFrame()).astype(np.float64) * subpixel_scale
            stack.append(d)

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
            lo, hi = np.percentile(residual, [2, 98])
            trimmed = residual[(residual >= lo) & (residual <= hi)]
            residual_stds.append(float(np.std(trimmed)))
            plane_disparities.append(float(np.median(roi[valid])))

    if not stack:
        print("No disparity frames received.")
        return 1

    # --- Temporal noise: this is sigma_d ---------------------------------
    volume = np.stack(stack)
    always_valid = np.all(volume > 0.5, axis=0)
    validity = float(always_valid.mean())
    print(f"  frames used:                    {len(stack)}")
    print(f"  pixels valid in every frame:    {100.0 * validity:.1f}%")

    if always_valid.sum() < 500:
        print("\n  Too few consistently valid pixels for any measurement. The scene almost")
        print("  certainly lacks texture -- point at something visually busy.")
        return 1

    temporal = volume.std(axis=0)[always_valid]
    sigma = float(np.median(temporal))
    print(f"  median disparity:               {float(np.median(volume[:, always_valid])):.2f} px")
    print(f"\n  TEMPORAL noise (sigma_d):        {sigma:.3f} px"
          f"   [p10 {np.percentile(temporal, 10):.3f}, p90 {np.percentile(temporal, 90):.3f}]")
    print(f"  subpixel quantisation step:     {subpixel_scale:.4f} px")

    if validity < 0.25:
        print(f"\n  CAUTION: only {100.0 * validity:.1f}% of pixels were valid throughout, so sigma_d is")
        print("  measured on whichever pixels the matcher found easiest -- a biased sample that")
        print("  will read optimistically. Treat it as a lower bound and re-measure on a")
        print("  more textured scene.")

    # --- Plane-fit residual: model error, not noise ------------------------
    if residual_stds:
        plane = float(np.median(residual_stds))
        print(f"  PLANE-FIT residual:              {plane:.3f} px  (over {float(np.median(plane_disparities)):.1f} px disparity)")
        if plane > 4.0 * max(sigma, 1e-6):
            print(f"\n  The plane-fit residual is {plane / max(sigma, 1e-6):.0f}x the temporal noise. On a flat wall")
            print("  that gap is model error, not sensor noise: disparity is only linear across a")
            print("  plane for an ideal rectified pinhole, so a large residual says the rectified")
            print("  pair is not behaving as one. That is NOT sigma_d and must not be used as one.")

    # --- Where does the pinhole model break down? --------------------------
    #
    # Fit the plane on the CENTRAL region, where distortion is smallest and the
    # model has the best chance of holding, then evaluate the residual across the
    # whole frame in rings out from the centre.
    #
    # A flat profile means the residual is texture and mismatching. A profile that
    # grows with radius is uncorrected lens distortion surviving rectification --
    # which on a 150 deg fisheye is exactly what one would expect, and it says
    # where to stop trusting features (vio.i_mask_border_fraction).
    reference = np.median(volume, axis=0)
    h, w = reference.shape
    ys, xs = np.mgrid[0:h, 0:w]
    y0, y1, x0, x1 = int(0.4 * h), int(0.6 * h), int(0.4 * w), int(0.6 * w)

    core = np.zeros_like(reference, dtype=bool)
    core[y0:y1, x0:x1] = True
    core &= reference > 0.5

    if validity < 0.40:
        print(f"\n  SKIPPING the model-error analysis: only {100.0 * validity:.1f}% of pixels are valid, and it")
        print("  needs a flat surface actually filling the frame. With coverage this sparse the")
        print("  surviving pixels are scattered fragments at assorted depths, not a plane, and")
        print("  fitting one to them produces confident-looking nonsense.")
        print("  Re-run against a brick wall, a large poster, or newspaper taped to a wall.")
    elif core.sum() > 500:
        A = np.column_stack([xs[core], ys[core], np.ones(core.sum())])
        coeffs, *_ = np.linalg.lstsq(A, reference[core], rcond=None)

        valid = reference > 0.5
        predicted = coeffs[0] * xs + coeffs[1] * ys + coeffs[2]
        residual = np.abs(reference - predicted)
        radius = np.hypot(xs - w / 2.0, ys - h / 2.0)
        max_radius = np.hypot(w / 2.0, h / 2.0)

        # Does the plane fit its OWN fit region? If not, the scene is not planar
        # and nothing extrapolated from it means anything. This is the check that
        # would have caught an earlier run reporting a 5.3 px residual inside the
        # fit region and a wildly non-monotonic profile outside it.
        core_residual = float(np.median(residual[core]))
        print(f"\n  Plane fit quality inside its own fit region: {core_residual:.2f} px "
              f"({core_residual / max(sigma, 1e-6):.1f}x sigma_d)")
        if core_residual > 3.0 * max(sigma, 1e-6):
            print("  The plane does not fit the region it was fitted to, so the surface is not")
            print("  planar (or the disparity there is unreliable). Not reporting a radial")
            print("  profile from it -- it could only mislead. Find a genuinely flat, textured")
            print("  surface and re-run.")
            print(f"\n  Set in params/vio.yaml:   vio.i_disparity_sigma_px: {sigma:.2f}")
            return 0

        print("\n  Deviation from the fitted plane, by distance from image centre")
        print("  (plane fitted on the central 20% only, then extrapolated outward):")
        print(f"    {'radius':>14}  {'median |resid|':>14}  {'as % of disparity':>18}")
        edges = np.linspace(0.0, max_radius, 7)
        profile = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            ring = valid & (radius >= lo) & (radius < hi)
            if ring.sum() < 200:
                continue
            med = float(np.median(residual[ring]))
            disp = float(np.median(reference[ring]))
            frac = 100.0 * med / disp if disp > 0.5 else float("nan")
            profile.append(med)
            print(f"    {100 * lo / max_radius:5.0f}-{100 * hi / max_radius:3.0f}% "
                  f"{med:>14.2f}  {frac:>17.1f}%")

        # Distortion rises monotonically with radius. Anything else is not
        # distortion, whatever else it may be.
        if len(profile) >= 4:
            rising = sum(b > a for a, b in zip(profile, profile[1:]))
            if rising >= len(profile) - 2:
                print("\n    Profile rises with radius: consistent with uncorrected lens distortion")
                print("    surviving rectification. Set vio.i_mask_border_fraction to exclude the")
                print("    radius where it passes a few percent.")
            else:
                print("\n    Profile is NOT monotonic in radius, so this is not a distortion")
                print("    signature -- distortion grows outward without exception. More likely")
                print("    uneven texture or depth structure in the scene. Do not set a border")
                print("    mask from this.")

    print(f"\n  Set in params/vio.yaml:   vio.i_disparity_sigma_px: {sigma:.2f}")
    print("  (the temporal figure -- the plane-fit residual above is a different quantity)")
    return 0


def main() -> int:
    # Options live on the SUBPARSERS, not the top-level parser, so they can be
    # given after the subcommand -- `rectify --device <ip>` -- which is both the
    # natural order and what this module's docstring shows. With them on the top
    # level, argparse demands `--device <ip> rectify` and rejects anything else.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default=None, help="Device IP. Omit to auto-discover.")
    common.add_argument("--fps", type=float, default=20.0)
    common.add_argument("--frames", type=int, default=30)
    common.add_argument("--settle", type=float, default=3.0, help="Seconds to let 3A settle.")
    common.add_argument("--alpha", type=float, nargs="*", default=None,
                        help="Alpha scaling values to sweep, e.g. --alpha 0 0.25 0.5 0.75 1")

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("calib", parents=[common], help="Dump calibration and infer the projection model.")
    sub.add_parser("rectify", parents=[common], help="Sweep alpha scaling; measure FoV and epipolar error.")
    sub.add_parser("noise", parents=[common], help="Measure disparity noise against a flat wall.")

    args = parser.parse_args()
    return {"calib": cmd_calib, "rectify": cmd_rectify, "noise": cmd_noise}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
