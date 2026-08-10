#!/usr/bin/env python3
"""Minimal FeatureTracker reproducer for RVC4, on the supported conversion path.

    Camera (NV12) -> ImageManip (GRAY8) -> FeatureTracker

This is deliberately the smallest thing that can demonstrate the behaviour: one
camera, one manip, one tracker, no stereo, no extra nodes. Suitable for pasting
straight into a bug report.

Before running, make sure the device is up to date and capture the output:

    oakctl device update
    oakctl device info

Then:

    python3 tools/repro_feature_tracker_minimal.py --device <ip>            # all sockets
    python3 tools/repro_feature_tracker_minimal.py --device <ip> --socket CAM_B

It sweeps CAM_A, CAM_B and CAM_C by default, because "works on the colour camera
but not the mono pair" would be a much more specific finding than "does not work",
and it costs nothing to check.

For each socket it reports the pixel format and pixel statistics of the frame
actually entering the tracker, so a zero-feature result cannot be confused with a
blank or wrongly-formatted input.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

try:
    import depthai as dai
except ImportError:
    sys.exit("depthai not installed.  pip install depthai")

SOCKETS = {
    "CAM_A": dai.CameraBoardSocket.CAM_A,
    "CAM_B": dai.CameraBoardSocket.CAM_B,
    "CAM_C": dai.CameraBoardSocket.CAM_C,
}


def connect(ip: str | None, attempts: int = 4):
    for attempt in range(attempts):
        try:
            return dai.Device(dai.DeviceInfo(ip)) if ip else dai.Device()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            print(f"  (connect failed: {exc}; retrying in 5 s)")
            time.sleep(5.0)
    raise RuntimeError("unreachable")


def run(ip, socket_name, socket, size, fps, seconds, target_features, settle,
        with_tracker=True, use_manip=True, resize_mode=None, label=None, threshold=20000):
    if label is None:
        label = "WITH FeatureTracker" if with_tracker else "CONTROL: no FeatureTracker"
    print(f"\n--- {socket_name} {size[0]}x{size[1]}: {label} ---")
    device = connect(ip)

    feature_counts: list[int] = []
    frame_type = None
    pixel_stats = None
    error = ""
    frames_seen = 0
    first_frame_at = None
    silent_at = None

    try:
        with dai.Pipeline(device) as pipeline:
            # 1. Camera. NV12 when a manip will convert it, GRAY8 when the
            #    tracker is fed directly -- the two candidate paths.
            cam = pipeline.create(dai.node.Camera).build(socket)
            requested_type = dai.ImgFrame.Type.NV12 if use_manip else dai.ImgFrame.Type.GRAY8

            # Pass only the arguments this variant is testing. Supplying
            # resizeMode or fps when the working example omits them is exactly the
            # kind of difference that could matter, so do not add them silently.
            kwargs = {}
            if resize_mode is not None:
                kwargs["resizeMode"] = resize_mode
            if fps is not None:
                kwargs["fps"] = fps
            cam_out = cam.requestOutput(size, type=requested_type, **kwargs)

            # 2. ImageManip, NV12 -> GRAY8, only on the manip path.
            if use_manip:
                manip = pipeline.create(dai.node.ImageManip)
                manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
                # Generously sized: the manip pads to hardware alignment, and an
                # exact width*height budget makes it skip every frame at sizes
                # whose height is not a multiple of 32.
                manip.setMaxOutputFrameSize(size[0] * size[1] * 3)
                cam_out.link(manip.inputImage)
                tracker_source = manip.out
            else:
                tracker_source = cam_out

            # 3. FeatureTracker -- omitted in the control run, which is otherwise
            #    byte-for-byte identical. If the device survives without it and
            #    crashes with it, the tracker is implicated; if it crashes either
            #    way, the fault is upstream and this is not a tracker bug at all.
            features_q = None
            if with_tracker:
                tracker = pipeline.create(dai.node.FeatureTracker)

                # Configure the corner detector EXPLICITLY, threshold included.
                # The automatic threshold (initialValue 0) returns empty feature
                # messages on RVC4 -- the node runs, finds nothing, and reports
                # nothing, with no error to say why.
                #
                # Scale matters here and is easy to get wrong: the Harris response
                # threshold is a large integer, order 10^4. An earlier version of
                # this probe tested an "explicit threshold" of 0.01, which is six
                # orders of magnitude below a working value, and recorded the
                # resulting failure as evidence that explicit thresholds did not
                # help. They do; that number was simply nonsense.
                corner = dai.FeatureTrackerConfig.CornerDetector()
                corner.numMaxFeatures = target_features
                corner.numTargetFeatures = target_features
                thresholds = dai.FeatureTrackerConfig.CornerDetector.Thresholds()
                thresholds.initialValue = threshold
                corner.thresholds = thresholds
                tracker.initialConfig.setCornerDetector(corner)

                tracker_source.link(tracker.inputImage)
                features_q = tracker.outputFeatures.createOutputQueue(8, False)

            tracker_input_q = tracker_source.createOutputQueue(4, False)

            pipeline.start()
            started = time.time()

            # Record from the very first frame rather than discarding a settle
            # window. The device crashes within a couple of seconds here, so a
            # settle period meant capturing nothing at all -- and reporting "no
            # frames" as though the scene were at fault. Statistics come from the
            # LAST frame received, which is the most settled one available.
            #
            # pipeline.isRunning() is not a reliable crash indicator: it kept
            # returning True after the device had gone, because the X_LINK errors
            # surface on the queue threads rather than this one. Silence is the
            # signal instead.
            deadline = started + settle + seconds
            last_activity = started
            while time.time() < deadline:
                got_something = False

                msg = features_q.tryGet() if features_q is not None else None
                if msg is not None:
                    feature_counts.append(len(msg.trackedFeatures))
                    got_something = True

                frame = tracker_input_q.tryGet()
                if frame is not None:
                    frames_seen += 1
                    if first_frame_at is None:
                        first_frame_at = time.time() - started
                    frame_type = int(frame.getType())
                    array = np.asarray(frame.getFrame())
                    if array.size:
                        pixel_stats = (float(array.mean()), float(array.std()),
                                       int(array.min()), int(array.max()))
                    got_something = True

                now = time.time()
                if got_something:
                    last_activity = now
                elif frames_seen > 0 and now - last_activity > 2.0:
                    silent_at = now - started
                    break
                elif frames_seen == 0 and now - started > 6.0:
                    silent_at = now - started
                    break

                time.sleep(0.005)

            if silent_at is not None:
                error = f"stopped delivering after {silent_at:.1f} s (device crash)"
    except Exception as exc:  # noqa: BLE001 - the failure IS the result
        error = str(exc).splitlines()[0]
    finally:
        try:
            device.close()
        except Exception:  # noqa: BLE001
            pass

    gray8 = int(dai.ImgFrame.Type.GRAY8)
    print(f"  tracker input format: {frame_type}  (GRAY8 is {gray8})"
          + ("  <-- correct" if frame_type == gray8 else "  <-- NOT GRAY8"))
    if pixel_stats:
        mean, std, lo, hi = pixel_stats
        # Judge on the MEAN as well as the spread. An earlier version gated on std
        # alone and called mean 0.1 / std 2.2 "real content" -- that is a black
        # frame with a handful of hot pixels, and a tracker is right to find
        # nothing in it.
        if mean < 5.0:
            verdict = "  <-- ESSENTIALLY BLACK, no corners to find"
        elif std < 5.0:
            verdict = "  <-- flat, very little texture"
        else:
            verdict = "  <-- has real content"
        print(f"  tracker input pixels: mean {mean:.1f}, std {std:.1f}, range [{lo}, {hi}]{verdict}")
        usable_input = mean >= 5.0 and std >= 5.0
    else:
        print("  tracker input pixels: no frame captured")
        usable_input = False

    print(f"  frames to tracker:    {frames_seen}"
          + (f", first at {first_frame_at:.2f} s" if first_frame_at is not None else ""))

    if not with_tracker:
        # No tracker in this pipeline, so "zero features" is the definition, not a
        # finding. What matters from a control run is whether the device survived
        # and how good the input was.
        print(f"  RESULT: control ran {frames_seen} frames"
              + (f" -- {error}" if error else " -- device stayed up"))
    elif not feature_counts:
        print("  RESULT: ZERO TrackedFeatures messages", end="")
    elif max(feature_counts) == 0:
        print(f"  RESULT: {len(feature_counts)} messages, all EMPTY", end="")
    else:
        print(f"  RESULT: {len(feature_counts)} messages, features per message: "
              f"min {min(feature_counts)}, median {int(np.median(feature_counts))}, "
              f"max {max(feature_counts)}", end="")
    if with_tracker:
        print(f" -- {error}" if error else "")

    # Only blame the scene when the scene is actually what was seen. A crash
    # before any frame arrived says nothing about lighting, and an earlier version
    # advised improving it in exactly that case.
    if error and frames_seen == 0:
        print("  NOTE: the device stopped before delivering a single frame, so nothing here")
        print("        reflects the scene, the lighting or the tracker's input. This is a crash")
        print("        during pipeline start-up.")
    elif not usable_input and frames_seen > 0 and not error:
        print("  NOTE: the input was not usable, so a zero-feature result says nothing about")
        print("        the tracker. Improve the lighting or the scene and re-run.")

    time.sleep(2.0)
    return bool(feature_counts) and max(feature_counts) > 0, error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="Device IP. Omit to auto-discover.")
    parser.add_argument("--socket", choices=sorted(SOCKETS), default=None,
                        help="Single socket to test. Default sweeps all three.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--settle", type=float, default=3.0,
                        help="Seconds to let auto-exposure settle before measuring.")
    parser.add_argument("--target-features", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=20000,
                        help="Harris initial threshold. 0 selects automatic, which returns "
                             "empty messages on RVC4. Order 10^4 is the working scale.")
    parser.add_argument("--threshold-sweep", action="store_true",
                        help="Try a range of thresholds on one socket to characterise the effect.")
    parser.add_argument("--direct", action="store_true",
                        help="Link Camera GRAY8 straight to the tracker, no ImageManip.")
    parser.add_argument("--matrix", action="store_true",
                        help="Isolate what differs between the known-working configuration and the failing one, changing one variable at a time.")
    parser.add_argument("--skip-control", action="store_true",
                        help="Skip the no-FeatureTracker control run (halves the runtime, but the control is what makes the result evidence rather than correlation).")
    args = parser.parse_args()

    print("FeatureTracker minimal reproducer")
    print("Path: Camera (NV12) -> ImageManip (GRAY8) -> FeatureTracker\n")
    print(f"depthai version: {dai.__version__}")
    try:
        probe = connect(args.device)
        print(f"device:          {probe.getDeviceName()}")
        print(f"platform:        {probe.getPlatformAsString()}")
        probe.close()
        time.sleep(1.0)
    except Exception as exc:  # noqa: BLE001
        print(f"device:          could not query ({exc})")

    print("\nPoint the camera at something textured and well lit -- a bookshelf, a desk,")
    print("anything visually busy. A blank wall legitimately has no corners to find.")

    if args.threshold_sweep:
        # Characterise the threshold rather than confirming a single value. What
        # matters for the VO is not just "does 20000 work" but how sensitive the
        # feature count is to it -- a knob that only works at one exact value is a
        # different proposition from one with a broad usable range.
        socket_name = args.socket or "CAM_B"
        print(f"\nThreshold sweep on {socket_name}. 0 is the automatic setting.")
        for value in (0, 1000, 5000, 20000, 50000, 200000):
            run(args.device, socket_name, SOCKETS[socket_name],
                (args.width, args.height), args.fps, args.seconds,
                args.target_features, args.settle, with_tracker=True,
                use_manip=not args.direct, threshold=value,
                label=f"threshold {value}" + (" (automatic)" if value == 0 else ""))
        print("\n" + "=" * 72)
        print("A broad plateau means the threshold is a safe thing to set and forget.")
        print("A narrow peak means it will need tuning per scene, which for VO would be a")
        print("real drawback -- lighting changes as the camera moves.")
        return 0

    if args.matrix:
        # A working configuration exists: Camera GRAY8 linked STRAIGHT to the
        # tracker, 640x480, with neither resizeMode nor fps passed. That differs
        # from the failing configuration in three ways at once, so change one
        # thing at a time to find which one matters.
        socket_name = args.socket or "CAM_B"
        socket = SOCKETS[socket_name]
        matrix = [
            ("known-good: direct GRAY8, 640x480, no resizeMode, no fps",
             dict(size=(640, 480), use_manip=False, resize_mode=None, fps=None)),
            ("+ fps=30",
             dict(size=(640, 480), use_manip=False, resize_mode=None, fps=args.fps)),
            ("+ resizeMode=CROP",
             dict(size=(640, 480), use_manip=False, resize_mode=dai.ImgResizeMode.CROP, fps=None)),
            ("640x400 instead of 640x480",
             dict(size=(640, 400), use_manip=False, resize_mode=None, fps=None)),
            ("via ImageManip (the advised path)",
             dict(size=(640, 480), use_manip=True, resize_mode=None, fps=None)),
            ("everything the failing config had",
             dict(size=(640, 400), use_manip=True, resize_mode=dai.ImgResizeMode.CROP, fps=args.fps)),
        ]
        print(f"\nIsolating what differs between the working and failing configurations, on {socket_name}.")
        outcomes = []
        for desc, kwargs in matrix:
            ok, err = run(args.device, socket_name, socket,
                          kwargs["size"], kwargs["fps"], args.seconds, args.target_features,
                          args.settle, with_tracker=True, use_manip=kwargs["use_manip"],
                          resize_mode=kwargs["resize_mode"], label=desc, threshold=args.threshold)
            outcomes.append((desc, ok, err))

        print("\n" + "=" * 72)
        print("Matrix result:")
        for desc, ok, err in outcomes:
            print(f"  {'FEATURES' if ok else 'no features':<12}  {desc}")
        good = [d for d, ok, _ in outcomes if ok]
        bad = [d for d, ok, _ in outcomes if not ok]
        if good and bad:
            print(f"\nThe first failing row after a working one is the change that breaks it.")
        elif good:
            print("\nEverything worked. The earlier failures were not caused by any of these.")
        else:
            print("\nNothing worked, including the known-good configuration -- so the difference")
            print("is elsewhere: device state, scene, or something about this session.")
        return 0

    names = [args.socket] if args.socket else ["CAM_A", "CAM_B", "CAM_C"]
    results = {}
    controls = {}
    for name in names:
        results[name] = run(args.device, name, SOCKETS[name],
                            (args.width, args.height), args.fps, args.seconds,
                            args.target_features, args.settle, with_tracker=True,
                            use_manip=not args.direct, threshold=args.threshold)
        if not args.skip_control:
            controls[name] = run(args.device, name, SOCKETS[name],
                                 (args.width, args.height), args.fps, args.seconds,
                                 args.target_features, args.settle, with_tracker=False,
                                 use_manip=not args.direct, threshold=args.threshold)

    working = [n for n, (ok, _) in results.items() if ok]
    print("\n" + "=" * 72)

    # The control is the load-bearing comparison. Without it, "the device crashed
    # while a FeatureTracker was in the pipeline" is not evidence that the tracker
    # caused it -- and that is the first thing any maintainer will ask.
    if controls:
        tracker_failed = [n for n, (_, err) in results.items() if err]
        control_failed = [n for n, (_, err) in controls.items() if err]
        print("Control comparison -- identical pipeline with and without the tracker:")
        for name in names:
            with_err = results[name][1] or "ok"
            without_err = controls[name][1] or "ok"
            print(f"  {name:<6} with tracker: {with_err[:44]:<44} without: {without_err[:30]}")
        print()
        if tracker_failed and not control_failed:
            print("The device is stable without the FeatureTracker and fails with it, on an")
            print("otherwise identical pipeline. That implicates the tracker directly.")
        elif tracker_failed and control_failed:
            print("The device fails WITH AND WITHOUT the FeatureTracker. The tracker is not the")
            print("cause -- something upstream of it is unstable on this device, and a report")
            print("blaming the tracker would be wrong. Investigate the camera and manip path,")
            print("and note CAM_A's 'Signal not present on input', which suggests FSYNC.")
        elif not tracker_failed:
            print("Nothing failed this run.")
    print()

    if working == list(results):
        print("FeatureTracker produced features on every socket tested. Working as intended.")
    elif working:
        print(f"Features on {', '.join(working)} but NOT on "
              f"{', '.join(n for n in results if n not in working)}.")
        print("A per-socket difference is the useful detail here -- include it in any report,")
        print("since it narrows the problem considerably compared with 'does not work'.")
    else:
        print("No features on any socket, on the supported NV12 -> GRAY8 -> tracker path.")
        print("\nInclude with a report:")
        print("  * this output in full, including the control comparison")
        print("  * oakctl device info      (Luxonis OS version)")
        print("  * oakctl device update    (output, even if already current)")
        print("  * this script, which is self-contained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
