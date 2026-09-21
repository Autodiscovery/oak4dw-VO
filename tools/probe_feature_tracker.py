#!/usr/bin/env python3
"""Isolate why the RVC4 FeatureTracker reports zero corners.

Runs from the HOST against the live device, so each variant takes seconds rather
than a container rebuild.

START HERE:
    python3 tools/probe_feature_tracker.py --device <ip> --luxonis-config

That runs the configuration Luxonis identified as working and sweeps the Harris
threshold. Two things are needed together, and either one missing produces
exactly the same symptom -- a node that runs, emits empty feature messages, and
explains nothing:

  * corner.numMaxFeatures defaults to ZERO in the bindings. Setting only
    numTargetFeatures caps the detector at no features regardless of anything
    else. This probe originally did exactly that.

  * The AUTOMATIC corner threshold returns empty feature messages on RVC4. The
    threshold has to be set explicitly, and the scale is not normalised: a
    working value is 20000. This probe originally passed 0.01, on the assumption
    that it was normalised, and its failure was recorded as evidence that
    explicit thresholds did not help. Six orders of magnitude out, and the wrong
    conclusion drawn from it -- which is why the 0.01 row is still below, now
    labelled as the negative control it always was.

Two further things learned the hard way, both encoded here:

  * Camera.requestOutput(..., GRAY8, ...) is NOT honoured on this device -- it
    returns NV12, and the FeatureTracker rejects NV12 outright with
    "Unsupported colorspace format type: 22" and takes the device firmware down
    with it. So every variant routes through an ImageManip to force GRAY8, and
    each variant gets a fresh device connection with reconnect handling.

  * The tracker throws loudly on a format it dislikes. It never threw on our
    RAW8 or GRAY8 input, which means it was accepting those frames and finding
    nothing in them. So the question is no longer format but CONTENT -- hence
    the pixel statistics below. A blank rectifiedLeft would explain everything,
    and would be consistent with depthai v3.3.0's changelog note about RVC4
    rectified outputs.

    mean/std near zero  -> the image is blank; the tracker is right to find
                           nothing and the fault is upstream of it
    mean/std plausible  -> the image has content and the tracker is at fault

Usage:
    python3 tools/probe_feature_tracker.py --device <ip> --luxonis-config
    python3 tools/probe_feature_tracker.py --device <ip> --sweep
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

LEFT = dai.CameraBoardSocket.CAM_B
RIGHT = dai.CameraBoardSocket.CAM_C


def connect(ip: str | None, attempts: int = 4):
    """Open the device, retrying -- a crashed device needs time to come back."""
    for attempt in range(attempts):
        try:
            return dai.Device(dai.DeviceInfo(ip)) if ip else dai.Device()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            print(f"    (connect failed: {exc}; retrying in 5 s)")
            time.sleep(5.0)
    raise RuntimeError("unreachable")


def stats(frame) -> tuple[float, float, int, int]:
    """mean, std, min, max of a frame's pixels."""
    try:
        array = np.asarray(frame.getFrame())
    except Exception:  # noqa: BLE001
        buffer = np.frombuffer(bytes(frame.getData()), dtype=np.uint8)
        array = buffer
    if array.size == 0:
        return 0.0, 0.0, 0, 0
    return float(array.mean()), float(array.std()), int(array.min()), int(array.max())


def run_variant(ip, *, source, width, height, fps, hw_resources, threshold,
                num_target, seconds, label, send_config=False, wait_for_config=False):
    device = connect(ip)
    counts: list[int] = []
    frame_stats: list[tuple[float, float, int, int]] = []
    tracker_input_type = -1
    error = ""

    try:
        with dai.Pipeline(device) as pipeline:
            left = pipeline.create(dai.node.Camera).build(LEFT)
            # Do NOT ask for GRAY8 here: the request is ignored and NV12 comes
            # back regardless. Take the native format and convert explicitly.
            left_out = left.requestOutput((width, height), None, dai.ImgResizeMode.CROP, fps)

            if source == "rectified":
                right = pipeline.create(dai.node.Camera).build(RIGHT)
                right_out = right.requestOutput((width, height), None, dai.ImgResizeMode.CROP, fps)
                stereo = pipeline.create(dai.node.StereoDepth)
                stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
                stereo.setRectification(True)
                stereo.setLeftRightCheck(True)
                stereo.setSubpixel(True)
                left_out.link(stereo.left)
                right_out.link(stereo.right)
                tracker_source = stereo.rectifiedLeft
            else:
                tracker_source = left_out

            # Always force GRAY8, since the tracker will kill the firmware on NV12.
            manip = pipeline.create(dai.node.ImageManip)
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
            manip.setMaxOutputFrameSize(width * height * 3)
            tracker_source.link(manip.inputImage)

            tracker = pipeline.create(dai.node.FeatureTracker)
            if threshold is not None:
                corner = dai.FeatureTrackerConfig.CornerDetector()
                corner.type = dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS
                # BOTH counts. numMaxFeatures defaults to ZERO, so setting only
                # numTargetFeatures caps the detector at no features at all --
                # silently, and independently of the threshold. This line is
                # half the reason the original "explicit threshold" variant came
                # back empty; the other half was passing 0.01 on a scale where
                # the working value is 20000.
                corner.numMaxFeatures = num_target
                corner.numTargetFeatures = num_target
                thresholds = dai.FeatureTrackerConfig.CornerDetector.Thresholds()
                thresholds.initialValue = threshold
                corner.thresholds = thresholds
                tracker.initialConfig.setCornerDetector(corner)
            else:
                # The automatic threshold. Kept as a variant precisely because
                # it is the failing case Luxonis identified: the node runs,
                # returns empty feature messages, and says nothing about why.
                tracker.initialConfig.setNumTargetFeatures(num_target)
            if hw_resources is not None:
                tracker.setHardwareResources(hw_resources, hw_resources)

            # setWaitForConfigInput appears in the official docs example but is
            # not present in the installed bindings, so the documented example
            # cannot run as written. Guarded rather than removed, so the variant
            # reports "not available" instead of crashing the sweep.
            if wait_for_config:
                if not hasattr(tracker, "setWaitForConfigInput"):
                    raise RuntimeError("setWaitForConfigInput absent from these bindings (docs example cannot run)")
                tracker.setWaitForConfigInput(True)
            config_queue = tracker.inputConfig.createInputQueue() if (send_config or wait_for_config) else None

            manip.out.link(tracker.inputImage)

            feature_queue = tracker.outputFeatures.createOutputQueue(4, False)
            input_queue = manip.out.createOutputQueue(2, False)

            pipeline.start()

            if config_queue is not None:
                config = dai.FeatureTrackerConfig()
                config.setCornerDetector(dai.FeatureTrackerConfig.CornerDetector.Type.HARRIS)
                config.setNumTargetFeatures(num_target)
                config.setMotionEstimator(True)
                config_queue.send(config)
            deadline = time.time() + seconds
            while time.time() < deadline and pipeline.isRunning():
                features = feature_queue.tryGet()
                if features is not None:
                    counts.append(len(features.trackedFeatures))
                frame = input_queue.tryGet()
                if frame is not None:
                    tracker_input_type = int(frame.getType())
                    frame_stats.append(stats(frame))
                time.sleep(0.005)
    except Exception as exc:  # noqa: BLE001 - a failing variant is data
        error = str(exc).splitlines()[0][:60]
    finally:
        try:
            device.close()
        except Exception:  # noqa: BLE001
            pass

    best = max(counts) if counts else 0
    mean_features = sum(counts) / len(counts) if counts else 0.0
    if frame_stats:
        mean_px = sum(s[0] for s in frame_stats) / len(frame_stats)
        std_px = sum(s[1] for s in frame_stats) / len(frame_stats)
        lo = min(s[2] for s in frame_stats)
        hi = max(s[3] for s in frame_stats)
        pixels = f"{mean_px:5.1f}+-{std_px:4.1f} [{lo:3d},{hi:3d}]"
    else:
        pixels = "     no frames     "

    verdict = "FEATURES" if best > 0 else ("ERROR" if error else "zero")
    print(f"  {label:<44} type={tracker_input_type:<3} px {pixels}  "
          f"feat mean {mean_features:6.1f} max {best:<4} {verdict}")
    if error:
        print(f"      error: {error}")
    # The device needs a moment between pipelines, especially after a crash.
    time.sleep(2.0)
    return best > 0, (frame_stats[0][1] if frame_stats else 0.0)


def run_luxonis_config(args) -> int:
    """Run the configuration Luxonis identified as working, sweeping the threshold.

    This is the narrow question -- "does the hardware tracker produce corners on
    THIS device, now?" -- separated from the wide sweep, because the answer
    gates a design decision: with it, the front-end costs ~0 CPU and tracking
    can run at 1280x800; without it, the CPU tracker stays and resolution is
    bounded by ARM cycles.

    Two things are being applied together, and either one alone reproduces the
    original zero-feature symptom:

        corner.numMaxFeatures    -- defaults to 0, which caps the detector at
                                    nothing regardless of the threshold
        thresholds.initialValue  -- the AUTOMATIC threshold returns empty
                                    feature messages on RVC4

    The threshold is swept rather than fixed at the 20000 Luxonis gave, because
    a Harris threshold is scene-dependent: too high on a low-contrast wall finds
    nothing, too low finds noise. The useful output is the shape of the sweep,
    not one number.
    """
    print("Checking the hardware FeatureTracker with the configuration Luxonis identified.")
    print("Both pieces matter: numMaxFeatures (defaults to 0) and an explicit threshold")
    print("(the automatic one returns empty messages on RVC4).\n")

    # Health first. Roughly twenty firmware crashes in one session taught this
    # the hard way: every result gathered after the first crash of a session is
    # suspect, and testing on a sick device generates noise that looks like data.
    print(f"  {'threshold':<12} {'source':<12} {'resolution':<12} features")
    print("  " + "-" * 60)

    results = []
    for threshold in args.thresholds:
        for source in ("rectified", "camera"):
            label = f"{threshold:<12.0f} {source:<12} {args.width}x{args.height}"
            ok, _ = run_variant(args.device, source=source, width=args.width, height=args.height,
                                fps=args.fps, hw_resources=args.hw_resources, threshold=threshold,
                                num_target=args.num_target, seconds=args.seconds, label=label)
            results.append((threshold, source, ok))

    working = [(t, s) for t, s, ok in results if ok]
    print()
    if working:
        best = sorted({t for t, _ in working})
        print("The hardware feature tracker WORKS on this device.")
        print(f"  Thresholds that produced corners: {', '.join(f'{t:.0f}' for t in best)}")
        print(f"  Sources that produced corners:    {', '.join(sorted({s for _, s in working}))}")
        print()
        print("To use it, in params/vio.yaml:")
        print("    vio.i_use_hw_tracker: true")
        print(f"    vio.i_hw_tracker_threshold: {best[len(best) // 2]:.0f}   # middle of the working range")
        print()
        print("Then change ONE thing at a time. Measure closed-loop drift at 640x400 first,")
        print("so the front-end swap is isolated; only then raise i_width/i_height to 1280x800")
        print("to buy back the focal length (depth sigma at 5 m roughly halves, ~59 cm -> ~29 cm).")
        print()
        print("Also worth re-running the stationary test: the hardware block has no")
        print("forward-backward consistency check and no subpixel refinement, both of which")
        print("the CPU tracker does and both of which feed straight into pose error. A worse")
        print("stationary drift number with a cheaper front-end is a real trade, not a bug.")
        return 0

    print("No threshold produced a single corner.")
    print()
    print("Before concluding the node is still broken, rule out the two things that")
    print("produced false negatives here before:")
    print("  1. Device health. If the firmware has crashed at all this session, power-cycle")
    print("     and start again -- 'repro_feature_tracker_minimal.py --health' is the check.")
    print("     Results after the first crash of a session are not evidence.")
    print("  2. Versions. These results were taken on depthai 3.8.0 / Luxonis OS 1.37.0.")
    print("     Luxonis asked for both to be updated; if they have not been, do that first.")
    print()
    print("If it still finds nothing on a healthy, updated device, the report in")
    print("docs/luxonis-bug-featuretracker-rvc4.md stands and should be updated with this")
    print("sweep attached -- an explicit-threshold sweep with numMaxFeatures set is a")
    print("stronger result than anything filed so far.")
    print()
    print("Either way the VO is unaffected: leave vio.i_use_hw_tracker false and the CPU")
    print("front-end carries on as it has been.")
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="Device IP. Omit to auto-discover.")
    parser.add_argument("--source", choices=("camera", "rectified"), default="camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hw-resources", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--num-target", type=int, default=320)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--send-config", action="store_true",
                        help="Send a FeatureTrackerConfig via inputConfig after start.")
    parser.add_argument("--wait-for-config", action="store_true",
                        help="setWaitForConfigInput(true) and send a config, as the docs example does.")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--luxonis-config", action="store_true",
                        help="Run the configuration Luxonis identified as working, sweeping the "
                             "Harris threshold. This is the check to run before enabling "
                             "vio.i_use_hw_tracker.")
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[1000.0, 5000.0, 20000.0, 50000.0, 200000.0],
                        help="Thresholds to sweep with --luxonis-config (default brackets the "
                             "20000 Luxonis gave, since the right value depends on the scene).")
    args = parser.parse_args()

    print("Point the camera at something textured and well lit.\n")

    if args.luxonis_config:
        return run_luxonis_config(args)

    if not args.sweep:
        ok, _ = run_variant(args.device, source=args.source, width=args.width, height=args.height,
                            fps=args.fps, hw_resources=args.hw_resources, threshold=args.threshold,
                            num_target=args.num_target, seconds=args.seconds,
                            send_config=args.send_config, wait_for_config=args.wait_for_config,
                            label=f"{args.source} {args.width}x{args.height}")
        return 0 if ok else 1

    variants = [
        # The Luxonis-identified configuration first, since it is the one with a
        # reason to work. Everything below it is the historical sweep, retained
        # so a regression is distinguishable from a fresh fault.
        ("LUXONIS CONFIG: explicit threshold 20000 + numMaxFeatures",
         dict(source="rectified", width=640, height=400, hw_resources=None, threshold=20000.0)),
        ("LUXONIS CONFIG from camera, not rectified",
         dict(source="camera", width=640, height=400, hw_resources=None, threshold=20000.0)),
        # The two documented-example variants: setWaitForConfigInput plus an
        # explicit inputConfig send is what the official docs example does,
        # though the method is absent from the installed bindings.
        ("docs example: setWaitForConfigInput + config send",
         dict(source="camera", width=640, height=400, hw_resources=2, threshold=None,
              send_config=True, wait_for_config=True)),
        ("explicit config send, no wait flag",
         dict(source="camera", width=640, height=400, hw_resources=2, threshold=None,
              send_config=True, wait_for_config=False)),
        ("camera 640x400, defaults",
         dict(source="camera", width=640, height=400, hw_resources=None, threshold=None)),
        ("camera 640x400 + hwResources(2,2)",
         dict(source="camera", width=640, height=400, hw_resources=2, threshold=None)),
        # Retained as a NEGATIVE control, not as a test of explicit thresholds.
        # 0.01 was chosen on the assumption that the field was normalised 0-1;
        # the working value is 20000. This row is what a threshold six orders of
        # magnitude too low looks like, which is the same as no corners at all --
        # and that indistinguishability is why the original conclusion was wrong.
        ("negative control: threshold 0.01 (wrong by 10^6)",
         dict(source="camera", width=640, height=400, hw_resources=None, threshold=0.01)),
        ("camera 1280x720",
         dict(source="camera", width=1280, height=720, hw_resources=None, threshold=None)),
        ("camera 1280x800 (our resolution)",
         dict(source="camera", width=1280, height=800, hw_resources=None, threshold=None)),
        ("rectifiedLeft 640x400",
         dict(source="rectified", width=640, height=400, hw_resources=None, threshold=None)),
        ("rectifiedLeft 1280x800 (our app)",
         dict(source="rectified", width=1280, height=800, hw_resources=2, threshold=None)),
    ]

    print(f"  {'variant':<44} {'type':<8} {'pixels mean+-std [min,max]':<21}  features")
    print("  " + "-" * 104)
    results = []
    for label, kwargs in variants:
        ok, std = run_variant(args.device, fps=args.fps, num_target=args.num_target,
                              seconds=args.seconds, label=label, **kwargs)
        results.append((label, ok, std))

    working = [label for label, ok, _ in results if ok]
    blank = [label for label, _, std in results if std < 1.0]

    print()
    if working:
        print("Variants that produced features:")
        for label in working:
            print(f"  - {label}")
        print("\nThe first failing row after a working one identifies the culprit.")
    elif blank and len(blank) == len(results):
        print("No variant produced features AND every frame was essentially blank")
        print("(std < 1 grey level). The tracker is not the problem -- it is being fed")
        print("empty images. Look at whether the ImageManip conversion is actually")
        print("producing pixels on this platform.")
    else:
        print("No variant produced a single feature, and the frames were NOT blank --")
        print("the tracker is receiving real images and detecting nothing in them.")
        print("That points at the FeatureTracker on this device/firmware rather than at")
        print("our pipeline. Worth raising with Luxonis, quoting Luxonis OS and depthai")
        print("versions and the 'Unsupported colorspace format type: 22' firmware crash.")
        print("\nThe estimator is unaffected either way: it consumes (id, u, v, disparity)")
        print("tuples, so swapping in a CPU Harris + Lucas-Kanade front-end is contained")
        print("to one file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
