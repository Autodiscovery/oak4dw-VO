#!/usr/bin/env python3
"""Isolate why the RVC4 FeatureTracker reports zero corners.

Runs from the HOST against the live device, so each variant takes seconds
instead of a container rebuild. Every remaining hypothesis about the tracker is
a flag here.

The single most important run is the baseline, which is as close to the official
Luxonis example as possible -- camera straight into the tracker, no stereo, no
rectification:

    python3 tools/probe_feature_tracker.py --device <ip>

If that yields zero features, the node is not working on this device and the
problem is not in our pipeline at all. If it yields features, add variables one
at a time until it breaks:

    python3 tools/probe_feature_tracker.py --device <ip> --source rectified
    python3 tools/probe_feature_tracker.py --device <ip> --width 1280 --height 800
    python3 tools/probe_feature_tracker.py --device <ip> --hw-resources 2
    python3 tools/probe_feature_tracker.py --device <ip> --threshold 0.01

Or sweep the interesting ones automatically:

    python3 tools/probe_feature_tracker.py --device <ip> --sweep
"""

from __future__ import annotations

import argparse
import sys
import time

try:
    import depthai as dai
except ImportError:
    sys.exit("depthai not installed.  pip install depthai")

LEFT = dai.CameraBoardSocket.CAM_B
RIGHT = dai.CameraBoardSocket.CAM_C


def run_variant(device, *, source, width, height, fps, hw_resources, threshold,
                num_target, use_manip, seconds, label):
    """Build one pipeline variant and report the feature counts it produces."""
    counts: list[int] = []
    exposures: list[float] = []

    with dai.Pipeline(device) as pipeline:
        left = pipeline.create(dai.node.Camera).build(LEFT)
        left_out = left.requestOutput(
            (width, height),
            dai.ImgFrame.Type.GRAY8 if not use_manip else None,
            dai.ImgResizeMode.CROP,
            fps,
        )

        if source == "rectified":
            # Full VO path: rectification, which is what our app actually feeds
            # the tracker.
            right = pipeline.create(dai.node.Camera).build(RIGHT)
            right_out = right.requestOutput((width, height), dai.ImgFrame.Type.GRAY8,
                                            dai.ImgResizeMode.CROP, fps)
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

        tracker = pipeline.create(dai.node.FeatureTracker)
        if threshold is not None:
            # Explicit Harris threshold. The defaults are AUTO (0), meaning
            # auto-adaptation -- which may simply not be implemented on RVC4.
            corner = dai.FeatureTrackerConfig.CornerDetector()
            corner.numTargetFeatures = num_target
            corner.thresholds.initialValue = threshold
            tracker.initialConfig.setCornerDetector(corner)
        else:
            tracker.initialConfig.setNumTargetFeatures(num_target)

        if hw_resources is not None:
            tracker.setHardwareResources(hw_resources, hw_resources)

        if use_manip:
            manip = pipeline.create(dai.node.ImageManip)
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
            manip.setMaxOutputFrameSize(width * height * 2)
            tracker_source.link(manip.inputImage)
            manip.out.link(tracker.inputImage)
        else:
            tracker_source.link(tracker.inputImage)

        feature_queue = tracker.outputFeatures.createOutputQueue(4, False)
        image_queue = left_out.createOutputQueue(1, False)

        pipeline.start()
        deadline = time.time() + seconds
        while time.time() < deadline and pipeline.isRunning():
            features = feature_queue.tryGet()
            if features is not None:
                counts.append(len(features.trackedFeatures))
            frame = image_queue.tryGet()
            if frame is not None:
                exposures.append(frame.getExposureTime().total_seconds() * 1000.0)
            time.sleep(0.005)

    n = len(counts)
    best = max(counts) if counts else 0
    mean = sum(counts) / n if n else 0.0
    exposure = sum(exposures) / len(exposures) if exposures else float("nan")
    verdict = "FEATURES" if best > 0 else "zero"
    print(f"  {label:<52} {n:>4} msgs  mean {mean:>6.1f}  max {best:>4}  "
          f"exp {exposure:>5.1f} ms   {verdict}")
    return best > 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="Device IP. Omit to auto-discover.")
    parser.add_argument("--source", choices=("camera", "rectified"), default="camera")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--hw-resources", type=int, default=None,
                        help="Call setHardwareResources(n, n). Omit to leave it unset.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Explicit Harris initial threshold. Omit for AUTO.")
    parser.add_argument("--num-target", type=int, default=320)
    parser.add_argument("--manip", action="store_true",
                        help="Insert an ImageManip forcing GRAY8 before the tracker.")
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--sweep", action="store_true", help="Try the interesting combinations.")
    args = parser.parse_args()

    device = dai.Device(dai.DeviceInfo(args.device)) if args.device else dai.Device()
    print(f"Device: {device.getDeviceName()}  platform={device.getPlatformAsString()}")
    print("Point the camera at something textured and well lit.\n")

    if not args.sweep:
        ok = run_variant(
            device, source=args.source, width=args.width, height=args.height, fps=args.fps,
            hw_resources=args.hw_resources, threshold=args.threshold,
            num_target=args.num_target, use_manip=args.manip, seconds=args.seconds,
            label=f"{args.source} {args.width}x{args.height}",
        )
        return 0 if ok else 1

    # Ordered so the simplest configuration is first. The first row that
    # produces features tells us what is sufficient; the first that stops
    # producing them tells us what breaks it.
    variants = [
        ("baseline: camera 640x400, all defaults",
         dict(source="camera", width=640, height=400, hw_resources=None, threshold=None, use_manip=False)),
        ("+ setHardwareResources(2,2)",
         dict(source="camera", width=640, height=400, hw_resources=2, threshold=None, use_manip=False)),
        ("+ setHardwareResources(1,1)",
         dict(source="camera", width=640, height=400, hw_resources=1, threshold=None, use_manip=False)),
        ("+ explicit Harris threshold 0.01",
         dict(source="camera", width=640, height=400, hw_resources=None, threshold=0.01, use_manip=False)),
        ("camera at 1280x800 (our app's resolution)",
         dict(source="camera", width=1280, height=800, hw_resources=None, threshold=None, use_manip=False)),
        ("camera 1280x720 (a documented tracker resolution)",
         dict(source="camera", width=1280, height=720, hw_resources=None, threshold=None, use_manip=False)),
        ("rectifiedLeft 640x400 (adds stereo)",
         dict(source="rectified", width=640, height=400, hw_resources=None, threshold=None, use_manip=False)),
        ("rectifiedLeft 640x400 + ImageManip GRAY8",
         dict(source="rectified", width=640, height=400, hw_resources=None, threshold=None, use_manip=True)),
        ("rectifiedLeft 1280x800 + manip (exactly our app)",
         dict(source="rectified", width=1280, height=800, hw_resources=2, threshold=None, use_manip=True)),
    ]

    print(f"  {'variant':<52} {'msgs':>4}  {'mean':>11}  {'max':>4}  {'exposure':>8}")
    print("  " + "-" * 92)
    results = []
    for label, kwargs in variants:
        try:
            results.append((label, run_variant(device, fps=args.fps, num_target=args.num_target,
                                               seconds=args.seconds, label=label, **kwargs)))
        except Exception as exc:  # noqa: BLE001 - a failing variant is data, keep going
            print(f"  {label:<52} FAILED: {exc}")
            results.append((label, False))

    working = [label for label, ok in results if ok]
    print()
    if not working:
        print("No variant produced a single feature, including the near-official baseline.")
        print("That points at the FeatureTracker node on this device/firmware rather than")
        print("at our pipeline. Worth raising with Luxonis, quoting Luxonis OS and depthai")
        print("versions. Meanwhile the front-end would need to change -- the estimator")
        print("itself only needs (id, u, v, disparity) tuples from somewhere.")
    else:
        print("Variants that produced features:")
        for label in working:
            print(f"  - {label}")
        print("\nThe first failing row after a working one identifies the culprit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
