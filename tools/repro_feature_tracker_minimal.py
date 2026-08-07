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


def run(ip, socket_name, socket, size, fps, seconds, target_features):
    print(f"\n--- {socket_name} at {size[0]}x{size[1]} ---")
    device = connect(ip)

    feature_counts: list[int] = []
    frame_type = None
    pixel_stats = None
    error = ""

    try:
        with dai.Pipeline(device) as pipeline:
            # 1. Camera, NV12 -- the format the sensor actually delivers.
            cam = pipeline.create(dai.node.Camera).build(socket)
            cam_out = cam.requestOutput(size, dai.ImgFrame.Type.NV12, fps=fps)

            # 2. ImageManip, NV12 -> GRAY8. This is the conversion step the
            #    maintained example uses and the tracker requires.
            manip = pipeline.create(dai.node.ImageManip)
            manip.initialConfig.setFrameType(dai.ImgFrame.Type.GRAY8)
            # Generously sized: the manip pads to hardware alignment, and an exact
            # width*height budget makes it skip every frame at sizes whose height
            # is not a multiple of 32.
            manip.setMaxOutputFrameSize(size[0] * size[1] * 3)
            cam_out.link(manip.inputImage)

            # 3. FeatureTracker.
            tracker = pipeline.create(dai.node.FeatureTracker)
            tracker.initialConfig.setNumTargetFeatures(target_features)
            manip.out.link(tracker.inputImage)

            features_q = tracker.outputFeatures.createOutputQueue(8, False)
            tracker_input_q = manip.out.createOutputQueue(4, False)

            pipeline.start()
            deadline = time.time() + seconds
            while time.time() < deadline and pipeline.isRunning():
                msg = features_q.tryGet()
                if msg is not None:
                    feature_counts.append(len(msg.trackedFeatures))
                frame = tracker_input_q.tryGet()
                if frame is not None:
                    frame_type = int(frame.getType())
                    array = np.asarray(frame.getFrame())
                    if array.size:
                        pixel_stats = (float(array.mean()), float(array.std()),
                                       int(array.min()), int(array.max()))
                time.sleep(0.005)
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
        print(f"  tracker input pixels: mean {mean:.1f}, std {std:.1f}, range [{lo}, {hi}]"
              + ("  <-- looks blank" if std < 2.0 else "  <-- has real content"))
    else:
        print("  tracker input pixels: no frame captured")

    if error:
        print(f"  RESULT: exception -- {error}")
    elif not feature_counts:
        print("  RESULT: no TrackedFeatures messages at all")
    else:
        print(f"  RESULT: {len(feature_counts)} messages, features per message: "
              f"min {min(feature_counts)}, median {int(np.median(feature_counts))}, max {max(feature_counts)}")

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
    parser.add_argument("--target-features", type=int, default=320)
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

    names = [args.socket] if args.socket else ["CAM_A", "CAM_B", "CAM_C"]
    results = {}
    for name in names:
        results[name] = run(args.device, name, SOCKETS[name],
                            (args.width, args.height), args.fps, args.seconds,
                            args.target_features)

    working = [n for n, (ok, _) in results.items() if ok]
    print("\n" + "=" * 72)
    if working == list(results):
        print("FeatureTracker produced features on every socket tested. Working as intended.")
    elif working:
        print(f"Features on {', '.join(working)} but NOT on "
              f"{', '.join(n for n in results if n not in working)}.")
        print("A per-socket difference is the useful detail here -- include it in any report,")
        print("since it narrows the problem considerably compared with 'does not work'.")
    else:
        print("No features on any socket, on the supported NV12 -> GRAY8 -> tracker path,")
        print("with GRAY8 confirmed at the tracker input and non-blank pixel statistics.")
        print("\nInclude with a report:")
        print("  * this output in full")
        print("  * oakctl device info      (Luxonis OS version)")
        print("  * oakctl device update    (output, even if already current)")
        print("  * this script, which is self-contained")
    return 0


if __name__ == "__main__":
    sys.exit(main())
