#!/usr/bin/env python3
"""Find out what puts this device into external FSYNC slave mode.

The app is capped at 10 FPS because setting the sensor rate fails with

    Cannot override fps while using external FSYNC slave mode

Before reporting that as a device-level fault, rule out the possibility that our
own pipeline is what puts it there. Stereo does legitimately need the two sensors
to expose simultaneously, and FSYNC is the mechanism -- so it is entirely
plausible that asking for a stereo pair is what selects a sync mode, and that the
"external" part is either a depthai setup bug or our own misuse.

The variants isolate that, cheapest first:

    one camera, no fps        does a single sensor run faster than 10 Hz on its own?
    one camera, fps=30        can a single sensor's rate be set at all?
    two cameras, fps=30       does merely opening both sensors trigger it?
    stereo pair, no fps       what rate does the full stereo path give?
    stereo pair, fps=30       reproduces the app's failure

Read it like this:

  * single camera accepts fps, stereo pair rejects it -> the stereo pair setup
    selects the sync mode. Our usage or depthai's stereo configuration, not the
    device sitting in a bad state.
  * every variant rejects fps, including one bare camera -> the device really is
    in external slave mode regardless of what we build, and the bug report stands.
  * single camera runs at 30 Hz without asking -> the 10 Hz is specific to the
    pair, which is a strong hint on its own.

Usage:
    python3 tools/probe_frame_rate.py --device <ip>
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
SENSOR = (1280, 800)


def connect(ip: str | None, attempts: int = 4):
    for attempt in range(attempts):
        try:
            return dai.Device(dai.DeviceInfo(ip)) if ip else dai.Device()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            print(f"    (connect failed: {exc}; retrying in 5 s)")
            time.sleep(5.0)
    raise RuntimeError("unreachable")


def run_variant(ip, *, label, cameras, use_stereo, set_fps, fps, seconds, capability=False):
    """Returns (rate_hz, error_string).

    `capability` selects a different way of asking for the frame rate: declare it
    on an ImgFrameCapability and pass that to requestOutput, rather than
    overriding it via build(sensorFps=...) or requestOutput(fps=...). Both of
    those are refused in FSYNC slave mode, but a capability is a statement of what
    the stream should be rather than an override of sensor timing, so it may go
    through a different path.
    """
    device = connect(ip)
    frames = 0
    error = ""
    start = None

    try:
        with dai.Pipeline(device) as pipeline:
            sensor_fps = fps if set_fps else None
            built = []
            for socket in cameras:
                cam = pipeline.create(dai.node.Camera).build(socket, sensorResolution=SENSOR, sensorFps=sensor_fps)
                built.append(cam)

            if capability:
                cap = dai.ImgFrameCapability()
                cap.size.fixed(SENSOR)
                cap.fps.fixed(fps)
                try:
                    cap.type = dai.ImgFrame.Type.GRAY8
                except Exception:  # noqa: BLE001 - older bindings may not expose it
                    pass
                outputs = [c.requestOutput(cap) for c in built]
            else:
                outputs = [c.requestOutput(SENSOR, type=dai.ImgFrame.Type.GRAY8, fps=fps) for c in built]

            # Every requested output must be linked or queued -- depthai rejects a
            # dangling one with "Always call output->createOutputQueue() or
            # output->link()". The two-camera-no-stereo variant left the second
            # output unused and failed on that rather than on anything about FSYNC.
            spare_queues = []
            if not use_stereo:
                spare_queues = [out.createOutputQueue(1, False) for out in outputs[1:]]

            if use_stereo:
                stereo = pipeline.create(dai.node.StereoDepth)
                stereo.setRectification(True)
                stereo.setLeftRightCheck(True)
                outputs[0].link(stereo.left)
                outputs[1].link(stereo.right)
                measured = stereo.disparity.createOutputQueue(2, False)
            else:
                measured = outputs[0].createOutputQueue(2, False)

            pipeline.start()
            # Discard the first second: startup transients skew a short window.
            settle = time.time() + 1.0
            while time.time() < settle:
                measured.tryGet()
                time.sleep(0.005)

            start = time.time()
            deadline = start + seconds
            while time.time() < deadline and pipeline.isRunning():
                if measured.tryGet() is not None:
                    frames += 1
                for spare in spare_queues:
                    spare.tryGet()  # keep unused outputs drained
                time.sleep(0.002)
    except Exception as exc:  # noqa: BLE001 - a rejected variant is the result
        error = str(exc).splitlines()[0]
    finally:
        try:
            device.close()
        except Exception:  # noqa: BLE001
            pass

    elapsed = (time.time() - start) if start else 0.0
    rate = frames / elapsed if elapsed > 0.5 else float("nan")

    if error:
        short = error if len(error) <= 58 else error[:55] + "..."
        print(f"  {label:<34} {'--':>8}   REJECTED: {short}")
    else:
        print(f"  {label:<34} {rate:>7.1f} Hz   ok")
    time.sleep(2.0)
    return rate, error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="Device IP. Omit to auto-discover.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    args = parser.parse_args()

    print(f"Measuring frame rate, requesting {args.fps:.0f} FPS at {SENSOR[0]}x{SENSOR[1]}.\n")
    print(f"  {'variant':<34} {'rate':>10}   result")
    print("  " + "-" * 78)

    variants = [
        ("one camera, no fps set", dict(cameras=[LEFT], use_stereo=False, set_fps=False), args.fps),
        ("one camera, sensorFps set", dict(cameras=[LEFT], use_stereo=False, set_fps=True), args.fps),
        ("two cameras, no stereo, fps set", dict(cameras=[LEFT, RIGHT], use_stereo=False, set_fps=True), args.fps),
        ("stereo pair, no fps set", dict(cameras=[LEFT, RIGHT], use_stereo=True, set_fps=False), args.fps),
        ("stereo pair, sensorFps set", dict(cameras=[LEFT, RIGHT], use_stereo=True, set_fps=True), args.fps),
        # Capability route: declare the rate on an ImgFrameCapability instead of
        # overriding sensor timing. Different code path, so it may not hit the
        # FSYNC check that refuses the other two.
        (f"one camera, Capability fps.fixed({args.fps:.0f})",
         dict(cameras=[LEFT], use_stereo=False, set_fps=False, capability=True), args.fps),
        (f"stereo pair, Capability fps.fixed({args.fps:.0f})",
         dict(cameras=[LEFT, RIGHT], use_stereo=True, set_fps=False, capability=True), args.fps),
        ("stereo pair, Capability fps.fixed(60)",
         dict(cameras=[LEFT, RIGHT], use_stereo=True, set_fps=False, capability=True), 60.0),
    ]

    results = {}
    for label, kwargs, variant_fps in variants:
        results[label] = run_variant(args.device, label=label, fps=variant_fps, seconds=args.seconds, **kwargs)

    single_ok = not results["one camera, sensorFps set"][1]
    pair_ok = not results["stereo pair, sensorFps set"][1]
    single_rate = results["one camera, no fps set"][0]

    # The capability route is the one that would actually unblock the app, so
    # report on it first regardless of what the override rows say.
    cap_rows = {k: v for k, v in results.items() if "Capability" in k}
    cap_worked = [k for k, (rate, err) in cap_rows.items() if not err and rate == rate and rate > 15.0]
    cap_started = [k for k, (_, err) in cap_rows.items() if not err]

    print("\nCapability route:")
    if cap_worked:
        best = max(cap_worked, key=lambda k: cap_rows[k][0])
        print(f"  WORKS. {best} reached {cap_rows[best][0]:.1f} Hz.")
        print("  Declaring the rate on an ImgFrameCapability gets past the FSYNC restriction that")
        print("  refuses build(sensorFps=...) and requestOutput(fps=...). Switch the app to this")
        print("  and the 10 Hz cap is lifted -- and the FSYNC bug report needs rewriting, because")
        print("  the rate is settable after all, just not by the two obvious routes.")
    elif cap_started:
        rates = ", ".join(f"{cap_rows[k][0]:.1f} Hz" for k in cap_started)
        print(f"  Accepted but did not raise the rate ({rates}).")
        print("  No exception, no effect -- the same silent no-op as requestOutput(fps=...).")
        print("  The rate really is externally fixed.")
    else:
        print("  Rejected as well, so all three routes to the frame rate are blocked.")

    print("\nConclusion on where the mode comes from:")
    if single_ok and not pair_ok:
        print("  A single camera accepts the frame rate; the stereo pair does not. So the sync")
        print("  mode is selected when the pair is set up, not something the device sits in")
        print("  regardless. That reframes it as a depthai stereo-configuration issue (or our")
        print("  misuse of it) rather than a device stuck in external slave mode -- and")
        print("  docs/luxonis-bug-fsync-fps-lock.md needs rewriting before it is filed.")
    elif not single_ok and not pair_ok:
        print("  Even one bare camera refuses the frame rate, with no stereo involved. The")
        print("  device really is in external FSYNC slave mode independently of anything we")
        print("  build, and the bug report stands as written.")
    elif single_ok and pair_ok:
        print("  Both accepted the frame rate. Something differs between this probe and the")
        print("  app -- compare the app's camera setup against run_variant() above; whatever")
        print("  the app does additionally is what triggers the slave mode.")
    else:
        print("  Unexpected combination; read the rows individually.")

    if single_rate == single_rate and single_rate > 15.0:
        print(f"\n  Note a single camera reached {single_rate:.1f} Hz unprompted, so the 10 Hz is not a")
        print("  sensor-wide limit -- it is specific to whatever the pair configuration does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
