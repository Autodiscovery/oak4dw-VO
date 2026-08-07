#!/usr/bin/env python3
"""Find out which call shape actually sets the frame rate on an OAK4-D.

Luxonis report that this works, at 20 and at 13 FPS, with the stereo pair staying
FSYNC-synchronised under the default AUTO sync mode:

    left_cam  = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_cam = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    left  = left_cam.requestOutput((640, 400), fps=fps)
    right = right_cam.requestOutput((640, 400), fps=fps)

Note what is NOT there: no sensorResolution on build(), no type, no resizeMode.
Our app passes all three, and gets ~10 Hz regardless of what it asks for. Any of
them could be the reason, and so could the rate itself -- we only ever tried 30,
never 20 or 13.

So vary one thing at a time and measure the rate actually delivered.

    python3 tools/probe_fps_matrix.py --device <ip>
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


def connect(ip, attempts=4):
    for attempt in range(attempts):
        try:
            return dai.Device(dai.DeviceInfo(ip)) if ip else dai.Device()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1:
                raise
            print(f"    (connect failed: {exc}; retrying in 5 s)")
            time.sleep(5.0)
    raise RuntimeError("unreachable")


def wait_until_healthy(ip, timeout=90.0):
    """Block until the device can be opened and closed cleanly.

    A firmware crash takes 10-20 s just to extract its dump, and longer to be
    reachable again. Running the next variant before then produces a failure
    that belongs to the previous crash, not to the configuration being tested.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            dev = dai.Device(dai.DeviceInfo(ip)) if ip else dai.Device()
            dev.close()
            time.sleep(1.0)
            return True
        except Exception:  # noqa: BLE001
            time.sleep(3.0)
    return False


def measure(ip, *, label, size, fps, sensor_resolution, frame_type, resize_mode,
            both_cameras, seconds):
    """Build one variant and return the rate each camera actually delivered."""
    device = connect(ip)
    counts = [0, 0]
    error = ""
    started = None

    try:
        with dai.Pipeline(device) as pipeline:
            sockets = [LEFT, RIGHT] if both_cameras else [LEFT]
            queues = []
            for socket in sockets:
                build_kwargs = {}
                if sensor_resolution is not None:
                    build_kwargs["sensorResolution"] = sensor_resolution
                cam = pipeline.create(dai.node.Camera).build(socket, **build_kwargs)

                request_kwargs = {}
                if frame_type is not None:
                    request_kwargs["type"] = frame_type
                if resize_mode is not None:
                    request_kwargs["resizeMode"] = resize_mode
                if fps is not None:
                    request_kwargs["fps"] = fps

                out = cam.requestOutput(size, **request_kwargs)
                queues.append(out.createOutputQueue(4, False))

            pipeline.start()
            started = time.time()
            # A second of settle so start-up transients do not skew a short window.
            settle_until = started + 1.0
            while time.time() < settle_until:
                for q in queues:
                    q.tryGet()
                time.sleep(0.005)

            started = time.time()
            deadline = started + seconds
            while time.time() < deadline:
                for i, q in enumerate(queues):
                    if q.tryGet() is not None:
                        counts[i] += 1
                time.sleep(0.002)
    except Exception as exc:  # noqa: BLE001 - a rejected variant is the result
        error = str(exc).splitlines()[0]
    finally:
        try:
            device.close()
        except Exception:  # noqa: BLE001
            pass

    elapsed = (time.time() - started) if started else 0.0
    rates = [c / elapsed if elapsed > 0.5 else float("nan") for c in counts]

    asked = f"{fps:.0f}" if fps else "unset"
    if error:
        print(f"  {label:<44} asked {asked:>5}   ERROR: {error[:38]}")
    elif both_cameras:
        synced = "in step" if abs(rates[0] - rates[1]) < 1.0 else "MISMATCHED"
        print(f"  {label:<44} asked {asked:>5}   got {rates[0]:5.1f} / {rates[1]:5.1f} Hz   {synced}")
    else:
        print(f"  {label:<44} asked {asked:>5}   got {rates[0]:5.1f} Hz")

    # A crashed device needs far longer than a healthy one before the next
    # variant means anything.
    time.sleep(2.0 if (not error and max(counts) > 0) else 10.0)
    return rates, error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--only", default=None,
                        help="Run only variants whose label contains this substring.")
    parser.add_argument("--recover", type=float, default=90.0,
                        help="Max seconds to wait for the device to become reachable again after a crash.")
    args = parser.parse_args()

    print(f"depthai {dai.__version__}\n")
    print(f"  {'variant':<44} {'asked':>11}   delivered")
    print("  " + "-" * 84)

    base = dict(size=(640, 400), sensor_resolution=None, frame_type=None,
                resize_mode=None, both_cameras=True, seconds=args.seconds)

    variants = [
        # Exactly the reported-working call shape, at the two rates verified.
        ("Luxonis form, fps=20", dict(base, fps=20.0)),
        ("Luxonis form, fps=13", dict(base, fps=13.0)),
        ("Luxonis form, fps=30", dict(base, fps=30.0)),
        ("Luxonis form, no fps at all", dict(base, fps=None)),

        # One deviation at a time, to find which of ours breaks it.
        ("+ type=GRAY8", dict(base, fps=20.0, frame_type=dai.ImgFrame.Type.GRAY8)),
        ("+ resizeMode=STRETCH", dict(base, fps=20.0, resize_mode=dai.ImgResizeMode.STRETCH)),
        ("+ sensorResolution=(1280,800) on build()",
         dict(base, fps=20.0, sensor_resolution=(1280, 800))),
        ("output 1280x800 instead of 640x400", dict(base, fps=20.0, size=(1280, 800))),
        ("single camera only", dict(base, fps=20.0, both_cameras=False)),

        # Everything our app currently does, for comparison.
        ("everything our app does",
         dict(base, fps=20.0, sensor_resolution=(1280, 800),
              frame_type=dai.ImgFrame.Type.GRAY8, resize_mode=dai.ImgResizeMode.STRETCH)),
    ]

    if args.only:
        variants = [(l, k) for l, k in variants if args.only.lower() in l.lower()]
        if not variants:
            print(f"  no variant matches {args.only!r}")
            return 1

    results = []
    previous_failed = False
    for label, kwargs in variants:
        if previous_failed:
            print("  (waiting for the device to recover from the previous crash...)")
            if not wait_until_healthy(args.device, args.recover):
                print("  device did not come back; stopping rather than reporting cascade failures.")
                print("  Power-cycle it and re-run, ideally with --only to test fewer variants.")
                break
        rates, error = measure(args.device, label=label, **kwargs)
        crashed = bool(error) or not any(r == r and r > 0.0 for r in rates)
        if crashed and previous_failed:
            print("      ^ follows an earlier crash; treat with suspicion")
        previous_failed = crashed
        results.append((label, kwargs.get("fps"), rates, error))

    print("\n" + "=" * 88)
    honoured = []
    ignored = []
    for label, fps, rates, error in results:
        if error or fps is None:
            continue
        if rates and rates[0] == rates[0] and abs(rates[0] - fps) < 0.2 * fps:
            honoured.append((label, fps, rates[0]))
        else:
            ignored.append((label, fps, rates[0] if rates else float("nan")))

    if honoured:
        print("Rate was honoured for:")
        for label, fps, got in honoured:
            print(f"  {got:5.1f} Hz for {fps:.0f} asked   {label}")
    if ignored:
        print("\nRate was IGNORED for:")
        for label, fps, got in ignored:
            print(f"  {got:5.1f} Hz for {fps:.0f} asked   {label}")

    if honoured and ignored:
        print("\nThe difference between those two lists is what actually controls the rate.")
        print("Whatever the first ignored row adds over the last honoured one is the culprit,")
        print("and our app should stop doing it.")
    elif honoured:
        print("\nEvery variant honoured the request, including the ones our app uses -- so the")
        print("10 Hz has some other cause. Re-check the app against these call shapes.")
    else:
        print("\nNo variant honoured the request, including the reported-working one. Either")
        print("something differs about this rig -- it is FSYNC-slaved in a multi-camera setup --")
        print("or the rate is externally governed here regardless of call shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
