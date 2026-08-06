#!/usr/bin/env python3
"""Find out whether this device's frame rate can be set at all, and by which route.

The app is capped at ~10 FPS. Two ways of asking for a rate are confirmed blocked
by external FSYNC slave mode:

    build(sensorFps=...)        raises "Cannot override fps while using external
                                FSYNC slave mode"
    requestOutput(fps=...)      silently ignored, no error, no effect

A third route -- declaring the rate on an ImgFrameCapability -- is a different
code path and is what this probe is really for.

It also prints the actual pybind11 signatures of build() and requestOutput()
before running anything. pybind11 records every overload in the docstring, so
there is no reason to guess at call shapes: an earlier version of this probe did
guess, got "incompatible function arguments", and reported that Python-level
mistake as though the device had refused.

Errors are classified, because those are not the same thing:

    fsync        the device refused -- a real result
    api          our call did not match the bindings -- our bug, route untested
    error        something else

Usage:
    python3 tools/probe_frame_rate.py --device <ip>
    python3 tools/probe_frame_rate.py --device <ip> --show-api   # signatures only
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


def show_api() -> None:
    """Dump the real signatures rather than guessing at them."""
    for name in ("build", "requestOutput"):
        member = getattr(dai.node.Camera, name, None)
        print(f"=== dai.node.Camera.{name} ===")
        if member is None:
            print("  not present in these bindings\n")
            continue
        doc = (member.__doc__ or "").strip()
        for line in doc.splitlines():
            if "(" in line and ("self" in line or name in line):
                print(f"  {line.strip()}")
        print()

    cap = dai.ImgFrameCapability()
    print("=== dai.ImgFrameCapability ===")
    print(f"  attributes: {', '.join(a for a in dir(cap) if not a.startswith('_'))}\n")


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


def classify(error: str) -> str:
    if not error:
        return "ok"
    low = error.lower()
    if "incompatible function arguments" in low or "has no attribute" in low or "no matching overload" in low:
        return "api"
    if "fsync" in low:
        return "fsync"
    return "error"


def make_capability(fps: float):
    cap = dai.ImgFrameCapability()
    cap.size.fixed(SENSOR)
    cap.fps.fixed(fps)
    try:
        cap.type = dai.ImgFrame.Type.GRAY8
    except Exception:  # noqa: BLE001 - not exposed in every binding version
        pass
    return cap


def request_capability_output(cam, cap):
    """requestOutput's capability overload, trying the plausible call shapes."""
    attempts = (
        lambda: cam.requestOutput(cap),
        lambda: cam.requestOutput(cap, False),
        lambda: cam.requestOutput(capability=cap),
        lambda: cam.requestOutput(capability=cap, onHostChange=False),
    )
    last = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last = exc
    raise TypeError(f"no requestOutput capability overload matched: {last}")


def run_variant(ip, *, label, cameras, use_stereo, set_fps, fps, seconds, capability=False):
    device = connect(ip)
    frames = 0
    error = ""
    start = None

    try:
        with dai.Pipeline(device) as pipeline:
            sensor_fps = fps if set_fps else None
            built = [
                pipeline.create(dai.node.Camera).build(socket, sensorResolution=SENSOR, sensorFps=sensor_fps)
                for socket in cameras
            ]

            if capability:
                cap = make_capability(fps)
                outputs = [request_capability_output(c, cap) for c in built]
            else:
                outputs = [c.requestOutput(SENSOR, type=dai.ImgFrame.Type.GRAY8, fps=fps) for c in built]

            # Every requested output must be linked or queued; depthai rejects a
            # dangling one outright.
            spare = [] if use_stereo else [o.createOutputQueue(1, False) for o in outputs[1:]]

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
            settle = time.time() + 1.0
            while time.time() < settle:
                measured.tryGet()
                time.sleep(0.005)

            start = time.time()
            deadline = start + seconds
            while time.time() < deadline and pipeline.isRunning():
                if measured.tryGet() is not None:
                    frames += 1
                for q in spare:
                    q.tryGet()
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
    kind = classify(error)

    if kind == "ok":
        print(f"  {label:<40} {rate:>7.1f} Hz   ok")
    else:
        short = error if len(error) <= 46 else error[:43] + "..."
        tag = {"fsync": "FSYNC REFUSED", "api": "our API misuse", "error": "error"}[kind]
        print(f"  {label:<40} {'--':>10}   {tag}: {short}")
    time.sleep(2.0)
    return rate, error, kind


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="Device IP. Omit to auto-discover.")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--seconds", type=float, default=6.0)
    parser.add_argument("--show-api", action="store_true", help="Print signatures and exit.")
    args = parser.parse_args()

    show_api()
    if args.show_api:
        return 0

    print(f"Measuring frame rate, requesting {args.fps:.0f} FPS at {SENSOR[0]}x{SENSOR[1]}.\n")
    print(f"  {'variant':<40} {'rate':>10}   result")
    print("  " + "-" * 84)

    f = args.fps
    variants = [
        ("one camera, no fps set", dict(cameras=[LEFT], use_stereo=False, set_fps=False), f),
        ("one camera, sensorFps set", dict(cameras=[LEFT], use_stereo=False, set_fps=True), f),
        ("stereo pair, no fps set", dict(cameras=[LEFT, RIGHT], use_stereo=True, set_fps=False), f),
        ("stereo pair, sensorFps set", dict(cameras=[LEFT, RIGHT], use_stereo=True, set_fps=True), f),
        # Same route as sensorFps above, just a different value. The refusal is
        # about overriding at all, not about the number, so this is expected to
        # fail identically -- included to show that rather than assert it.
        ("stereo pair, sensorFps 60", dict(cameras=[LEFT, RIGHT], use_stereo=True, set_fps=True), 60.0),
        (f"one camera, Capability fps.fixed({f:.0f})",
         dict(cameras=[LEFT], use_stereo=False, set_fps=False, capability=True), f),
        (f"stereo pair, Capability fps.fixed({f:.0f})",
         dict(cameras=[LEFT, RIGHT], use_stereo=True, set_fps=False, capability=True), f),
        ("stereo pair, Capability fps.fixed(60)",
         dict(cameras=[LEFT, RIGHT], use_stereo=True, set_fps=False, capability=True), 60.0),
    ]

    results = {}
    for label, kwargs, variant_fps in variants:
        results[label] = run_variant(args.device, label=label, fps=variant_fps, seconds=args.seconds, **kwargs)

    cap_rows = {k: v for k, v in results.items() if "Capability" in k}
    cap_fast = [k for k, (rate, _, kind) in cap_rows.items() if kind == "ok" and rate == rate and rate > 15.0]
    cap_ok = [k for k, (_, _, kind) in cap_rows.items() if kind == "ok"]
    cap_api = [k for k, (_, _, kind) in cap_rows.items() if kind == "api"]

    print("\nCapability route:")
    if cap_fast:
        best = max(cap_fast, key=lambda k: cap_rows[k][0])
        print(f"  WORKS -- {best} reached {cap_rows[best][0]:.1f} Hz.")
        print("  Declaring the rate on a capability gets past the restriction that refuses")
        print("  build(sensorFps=...). Switch the app to this and the cap is lifted; the FPS bug")
        print("  report becomes an API-consistency issue rather than a hard cap.")
    elif cap_ok:
        rates = ", ".join(f"{cap_rows[k][0]:.1f} Hz" for k in cap_ok)
        print(f"  Accepted but had no effect ({rates}) -- the same silent no-op as")
        print("  requestOutput(fps=...). The rate really is externally fixed.")
    elif cap_api:
        print("  UNTESTED. Every call shape was rejected by the bindings, not by the device:")
        print(f"    {cap_rows[cap_api[0]][1]}")
        print("  Read the signatures printed above and add the correct shape. Do not conclude")
        print("  anything about the frame rate from these rows.")
    else:
        print("  Refused by the device like the other routes.")

    fsync_rows = [k for k, (_, _, kind) in results.items() if kind == "fsync"]
    if fsync_rows:
        print(f"\n{len(fsync_rows)} variant(s) refused by FSYNC slave mode, including single-camera pipelines")
        print("with no stereo node, so the mode is a property of the device rather than of")
        print("anything the pipeline requests.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
