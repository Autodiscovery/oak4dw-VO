#!/usr/bin/env python3
"""Phase 5: does LENS neural depth actually co-run with the VO, and is its depth usable for pose?

Two separate questions, and they have different answers, so this tool asks them
separately.

1. CONTENTION. The design argument for Phase 5 is that the engines are disjoint:
   VO uses the stereo block (and optionally the feature-tracker block) while
   NeuralDepth runs on the DSP. That is a claim about hardware, and it should be
   measured rather than assumed.

   The thing to measure is NOT frame rate. This camera is an FSYNC slave, so its
   rate is fixed externally at ~10 Hz and will not move whatever the device is
   doing -- a throughput test on a device that cannot change throughput reports
   "no contention" no matter what. Contention shows up instead as LATENCY and
   JITTER: the same frames arriving later, and less regularly, once the DSP is
   busy. So this measures the delay from device timestamp to host receipt, and
   its spread, with NeuralDepth off and then on.

2. AGREEMENT. Whether the VO can eat neural depth is a different question from
   whether the two can run together. Neural depth is smoothed and, on low
   texture, partly inferred -- excellent for dense perception, and a locally
   biased depth at a feature is a biased pose. --compare-depth measures how far
   the two disagree, per pixel and as a function of range, which is the number
   that decides whether vio.i_depth_source: neural is worth trying.

Usage:
    python3 tools/phase5_resource_check.py --device <ip>
    python3 tools/phase5_resource_check.py --device <ip> --compare-depth
    python3 tools/phase5_resource_check.py --device <ip> --model NEURAL_DEPTH_MEDIUM

Run it pointed at a normal working scene at working distances, not a blank wall:
neural depth on a textureless surface is exactly where it is inferring rather
than measuring, which flatters the agreement figure in arm 2 and is not the
condition the VO cares about.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np

try:
    import depthai as dai
except ImportError:
    sys.exit("depthai not installed.  pip install depthai")

LEFT = dai.CameraBoardSocket.CAM_B
RIGHT = dai.CameraBoardSocket.CAM_C

MODELS = {
    "NEURAL_DEPTH_NANO": dai.DeviceModelZoo.NEURAL_DEPTH_NANO,
    "NEURAL_DEPTH_SMALL": dai.DeviceModelZoo.NEURAL_DEPTH_SMALL,
    "NEURAL_DEPTH_MEDIUM": dai.DeviceModelZoo.NEURAL_DEPTH_MEDIUM,
    "NEURAL_DEPTH_LARGE": dai.DeviceModelZoo.NEURAL_DEPTH_LARGE,
    "NEURAL_DEPTH_EXTRA_LARGE": dai.DeviceModelZoo.NEURAL_DEPTH_EXTRA_LARGE,
}


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


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarise(name: str, latencies: list[float], count: int, elapsed: float) -> dict:
    return {
        "name": name,
        "frames": count,
        "hz": count / elapsed if elapsed > 0 else 0.0,
        "latency_median_ms": statistics.median(latencies) * 1e3 if latencies else float("nan"),
        "latency_p95_ms": percentile(latencies, 0.95) * 1e3 if latencies else float("nan"),
        # Jitter as the interquartile spread rather than the standard deviation:
        # a single stall would dominate a standard deviation and read as a
        # systematically worse pipeline, which is the wrong conclusion from one
        # outlier.
        "latency_iqr_ms": (percentile(latencies, 0.75) - percentile(latencies, 0.25)) * 1e3 if latencies else float("nan"),
    }


def run_arm(ip, *, with_neural, model, width, height, seconds, collect_depth=False):
    """One arm of the A/B: stereo, optionally with NeuralDepth co-running."""
    device = connect(ip)
    disparity_latency: list[float] = []
    depth_latency: list[float] = []
    disparity_count = 0
    depth_count = 0
    temperatures: list[float] = []
    samples: list[tuple[np.ndarray, np.ndarray]] = []
    error = ""

    try:
        with dai.Pipeline(device) as pipeline:
            left = pipeline.create(dai.node.Camera).build(LEFT)
            right = pipeline.create(dai.node.Camera).build(RIGHT)
            left_out = left.requestOutput((width, height), None, dai.ImgResizeMode.STRETCH)
            right_out = right.requestOutput((width, height), None, dai.ImgResizeMode.STRETCH)

            stereo = pipeline.create(dai.node.StereoDepth)
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DETAIL)
            stereo.setRectification(True)
            stereo.setLeftRightCheck(True)
            stereo.setSubpixel(True)
            left_out.link(stereo.left)
            right_out.link(stereo.right)

            disparity_queue = stereo.disparity.createOutputQueue(4, False)
            depth_queue = None

            if with_neural:
                # Fed the rectified pair with its own rectification off -- the
                # same wiring the VO node uses, so this measures the arrangement
                # that will actually ship rather than a different one.
                neural = pipeline.create(dai.node.NeuralDepth)
                neural.setRectification(False)
                neural.build(stereo.rectifiedLeft, stereo.rectifiedRight, MODELS[model])
                depth_queue = neural.depth.createOutputQueue(4, False)

            # Device-side system information. On RVC4 these fields come from an
            # RVC2-shaped message and may read zero; reported only when they do
            # not, rather than printing a confident 0.0 C.
            logger = pipeline.create(dai.node.SystemLogger)
            logger.setRate(1.0)
            system_queue = logger.out.createOutputQueue(4, False)

            pipeline.start()
            start = time.time()
            deadline = start + seconds
            while time.time() < deadline and pipeline.isRunning():
                frame = disparity_queue.tryGet()
                if frame is not None:
                    disparity_count += 1
                    disparity_latency.append(latency_from(frame))
                depth_frame = depth_queue.tryGet() if depth_queue is not None else None
                if depth_frame is not None:
                    depth_count += 1
                    depth_latency.append(latency_from(depth_frame))
                if collect_depth and frame is not None and depth_frame is not None and len(samples) < 20:
                    samples.append((np.asarray(frame.getFrame()), np.asarray(depth_frame.getFrame())))
                info = system_queue.tryGet()
                if info is not None:
                    average = float(info.chipTemperature.average)
                    if average > 1.0:
                        temperatures.append(average)
                time.sleep(0.002)
            elapsed = time.time() - start
    except Exception as exc:  # noqa: BLE001 - a failing arm is data
        error = str(exc).splitlines()[0][:100]
        elapsed = max(1e-6, seconds)
    finally:
        try:
            device.close()
        except Exception:  # noqa: BLE001
            pass
    time.sleep(2.0)

    result = summarise("with NeuralDepth" if with_neural else "stereo only", disparity_latency, disparity_count, elapsed)
    result["depth_frames"] = depth_count
    result["depth_hz"] = depth_count / elapsed if elapsed > 0 else 0.0
    # The DSP path's own latency. Relevant on its own: if neural depth arrives a
    # long way behind the stereo pair, syncing it into the VO (as
    # vio.i_depth_source: neural does) makes every frame wait for it.
    result["depth_latency_median_ms"] = statistics.median(depth_latency) * 1e3 if depth_latency else float("nan")
    result["temperature_max"] = max(temperatures) if temperatures else None
    result["error"] = error
    result["samples"] = samples
    return result


def latency_from(frame) -> float:
    """Host receipt minus device timestamp, seconds.

    Kept in one place because the timestamp type has moved between depthai
    releases, and a latency measurement that silently falls back to the wrong
    clock would read as contention that is not there.
    """
    stamp = frame.getTimestamp()
    try:
        return max(0.0, time.time() - stamp.timestamp())
    except AttributeError:
        return max(0.0, time.time() - stamp.total_seconds())


def compare_depth(samples, fx_baseline_m: float, subpixel_bits: int) -> None:
    """How far do the block matcher and the neural model disagree?"""
    if not samples:
        print("\nNo paired frames captured, so no depth comparison. Is the scene lit?")
        return

    relative_differences: list[float] = []
    by_range: dict[str, list[float]] = {"0-2 m": [], "2-5 m": [], "5-10 m": [], "10+ m": []}

    for disparity_raw, depth_mm in samples:
        if disparity_raw.shape != depth_mm.shape:
            print(f"\nShapes differ ({disparity_raw.shape} vs {depth_mm.shape}); skipping the comparison.")
            print("Both maps must be at the same resolution for a per-pixel comparison to mean anything.")
            return
        disparity = disparity_raw.astype(np.float64) / float(1 << subpixel_bits)
        valid = (disparity > 1.0) & (depth_mm > 0)
        if not np.any(valid):
            continue
        block_depth = fx_baseline_m / disparity[valid]
        neural_depth = depth_mm[valid].astype(np.float64) * 1e-3
        relative = np.abs(neural_depth - block_depth) / np.maximum(block_depth, 1e-6)
        relative_differences.extend(relative.tolist())
        for depth_value, difference in zip(block_depth, relative):
            if depth_value < 2.0:
                by_range["0-2 m"].append(difference)
            elif depth_value < 5.0:
                by_range["2-5 m"].append(difference)
            elif depth_value < 10.0:
                by_range["5-10 m"].append(difference)
            else:
                by_range["10+ m"].append(difference)

    if not relative_differences:
        print("\nNo pixels were valid in both maps, so there is nothing to compare.")
        return

    print("\n--- Arm 2: do the two depth sources agree? ---")
    print(f"  Pixels compared: {len(relative_differences)}")
    print(f"  Median relative difference: {100 * statistics.median(relative_differences):.1f}%")
    print(f"  p90 relative difference:    {100 * percentile(relative_differences, 0.90):.1f}%")
    print("\n  By range (block-matched depth as the reference):")
    for label, values in by_range.items():
        if values:
            print(f"    {label:<8} median {100 * statistics.median(values):5.1f}%   p90 {100 * percentile(values, 0.90):5.1f}%   ({len(values)} px)")

    median = statistics.median(relative_differences)
    print()
    print("  Read this against what the VO can tolerate. Depth error is scale error, and")
    print("  Phase 0 measured this rig's own disparity noise at sigma_d = 0.3 px, which is")
    print("  about 6% depth error at 5 m. A median disagreement well under that is in the")
    print("  noise; comparable or larger means neural depth would be the dominant error")
    print("  term in the pose, not the block matcher.")
    if median < 0.05:
        print(f"\n  At {100 * median:.1f}% median, vio.i_depth_source: neural is worth an A/B on loop drift.")
    else:
        print(f"\n  At {100 * median:.1f}% median, the two disagree by more than this rig's own depth")
        print("  noise. Keep vio.i_depth_source: block_matcher for the VO. That is not a")
        print("  criticism of the model -- smoothing is what makes it good at dense depth --")
        print("  it is the wrong tool for sparse feature depth.")
    print("\n  One caveat on direction: this measures DISAGREEMENT, not error. Neither")
    print("  source is ground truth here, so a large number says they differ, not which")
    print("  one is wrong. Closed-loop drift is the arbiter.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default=None, help="Device IP. Omit to auto-discover.")
    parser.add_argument("--model", choices=sorted(MODELS), default="NEURAL_DEPTH_SMALL")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--duration", type=float, default=20.0, help="Seconds per arm.")
    parser.add_argument("--compare-depth", action="store_true",
                        help="Also measure how far neural depth and the block matcher disagree.")
    parser.add_argument("--subpixel-bits", type=int, default=5,
                        help="Must match StereoDepth's subpixel setting, as vio.i_subpixel_fractional_bits does.")
    args = parser.parse_args()

    print("Phase 5 resource check. Point the camera at a normal working scene at working")
    print("distances -- not a blank wall.\n")

    print("--- Arm 1: does NeuralDepth co-running cost the stereo path anything? ---")
    print("Frame rate is NOT the measure here: this camera is FSYNC-slaved at a fixed rate,")
    print("so throughput cannot drop. Latency and jitter are what move under contention.\n")

    baseline = run_arm(args.device, with_neural=False, model=args.model, width=args.width,
                       height=args.height, seconds=args.duration)
    loaded = run_arm(args.device, with_neural=True, model=args.model, width=args.width,
                     height=args.height, seconds=args.duration, collect_depth=args.compare_depth)

    print(f"  {'arm':<20} {'disparity Hz':>13} {'lat median':>11} {'lat p95':>9} {'jitter IQR':>11} {'depth Hz':>9} {'max temp':>9}")
    print("  " + "-" * 92)
    for arm in (baseline, loaded):
        temperature = f"{arm['temperature_max']:.1f} C" if arm["temperature_max"] is not None else "n/a"
        print(f"  {arm['name']:<20} {arm['hz']:>13.2f} {arm['latency_median_ms']:>10.1f}ms "
              f"{arm['latency_p95_ms']:>8.1f}ms {arm['latency_iqr_ms']:>10.1f}ms "
              f"{arm['depth_hz']:>9.2f} {temperature:>9}")
        if arm["error"]:
            print(f"      error: {arm['error']}")

    if loaded["error"]:
        print("\nThe NeuralDepth arm failed outright, so co-running is not demonstrated.")
        print("Check the model name against the device's model zoo, and that Luxonis OS is")
        print("current. Leave vio.i_enable_neural_depth false until this arm runs clean.")
        return 1

    if loaded["depth_frames"] == 0:
        print("\nNeuralDepth produced no frames, so nothing was actually co-running and the")
        print("latency comparison above is meaningless. Try a smaller model (NEURAL_DEPTH_NANO).")
        return 1

    print()
    delta_median = loaded["latency_median_ms"] - baseline["latency_median_ms"]
    delta_jitter = loaded["latency_iqr_ms"] - baseline["latency_iqr_ms"]
    print(f"  Latency change: {delta_median:+.1f} ms median, jitter {delta_jitter:+.1f} ms IQR")
    print(f"  Neural depth ran at {loaded['depth_hz']:.2f} Hz against a {loaded['hz']:.2f} Hz stereo path,")
    print(f"  arriving {loaded['depth_latency_median_ms']:.1f} ms behind its own frame timestamp.")
    if loaded["depth_latency_median_ms"] > loaded["latency_median_ms"] + 0.5 * (1000.0 / max(baseline["hz"], 1e-6)):
        print("  That is more than half a frame interval behind the disparity path, so syncing")
        print("  it into the VO (vio.i_depth_source: neural) would hold every frame back waiting")
        print("  for the DSP. Fine for a co-run; think twice about consuming it.")

    # A single threshold would be arbitrary, so this is expressed against the
    # thing that matters: one frame interval. Extra latency well inside a frame
    # interval is absorbed by the pipeline; latency approaching or exceeding one
    # means the VO is now working on older data than it was.
    frame_interval_ms = 1000.0 / max(baseline["hz"], 1e-6)
    print(f"  One frame interval at the measured rate is {frame_interval_ms:.0f} ms.")
    if delta_median < 0.25 * frame_interval_ms and delta_jitter < 0.25 * frame_interval_ms:
        print("\n  The engines look independent: the added latency is a small fraction of a")
        print("  frame interval. Co-running is supported by this measurement.")
    else:
        print("\n  The stereo path got measurably slower with the DSP busy, which is contention")
        print("  rather than the independence the design assumed. Try a smaller model, and")
        print("  re-measure before committing to the co-run.")

    print("\n  Two things this does NOT measure, and both need the app:")
    print("    * ARM CPU. The VO's own cost lives there, and with vio.i_use_hw_tracker")
    print("      false the CPU tracker is 8-12 ms/frame of it. Check under load with")
    print("      'oakctl app exec <app-id> top -bn1' and compare against the figure")
    print("      recorded before neural depth was enabled.")
    print("    * Sustained thermal behaviour. Twenty seconds will not throttle anything.")
    print("      Leave the app running for ten minutes or more and watch whether the VO")
    print("      rate and the 'tracker N ms' figure in its health line drift.")

    if args.compare_depth:
        # fx * baseline in metre-pixels, from the device's own calibration --
        # the same quantity DepthMapSource uses, so a disagreement here is a real
        # disagreement rather than two different camera models.
        try:
            device = connect(args.device)
            calibration = device.readCalibration()
            intrinsics = calibration.getCameraIntrinsics(RIGHT, args.width, args.height)
            fx = intrinsics[0][0]
            baseline_m = calibration.getBaselineDistance() * 0.01
            device.close()
            print(f"\n  Camera model for the comparison: fx = {fx:.2f} px, baseline = {baseline_m:.4f} m")
            compare_depth(loaded["samples"], fx * baseline_m, args.subpixel_bits)
        except Exception as exc:  # noqa: BLE001
            print(f"\n  Could not read the calibration for the depth comparison: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
