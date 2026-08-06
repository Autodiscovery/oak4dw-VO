# Bug report: FeatureTracker non-functional on RVC4 (OAK-4-D-W)

Ready to file at <https://github.com/luxonis/depthai-core/issues>. Complete — no
placeholders remaining.

---

## Summary

`dai::node::FeatureTracker` returns zero features for every configuration tried
on an OAK-4-D-W (RVC4), and throws an internal assertion that brings the device
firmware down:

```
[FeatureTracker(4)] [error] Node threw exception, stopping the node.
                            Exception message: DS: Assert (hSession != NULL)
```

The node is receiving valid GRAY8 frames with good contrast — verified by
measuring pixel statistics on the exact frames being linked to `inputImage` —
so this is not a blank-input or format problem.

A second, separate issue: `Camera::requestOutput(size, ImgFrame::Type::GRAY8, ...)`
returns **NV12** rather than GRAY8, and feeding NV12 to `FeatureTracker` also
crashes the firmware rather than failing gracefully.

## Environment

| | |
|---|---|
| Device | OAK-4-D-W (reported model: Luxonis, Inc. OAK4-D R9) |
| Platform | RVC4, linux/arm64 |
| Luxonis OS | **RVC4 1.37.0** |
| oakctl agent | 0.25.0 (rvc4) |
| Device ID | 3549741690 |
| GPU | Adreno 740v2, enabled |
| Connection | PoE, 192.168.10.118 |
| depthai | **3.8.0** (installed via `ros-jazzy-depthai-v3`) |
| Host | Ubuntu 24.04, ROS 2 Jazzy, Python 3.12 |

## Issue 1 — FeatureTracker returns no features and asserts

### Expected

`outputFeatures` carries Harris corners for a textured, well-lit scene.

### Actual

Every message carries zero features. The node then throws
`DS: Assert (hSession != NULL)` and the firmware crashes, requiring a device
reconnect.

### Variants tried — all zero features

| Source | Resolution | Config | Pixel stats into tracker | Features |
|---|---|---|---|---|
| Camera CAM_B | 640×400 | explicit `inputConfig` send | mean 0.6, std 8.2 | 0 |
| Camera CAM_B | 640×400 | defaults | mean 0.6, std 8.4 | 0 |
| Camera CAM_B | 640×400 | `setHardwareResources(2,2)` | mean 0.6, std 8.3 | 0 |
| Camera CAM_B | 640×400 | explicit Harris threshold 0.01 | mean 0.8, std 8.6 | 0 |
| Camera CAM_B | 1280×720 | defaults | mean 0.9, std 9.0 | 0 |
| Camera CAM_B | 1280×800 | defaults | **mean 60.7, std 58.0** | 0 |
| StereoDepth `rectifiedLeft` | 640×400 | defaults | mean 3.1, std 5.2 | 0 |
| StereoDepth `rectifiedLeft` | 1280×800 | `setHardwareResources(2,2)` | **mean 43.1, std 36.0** | 0 |

Repeated across two independent sweeps with consistent results.

The bolded rows are the important ones: pixel mean ~50–70 with std ~35–58 over
the full 0–255 range is a normally exposed, well-contrasted image. Harris should
find hundreds of corners. (The low-contrast rows are runs where the firmware had
already crashed and frames were partial.)

Also confirmed independently: `StereoDepth` on the same camera pair produces
healthy disparity, ~40% valid pixels, so the sensors and rectification are fine.

### Configuration used

Defaults were left in place wherever possible, having confirmed against
`FeatureTrackerConfig.hpp` that they are already sensible —
`CornerDetector::type = HARRIS`, `numTargetFeatures = 320`, thresholds `AUTO`,
`MotionEstimator::enable = true`.

```cpp
auto tracker = pipeline->create<dai::node::FeatureTracker>();
tracker->initialConfig->setNumTargetFeatures(320);
manip->out.link(tracker->inputImage);   // GRAY8, verified type == 30
```

### Reproduction

`tools/probe_feature_tracker.py` in this repository sweeps all of the above
against a live device and reports pixel statistics alongside feature counts:

```bash
python3 tools/probe_feature_tracker.py --device <ip> --sweep
```

### Crash dumps

Generated on the host at
`~/.cache/depthai/crashdumps/<hash>/crash_dump_2026-08-06_19_5*.tar.gz`
— several available, one per variant.

## Issue 2 — requestOutput ignores GRAY8, and NV12 crashes the tracker

`Camera::requestOutput` with `ImgFrame::Type::GRAY8` on a mono OV9282 returns
NV12. Linking that to `FeatureTracker::inputImage` gives:

```
[FeatureTracker(1)] [error] Node threw exception, stopping the node.
                            Exception message: Unsupported colorspace format type: 22
[system] [error] FsyncController.cpp:553: Failed to perform ioctl() call to
                 fsync-stm kernel driver and set camera exposure:
                 Inappropriate ioctl for device (25)
```

followed by a firmware crash.

Two things worth separating here:

1. `requestOutput` silently returning a different format than the one requested
   is surprising in itself. If GRAY8 is not supported as a Camera output, an
   error would be much easier to work with than a silent substitution.
2. Regardless of that, an unsupported input format should be a node-level error,
   not a firmware crash. Rejecting the frame and logging would leave the device
   usable.

## Issue 3 — the documented example calls a method that does not exist

The [FeatureTracker node documentation](https://docs.luxonis.com/software-v3/depthai/depthai-components/nodes/feature_tracker/)
gives this as its example:

```python
featureTracker = pipeline.create(dai.node.FeatureTracker)
featureTracker.setHardwareResources(2, 2)
featureTracker.setWaitForConfigInput(True)
```

`setWaitForConfigInput` is not present in the installed Python bindings:

```
AttributeError: 'depthai.node.FeatureTracker' object has no attribute 'setWaitForConfigInput'
```

So the documented example cannot run as written, which also means the docs
cannot be used to check whether a caller is initialising the node correctly.

Sending a `FeatureTrackerConfig` through `inputConfig` explicitly after
`pipeline.start()` does work as an API call, but changes nothing: still zero
features, still a firmware crash.

The same page carries no platform compatibility information, so there is no
documented indication of whether this node is expected to work on RVC4 at all.

## Impact

This blocks hardware-accelerated feature tracking on RVC4, which is the natural
front-end for on-device visual odometry — the RVC4 datasheet advertises Harris
corner detection at 1080p60, and doing it on the ARM cores instead costs
roughly 25 ms/frame that would otherwise be free.

## Questions

1. Is `FeatureTracker` expected to be functional on RVC4 in this release? The
   changelog lists "Initial FeatureTracker node implementation for RVC4" from
   v3.0.0a13, but we cannot get it to produce a single feature.
2. If it is supported, is there a required initialisation step we are missing
   that would explain `hSession == NULL`?
3. Is there a minimum or maximum input resolution? Older documentation mentions
   720p and 480p; we tried 640×400, 1280×720 and 1280×800.
