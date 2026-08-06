# API report: frame rate silently ignored in FSYNC slave mode

Ready to file at <https://github.com/luxonis/depthai-core/issues>.

## What this is not

An earlier draft of this document claimed the device was wrongly in external FSYNC
slave mode with nothing attached to its auxiliary connector. **That was wrong.**
The camera is a slave in a multi-camera rig, wired through the M8 connector, and
being in external FSYNC slave mode is exactly correct for it. The ~10 Hz observed
is simply the rate the FSYNC master was driving; driving it at 30 Hz gives 30 Hz.

The device behaviour is right. What remains is an API and diagnostics problem, and
it is the part that cost real debugging time.

## Summary

When a camera is in external FSYNC slave mode, the three ways to request a frame
rate behave inconsistently:

| Route | Behaviour |
|---|---|
| `build(..., sensorFps=30)` | **raises** `Cannot override fps while using external FSYNC slave mode` |
| `requestOutput(size, ..., fps=30)` | accepted, **silently ignored**, no diagnostic |
| `ImgFrameCapability.fps.fixed(30)` via `requestOutput(capability, onHost)` | accepted, **silently ignored**, no diagnostic |

One route explains the situation clearly. Two accept a value and discard it
without a word. There is also no apparent way to *ask* whether a camera is FSYNC
slaved, so the only route to that knowledge is to trip the exception.

## Environment

| | |
|---|---|
| Device | OAK 4 D W (Luxonis, Inc. OAK4-D R9) |
| Platform | RVC4, linux/arm64 |
| Luxonis OS | RVC4 1.37.0 |
| oakctl agent | 0.25.0 (rvc4) |
| depthai | 3.8.0 |
| Configuration | FSYNC slave in a multi-camera rig, driven via M8 |

## Reproduce

With the camera FSYNC-slaved to an external master running at some rate R:

```python
import depthai as dai

with dai.Pipeline(dai.Device()) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B, sensorResolution=(1280, 800)
    )
    # Asks for 30. Gets R. No error, no warning.
    q = cam.requestOutput(
        (1280, 800), type=dai.ImgFrame.Type.GRAY8, fps=30.0
    ).createOutputQueue()
    pipeline.start()
```

Adding `sensorFps=30.0` to `build()` instead raises immediately and informatively.
The capability route behaves like `requestOutput`: accepted, ignored, silent.

Measured with `tools/probe_frame_rate.py` in this repository, which also prints
the pybind11 overload signatures it used.

## Requests

1. **Make the silent routes consistent with the loud one.** `requestOutput(fps=)`
   and `ImgFrameCapability.fps` should either honour the request or raise as
   `build(sensorFps=)` does. Silently accepting a value that has no effect is the
   hardest version of this to diagnose: nothing in the logs indicates the request
   was dropped, so the only way to discover it is to measure the delivered rate
   and notice the mismatch.

2. **Expose the FSYNC state.** Something like `device.getCameraSyncMode(socket)`,
   or simply an INFO line at pipeline start along the lines of *"CAM_B is an FSYNC
   slave; frame rate is governed externally"*. Either would have turned this from
   a multi-hour investigation into a five-second read. We eliminated auto-exposure
   (8.3 ms), bandwidth (rate identical at 1280×800 and 640×400), and message-sync
   back-pressure (all pipeline stages at the same rate) before the exception from
   route 1 revealed what was going on — and route 1 only appears if you happen to
   use `build(sensorFps=)` rather than the other two.

3. **Document the interaction.** The FSYNC documentation covers master/slave
   wiring, and the Camera documentation covers frame rate, but not what happens
   when the two meet. A sentence in the `Camera` docs noting that `fps` arguments
   are inoperative on an FSYNC-slaved sensor would be enough.

## Not part of this report

`FsyncController.cpp:553: Failed to perform ioctl() ... Inappropriate ioctl for
device (25)` appeared in one log during a separate crash. It may be unrelated to
normal slave operation, and is noted here only in case it is useful.
