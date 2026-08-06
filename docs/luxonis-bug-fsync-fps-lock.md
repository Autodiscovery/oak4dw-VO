# Bug report: OAK 4 D W locked to 10 FPS by external FSYNC slave mode

Ready to file at <https://github.com/luxonis/depthai-core/issues>.

All three routes to the frame rate have been tested against the bindings' actual
signatures, and a single camera with no stereo node behaves identically to the full
stereo pipeline. See the tables below.

## Summary

An OAK 4 D W with nothing attached to its auxiliary/FSYNC connector delivers
**10 FPS and cannot be made to run faster**. Any attempt to set the sensor frame
rate aborts the pipeline:

```
RPC 'startPipeline' failed: Cannot override fps while using external FSYNC slave mode
```

There appears to be no documented way to query the FSYNC mode, no documented way
to leave it, and no documentation that this mode is the default or that 10 FPS is
its rate. Separately, `requestOutput`'s `fps` argument is *silently ignored* in
this state rather than reported, so the cap is invisible until the frame rate is
measured.

## Environment

| | |
|---|---|
| Device | OAK 4 D W (reported model: Luxonis, Inc. OAK4-D R9) |
| Platform | RVC4, linux/arm64 |
| Luxonis OS | RVC4 1.37.0 |
| oakctl agent | 0.25.0 (rvc4) |
| depthai | 3.8.0 |
| Device ID | 3549741690 |
| Connection | PoE |
| Auxiliary / FSYNC connector | nothing attached |

## What FSYNC slave mode means, and why it caps the rate

FSYNC (frame sync, sometimes FSIN) is a hardware pulse marking the start of each
frame exposure. A sensor can either generate it (master / OUTPUT) or be driven by
it (slave / INPUT). In slave mode the sensor does not own its own timing: every
frame begins when a pulse arrives, so the frame rate is a property of the
incoming pulse train, not of any software setting.

That makes depthai's refusal correct in itself — it genuinely cannot honour an
fps request for a sensor whose timing comes from outside. The problem is the
state the device is in, not the refusal.

Note that OAK-D-style stereo devices legitimately use FSYNC *internally*: one
mono sensor is INPUT and the other OUTPUT so the pair exposes simultaneously.
That internal arrangement is expected and desirable. What is not expected is the
device reporting **external** slave mode when there is no external source.

## Steps to reproduce

A single camera is enough. No stereo, no other nodes:

```python
import depthai as dai

with dai.Pipeline(dai.Device()) as pipeline:
    cam = pipeline.create(dai.node.Camera).build(
        dai.CameraBoardSocket.CAM_B, sensorResolution=(1280, 800), sensorFps=30.0
    )
    queue = cam.requestOutput(
        (1280, 800), type=dai.ImgFrame.Type.GRAY8, fps=30.0
    ).createOutputQueue()
    pipeline.start()   # throws
```

Result:

```
RPC 'startPipeline' failed: Cannot override fps while using external FSYNC slave mode
```

Omitting `sensorFps` lets the pipeline start, and the delivered rate is then
**~10 Hz** regardless of everything else.

### All three routes to the frame rate

Every API route that can express a frame rate was tried, using the overload
signatures reported by the bindings themselves rather than guessed:

| Route | Result | Rate |
|---|---|---|
| `build(..., sensorFps=30)` | **raises** `Cannot override fps while using external FSYNC slave mode` | — |
| `requestOutput(size, ..., fps=30)` | accepted, **no effect, no diagnostic** | 9.9 Hz |
| `ImgFrameCapability` with `fps.fixed(30)`, via `requestOutput(capability, onHost)` | accepted, **no effect, no diagnostic** | 9.9 Hz |
| Same capability route with `fps.fixed(60)` | accepted, **no effect, no diagnostic** | 9.9 Hz |

So one route reports the restriction and two silently ignore it. The two silent
ones are the more damaging: with no error and no warning, the only way to discover
the cap is to measure the delivered rate and notice it does not match the request.

Relevant signatures on depthai 3.8.0, for reference:

```
build(self, boardSocket, sensorResolution: Optional[tuple[int,int]] = None,
      sensorFps: Optional[float] = None) -> Camera
requestOutput(self, size, type=None, resizeMode=..., fps: Optional[float] = None,
              enableUndistortion=None) -> Node.Output
requestOutput(self, capability: Capability, onHost: bool) -> Node.Output
```

The measured rate varies slightly between runs — 9.4, 9.6 and 9.9 Hz across
otherwise identical pipelines — which is itself consistent with an asynchronous
external clock rather than an internal divider off the sensor clock.

### Isolation: not caused by the stereo pair

Because a stereo pair legitimately needs FSYNC to expose both sensors together, we
checked whether requesting a pair is what selects the sync mode. It is not:

| Pipeline | `sensorFps` | Result |
|---|---|---|
| One camera (CAM_B only) | not set | starts, **9.9 Hz** |
| One camera (CAM_B only) | 30 | **rejected**, FSYNC slave mode |
| Stereo pair + StereoDepth | not set | starts, **9.9 Hz** |
| Stereo pair + StereoDepth | 30 | **rejected**, FSYNC slave mode |

One bare camera behaves identically to the full stereo pipeline, so the device is
in external FSYNC slave mode independently of what the pipeline contains.

Reproducible with `tools/probe_frame_rate.py --device <ip>` in this repository.

## Evidence that 10 Hz is a hard external cap, not a bottleneck

Each of these was measured rather than assumed:

| Observation | What it rules out |
|---|---|
| 10.0 Hz at 1280×800 **and** at 640×400, unchanged | Not a throughput or bandwidth limit. A rate that does not move with a 4× change in pixel count is not a bottleneck. |
| Exposure 8.3 ms, ISO 288 | Not auto-exposure. 30 FPS permits 33 ms; AE is using a quarter of that. |
| Camera, StereoDepth, ImageManip and downstream stages **all** measured at 10.0 Hz | Nothing downstream is back-pressuring. The camera is the source. |
| A single camera with no stereo node reaches only 9.9 Hz and refuses `sensorFps` identically | Not caused by the stereo pair configuration, and not something our pipeline induces. |
| `requestOutput(..., fps=30)` accepted without complaint, still 10 Hz | The fps argument has no effect in this state and no diagnostic is emitted. |

A related symptom, from a separate failure on this device:

```
[system] [error] /work/src/utilities/FsyncController.cpp:553: Failed to perform
                 ioctl() call to fsync-stm kernel driver and set camera exposure:
                 Inappropriate ioctl for device (25)
```

`Inappropriate ioctl for device` on the fsync-stm driver suggests the FSYNC
controller is being addressed in a way the kernel driver does not accept, which
may be related to why the device believes it is an external slave.

## What we would like to know or see changed

1. **Why is this device in external FSYNC slave mode with nothing connected?**
   If it is a default, that default caps every OAK 4 D W at 10 FPS out of the
   box, which seems unlikely to be intended for a camera advertised at up to
   60 FPS on the stereo pair.
2. **How is the FSYNC mode queried?** We could not find a `depthai` API or
   `oakctl` command that reports it. The only way we discovered the state was by
   triggering the exception.
3. **How is it changed?** Equivalently: how does one return a device to internal
   timing? If this is set in board config or EEPROM, a documented way to inspect
   and reset it would help.
4. **Is 10 FPS a documented fallback** for external slave mode with no incoming
   pulses? If a slave with no master free-runs at a fixed rate, that behaviour and
   the rate should be documented, and ideally warned about at pipeline start.
5. **The two silent routes should not be silent.** `requestOutput(fps=...)` and
   `ImgFrameCapability.fps.fixed(...)` both accept a rate and ignore it without
   any diagnostic, while `build(sensorFps=...)` raises for the same underlying
   reason. Consistency in either direction would help; raising is more useful
   than a no-op. As it stands, a caller using either of the silent routes has no
   way to learn the rate was refused except by measuring the delivered frame rate
   and noticing it does not match. That cost us several debugging cycles chasing
   exposure, bandwidth and message-sync explanations before per-stage rate
   measurement pointed at the camera itself.

## Impact

For visual odometry the frame rate is not a convenience. Inter-frame overlap
determines how much rotation can be tracked before features leave the search
window, so 10 Hz versus 30 Hz directly reduces the angular rate the system can
survive, and triples the distance travelled between frames for a given speed.

## Related

The same device also cannot use `dai::node::FeatureTracker` at all; see
[luxonis-bug-featuretracker-rvc4.md](luxonis-bug-featuretracker-rvc4.md).
