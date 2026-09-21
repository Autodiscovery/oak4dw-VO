# Bring-up: the hardware tracker, Phase 4 and Phase 5 on the device

Everything here is code that compiles and passes 68 unit tests, and **none of it
has touched the camera**. This is the order to run it in, what each step is
actually asking, and what a failure means. The ordering is not arbitrary: each
step either answers a question the next one depends on, or changes one thing so
that a later measurement can be attributed.

The rule throughout: **change one thing at a time**. The history of this project
is largely a history of compound changes producing uninterpretable results.

## 0. Before anything: is the device healthy?

Roughly twenty firmware crashes accumulated in one session, after which the
device began refusing configurations that had demonstrably worked earlier the
same day. Every result gathered after the first crash of a session is suspect.

```bash
# Power-cycle the camera first. Then:
oakctl list                                                    # find the IP
python3 tools/repro_feature_tracker_minimal.py --device <ip> --health
```

Luxonis also asked for both `oakctl` and the device firmware to be updated. The
existing results are from depthai 3.8.0 / Luxonis OS RVC4 1.37.0. **Update
before re-testing**, or the firmware-crash question cannot be closed either way.

If the health check fails: power-cycle and stop. Nothing measured from a sick
device is worth recording.

## 1. Does the hardware feature tracker work now?

This is the only step that can change the shape of the design, so it goes first.

```bash
python3 tools/probe_feature_tracker.py --device <ip> --luxonis-config
```

It applies both fixes together — an explicit Harris threshold, and
`numMaxFeatures` actually set — and sweeps the threshold, because a Harris
threshold is scene-dependent. Point the camera at something textured and well
lit.

| Outcome | What it means | What to do |
|---|---|---|
| Corners at some thresholds | The node works; it was misconfigured | Step 2 |
| Zero at every threshold, device healthy, firmware updated | The node really is broken on RVC4 | Stay on the CPU tracker, attach the sweep to the bug report |
| Firmware crash | The crash is independent of the configuration | Record it; that half of the bug report stands |

**Do not skip to 1280×800 if it works.** Two changes at once — front-end and
resolution — and you will not know which moved the drift number.

## 2. If it works: A/B the front-end at the current resolution

```yaml
# params/vio.yaml
vio.i_use_hw_tracker: true
vio.i_hw_tracker_threshold: <middle of the working range from step 1>
# leave i_width/i_height at 640x400
```

```bash
oakctl app run . --env OAK_ROS_PEER=<host-ip>
python3 tools/verify_vo_output.py --duration 20 --stationary   # on the host
```

Compare against the recorded CPU baseline: `TRACKING` 100%, 148→141→127→**126
inliers**, **1.5 mm** stationary drift over 20 s, 0.56 ms median estimator cost.

What to expect, and it is not all upside. The hardware block does **not** do
subpixel refinement, and does **not** do the forward–backward consistency check
— both of which the CPU tracker does, and both of which feed straight into pose
error. So:

- `tracker_ms` in the status message should collapse from 8–12 ms to ~0. That is
  the win, and it is the reason to do this.
- Stationary drift may get **worse**. If it does, that is a real trade to weigh
  against the freed CPU, not a bug. A millimetre-scale number becoming a
  centimetre-scale one is not worth 10 ms.

Only if stationary drift holds up, raise `i_width`/`i_height` to 1280×800 and
repeat. That buys back the focal length: depth sigma at 5 m goes from ~59 cm to
~29 cm.

## 3. Phase 4: is the gyro prior actually being used?

This step is not about accuracy. It is about whether the prior exists at all,
and it is separated for a specific reason: **a prior that is silently rejected
every frame looks exactly like a prior that does not help.**

`vio.i_imu_enabled` defaults to true, so this is already on.

```bash
oakctl app logs <app-id> | grep -E "IMU|gyro"
ros2 topic echo /oak/vo/status --field gyro_prior_used
ros2 topic echo /oak/vo/status --field gyro_prior_rejection
```

Startup should report the IMU enabled at its rate, and the IMU→rectified-camera
rotation resolved with the rectification contribution named. Then, every five
seconds:

```
IMU: 199.4 Hz (997 samples) | prior used 49, rejected 0 | gyro bias measured [...]
```

| Symptom | Likely cause | Fix |
|---|---|---|
| `IMU extrinsics unavailable` at startup | No IMU→camera link in the device calibration | Nothing to do; the VO runs on vision alone |
| `prior used 0, rejected N`, reason `gyro coverage ... too low` | Bursty IMU transport, or samples arriving after the frame callback | Raise `i_imu_max_sample_gap_factor`, or `i_imu_batch_report_threshold` to 1 |
| `prior used 0, rejected N`, reason `fewer than two gyro samples buffered` | The IMU queue is not delivering | Check the IMU node appears in the startup log at all |
| `prior used 0, rejected N`, reason `implausible rotation` | Timestamp mismatch — device vs host clock | Real bug; capture the log |
| `gyro bias still warming up` indefinitely | Platform never still enough | Expected on a moving robot; the prior works without it |

Leave it at least a minute stationary so the bias estimate converges, then check
the reported bias is small — order 1e-3 rad/s. A bias of 0.1 rad/s means the
static test is admitting motion.

## 4. Phase 4: does the prior help?

Now the accuracy question, and the only honest way to ask it is a paired A/B on
the same motion. The synthetic test deliberately claims nothing here: on 300
clean corners RANSAC succeeds from any seed.

The prior is for **fast rotation with motion blur**, so test that, not a gentle
walk. Yaw the camera briskly — a shake, not a sweep, so the scene stays in view.

```bash
# A: prior off
ros2 param set /oak vio.i_imu_rotation_prior_weight 0.0
# ... record the same motion, watch num_inliers and inlier_ratio ...
# B: prior on
ros2 param set /oak vio.i_imu_rotation_prior_weight 1.0
```

The parameter is runtime-settable precisely so this A/B does not need a restart.
Watch `num_inliers`, `inlier_ratio` and `solve_ms` during the fast rotation, and
whether `state` ever leaves `TRACKING`.

Expect the prior to help most where tracking was previously breaking, and to
change little elsewhere. **If the prior is being used and things are no better
or slightly worse, suspect the extrinsic** — a wrong IMU→camera rotation makes
the prior confidently wrong in a fixed direction, which is worse than no prior.
Sweeping the weight (0.0 / 0.5 / 1.0) distinguishes the two: a good prior
improves monotonically with weight, a wrong one degrades.

Also re-check the closed-loop drift number from README step 3, because commit
"Fix the iteration budget" changed the estimator's cost profile substantially
(roughly 10× fewer RANSAC iterations on smoothly-tracking frames). `solve_ms`
should have dropped noticeably; if it has not, the seed is not being accepted.

## 5. Phase 5: do LENS and the VO contend?

Host-side, no app changes:

```bash
python3 tools/phase5_resource_check.py --device <ip>
```

It measures **latency and jitter**, not frame rate — this camera is FSYNC-slaved
at a fixed rate, so throughput cannot drop and a throughput test would report
"no contention" whatever the DSP is doing.

Then enable it in the app and check the part the host cannot see:

```yaml
vio.i_enable_neural_depth: true
vio.i_neural_depth_model: NEURAL_DEPTH_SMALL
```

```bash
oakctl app exec <app-id> top -bn1        # compare against the pre-LENS figure
```

Leave it running **ten minutes or more** and watch whether the VO rate and
`tracker_ms` drift. Twenty seconds will not throttle anything, and thermal
throttling is the failure mode that only shows up under sustained load.

## 6. Phase 5: should the VO consume LENS depth?

A separate question, with a likely answer of no.

```bash
python3 tools/phase5_resource_check.py --device <ip> --compare-depth
```

Read the median disagreement against this rig's own depth noise: Phase 0
measured σ_d = 0.3 px, about 6% depth error at 5 m. Well under that and neural
depth is in the noise; comparable or larger and it would be the dominant error
term in the pose.

If you try it (`vio.i_depth_source: neural`), the number that decides it is
closed-loop drift, not agreement — neither source is ground truth, so
disagreement says they differ, not which is wrong.

## What is still open after all of this

- **Absolute scale.** The camera model may still be reading raw fisheye
  intrinsics rather than rectified ones. Rotation and trajectory shape are
  unaffected; absolute scale is not settled. README "Known gaps" has the detail,
  and Phase 0 `rectify` is the resolution.
- **The error-state filter.** Off, and it should stay off until steps 4 and 5
  have produced loop-drift numbers to compare it against. Its VO update is known
  to be over-confident by construction.
- **`stereo_vio_node.cpp` and `stereo_vio_pipeline.cpp` have not been
  compiled here** — no DepthAI or ROS headers on the machine that wrote this
  round. The estimator core and all 68 tests have. Expect to fix call signatures
  on first build; `initialConfigRef()` in `stereo_vio_node.hpp` already absorbs
  the shared_ptr-vs-value question the README flagged.
- **The firmware crash.** Unexplained by the configuration fix, and the one part
  of the bug report that still stands on its own evidence.
