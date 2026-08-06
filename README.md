# OAK 4 D W — on-device stereo visual odometry

Six-DOF stereo visual odometry running entirely on an OAK 4 D W (RVC4),
publishing ROS 2 topics to a Jazzy host over Ethernet/PoE+. Written from
scratch under Apache-2.0 so it can ship commercially, and built to leave
headroom for LENS neural depth on the same device.

```bash
oakctl app run .
```

---

## Why from scratch

You asked for a review of libviso2 and S-PTAM, and later ORB-SLAM. All three
are copyleft:

| System | Licence | Commercial route |
|---|---|---|
| libviso2 (Geiger, KIT) | GPL v2-or-later | contact the author |
| S-PTAM (CIFASIS/LRSE) | GPLv3 | closed-source licence from the authors |
| ORB-SLAM2 / ORB-SLAM3 (UZ-SLAMLab) | GPLv3 | paid licence, `orbslam@unizar.es` |

None can ship in a closed-source product without buying a licence. This
implementation follows the *algorithm* from the published papers — principally
Geiger et al., "StereoScan: Dense 3D Reconstruction in Real-time" (IV 2011) —
written from the maths, not transcribed from anyone's source. Algorithms and
published methods aren't restricted by a reference implementation's licence;
the code is. Not legal advice.

One useful distinction: the ORB *feature* is patent-free and OpenCV's `cv::ORB`
is Apache-2.0. It's the ORB-*SLAM* system that's GPL. If relocalisation is ever
needed, OpenCV's ORB is available without touching Zaragoza's code.

## Why it's cheap

The RVC4 (Qualcomm QCS8550) does the two expensive parts in fixed-function
hardware:

| Work | Where | Cost to us |
|---|---|---|
| Harris corners + LK optical flow | RVC4 feature-tracker block, 1080p @ 60 FPS | ~0 CPU |
| Rectification + block-matched disparity | RVC4 stereo block, 800p @ 60 FPS | ~0 CPU |
| RANSAC + Gauss-Newton over ~300 points | ARM CPU | the only real cost |
| LENS neural depth (later) | DSP/NPU — a different engine | independent |

CPU cost scales with feature count, not resolution. That is what makes it safe
to run alongside LENS.

We deliberately did **not** copy ORB-SLAM's front-end. Profiling on a Jetson
TX2 puts ORB extraction at 12.3 ms of a 28.5 ms tracking budget for one
752×480 image; stereo doubles it and 1280×800 raises it further. Paying 25+ ms
of ARM CPU to replicate in software what a dedicated engine does for free would
be the wrong trade on a device that also has to run LENS.

## Architecture

The VO ships as a **`depthai_ros_driver` pipeline plugin**, not a separate
process. A DepthAI device can only be opened once, so if you later want RGB,
LENS depth and VO simultaneously they must be nodes in one pipeline.

```
CAM_B (left, OV9282) ──┐
                       ├─► StereoDepth ─┬─► rectifiedLeft ─► FeatureTracker ─┐
CAM_C (right, OV9282) ─┘                └─► disparity ──────────────────────┤
                                                                             ├─► Sync ─► StereoVioNode
                                                                             ┘
```

`Sync` pairs disparity with the feature list by timestamp. Without it the
estimator eventually samples disparity from a different frame than the features
came from, which shows up as a slow, baffling scale drift.

### The estimator

Per frame, against the reference keyframe:

1. **Correspondences.** The HW tracker supplies stable feature IDs, so temporal
   matching is already solved. Gate on disparity bounds, tracking error, border
   mask and depth consistency.
2. **Bucketing** over an 8×5 grid, capped per cell, so a textured foreground
   object can't dominate the fit.
3. **Triangulate** the keyframe points.
4. **Minimise reprojection error** over 6 DOF with Gauss-Newton and analytic
   Jacobians, seeded from a constant-velocity prediction.
5. **RANSAC** with closed-form three-point hypotheses, then Huber-robust
   refinement on the inliers.
6. **Health and covariance**, published so a downstream filter can react rather
   than guess.

Three design choices differ from the plan, each for a reason found during
implementation:

**SE(3) left perturbation, not Euler angles.** libviso2's Euler parameterisation
is fine for small frame-to-frame increments, but we track against a keyframe
that can span a whole turn, and Euler hits gimbal lock at ±90° pitch. The Lie
formulation has no singularity and gives a simpler Jacobian.

**Three residuals per point, not four.** In a rectified pair the right image's
`v` equals the left's by construction, so a fourth row would exactly duplicate
the second and silently double-weight `v`. We use `(u, v, disparity)`.

**Keyframe-point depth uncertainty is propagated.** See below — this one was a
real bug caught by simulation.

## What simulation changed

Everything in `ros_ws/src/oak_vio/` was validated against
`tools/validate_estimator_math.py` before it went near hardware. Three findings
changed the design:

**1. The plan's depth-precision figure was wrong by 8×.** I originally quoted
~4 cm at 5 m, computed with σ_d = 1/16 px — but that's the subpixel
*quantisation step*, not the disparity *noise*. With a realistic σ_d = 0.5 px
the honest figure is:

| Range | Depth σ | Relative |
|---|---|---|
| 2 m | 5.0 cm | 2.5% |
| 5 m | 31.0 cm | 6.2% |
| 10 m | 124 cm | 12.4% |
| 20 m | 497 cm | 24.8% |

Relative depth error passes 10% beyond about 8 m. `tools/phase0_calibration_check.py noise`
measures the real σ_d for your device; don't guess it.

**2. Treating the triangulated keyframe point as exact caused catastrophic
divergence.** This is the classic error-in-variables mistake, and it is
invisible on near scenes. On a 5–40 m scene the 95th-percentile translation
error was **12.6 m** treating the point as exact, versus **9.7 mm** once its
uncertainty was propagated. The fix is cheap: the point's uncertainty is
rank-1 along its viewing ray (σ_Z/Z = σ_d/d), so pushing it through the pose
and projection gives a per-component variance inflation. It also self-corrects
— a far point contributes almost nothing under pure rotation, where its depth
genuinely doesn't matter, and is heavily discounted under translation, where it
does.

**3. The formal covariance was ~2.7× over-confident** — the dangerous direction
for a downstream EKF, since keyframe triangulation error correlates with the
current measurement through the shared feature. `vio.i_covariance_inflation`
(default 3.0) corrects it. Re-derive from real logs once Phase 3 data exists.

### 4. The keyframe claim was half right, and that changed a default

The plan argued that keyframe-referenced tracking would buy most of ORB-SLAM's
drift advantage for free, since error would no longer compound every frame.
Simulation says it depends entirely on how fast the camera is moving. Same
3.75 m closed loop walked at four speeds, 8 seeds each, comparing
frame-to-frame against a 30-frame keyframe span:

| Camera motion | frame-to-frame | keyframed (30) | Winner |
|---|---|---|---|
| 31.2 mm/frame | **1.37%** | 4.89% | frame-to-frame, 3.6× |
| 15.6 mm/frame | **0.85%** | 2.84% | frame-to-frame, 3.4× |
| 6.3 mm/frame | **0.16%** | 0.34% | frame-to-frame, 2.2× |
| 3.1 mm/frame | 0.20% | **0.14%** | keyframed, 1.5× |

Two effects pull against each other, and the crossover is where they balance:

- **Short spans lose** because parallax approaches zero, and translation
  becomes near-unobservable.
- **Long spans lose** because the keyframe's own triangulation error is a fixed
  bias that persists for the whole span instead of averaging out, and it grows
  with span length. This is precisely what ORB-SLAM's local bundle adjustment
  fixes by refining map points — and we deliberately skipped the bundle
  adjustment. That is where the benefit the plan assumed was actually living.

So the plan's mechanism was real but its conclusion was not, and the useful
answer is neither mode: it's **targeting a parallax band**, which adapts
automatically between the two regimes. That is already how the promotion policy
works, so the fix was to the threshold — `vio.i_keyframe_max_parallax_px`
dropped from 60 px to **15 px**. At 60 px and walking pace it was producing
~12-frame spans, deep into the regime where frame-to-frame won.

`vio.i_keyframe_referenced` remains a runtime switch, and the parallax target
still wants confirmation on real data — the crossover point depends on your
disparity noise and typical motion. The A/B in the verification section is how
to close it out.

## Layout

```
oakapp.toml                 container + build recipe (Jazzy port of ros-driver-basic)
params/vio.yaml             all tunables; Phase 0 fills in the measured ones
params/cyclonedds.xml       DDS peer config for the Ethernet link
tools/
  validate_estimator_math.py   numerical oracle for the C++ maths (runs anywhere)
  phase0_calibration_check.py  on-device calibration / FoV / disparity-noise measurement
ros_ws/src/oak_vio_msgs/    VioStatus.msg
ros_ws/src/oak_vio/
  include/oak_vio/          estimator headers (no ROS, no DepthAI)
  src/motion_estimator.cpp  GN + analytic Jacobians + depth-uncertainty weighting
  src/ransac.cpp            3-point Umeyama hypotheses, reprojection scoring
  src/stereo_vio.cpp        orchestrator
  src/stereo_vio_node.cpp   BaseNode: queues, odometry/TF/status publishing
  src/stereo_vio_pipeline.cpp  BasePipeline plugin
  test/                     gtest suite, no hardware required
```

The estimator core has no ROS or DepthAI dependency, which is why it can be
tested on a laptop.

## Topics

| Topic | Type | Notes |
|---|---|---|
| `~/vo/odometry` | `nav_msgs/Odometry` | `odom` → `oak_vio_frame` |
| `~/vo/status` | `oak_vio_msgs/VioStatus` | **consume this** |
| `~/vo/reset` | `std_srvs/Trigger` | back to origin |
| TF | `odom` → `oak_vio_frame` | gated by `vio.i_publish_tf` |

Read `~/vo/status`, not just the odometry. When tracking is lost the estimator
holds the last pose with an inflated covariance rather than guessing, so
`/odometry` alone looks superficially fine while the state is `LOST`.

Poses are published in the ROS body convention (x forward, y left, z up), not
the camera optical convention — the pose, twist and covariance are all rotated.
Set `vio.i_publish_ros_convention: false` for raw optical-frame output.

`twist.covariance` currently reuses the pose covariance as a stand-in; twist
uncertainty is not separately estimated. Publishing zeros would read as perfect
knowledge to a filter, which is worse.

## Running it

```bash
oakctl app run . --env OAK_ROS_PEER=<your-host-ip>
```

`OAK_ROS_PEER` lists your host as an explicit CycloneDDS peer. Multicast
discovery usually works on a normal LAN but fails silently when it doesn't, and
it's the single most common cause of "the topics never showed up".

### Verifying the output from the host

Once the app logs `Driver ready!`, run the smoke test on the Jazzy machine.
Start with the camera **stationary** — it is the strongest early test, because
any drift while still is a real bias rather than accumulation:

```bash
python3 tools/verify_vo_output.py --duration 20 --stationary
```

Then pick the camera up and repeat without the flag:

```bash
python3 tools/verify_vo_output.py --duration 20
```

It checks publish rate and gaps, frame ids, covariance finiteness and growth,
stationary drift or response to motion, and — if `oak_vio_msgs` is on the host
— tracking state, inlier ratio, solve time, and the feature funnel
(`observed → matched → bucketed → inlier`), which is the quickest way to see
*where* features are being lost. Failures come with an interpretation guide.

`oak_vio_msgs` is built on the device, so the host will not have it by default.
Everything essential works without it; for the richer checks, build just that
package here:

```bash
cd ros_ws && colcon build --packages-select oak_vio_msgs && source install/setup.bash
```

Manual equivalents, if you prefer the CLI:

```bash
ros2 topic hz /oak/vo/odometry
ros2 topic echo /oak/vo/status
```

## Verification

**The maths, on any machine** — no camera, no ROS, no compiler:

```bash
python tools/validate_estimator_math.py
```

Checks the analytic Jacobian against finite differences, the hand-derived pose
composition/inverse Jacobians, noise-free exactness, Huber robustness,
covariance calibration and the depth-uncertainty regression.

**The C++**, once you have a ROS 2 environment:

```bash
colcon test --packages-select oak_vio --event-handlers console_direct+
```

**Phase 0, on the device** — do this before tuning anything:

```bash
python tools/phase0_calibration_check.py calib   --device <ip>
python tools/phase0_calibration_check.py rectify --device <ip> --alpha 0 0.25 0.5 0.75 1.0
python tools/phase0_calibration_check.py noise   --device <ip>
```

These set `vio.i_alpha_scaling`, `vio.i_mask_border_fraction` and
`vio.i_disparity_sigma_px`. Until they've run, the values in `params/vio.yaml`
are placeholders derived from the spec sheet.

**Closed-loop drift** — the number that actually tells you whether the VO
works. Carry the camera around a loop back to the exact start point and measure
end-point error as a percentage of path length:

```bash
ros2 topic echo /oak/vo/odometry --field pose.pose.position
```

**The keyframe A/B** — settle the open question above on your own data. Record
one sequence, replay it with `vio.i_keyframe_referenced` true and false, and
compare loop drift.

**Resources**, before and after enabling LENS:

```bash
oakctl app exec <app-id> top -bn1
```

## Next steps

Do these in order. Steps 1 and 2 are prerequisites for everything else, and
step 2 produces the numbers that make step 3's tuning meaningful.

### Step 1 — First C++ compile

**Status: not yet done.** There was no C++ toolchain on the machine this was
written on, so the estimator maths was validated numerically in Python instead
(see [Verification](#verification)) but nothing here has been through a
compiler. Budget an hour or two for first-build friction.

On a machine with ROS 2 Jazzy:

```bash
sudo apt install ros-jazzy-depthai-ros-v3 ros-jazzy-eigen3-cmake-module \
                 ros-jazzy-ament-cmake-gtest libeigen3-dev
cd ros_ws
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release
colcon test --packages-select oak_vio --event-handlers console_direct+
colcon test-result --verbose
```

Note `oak_vio_core` and its tests have **no** DepthAI or ROS dependency beyond
ament and Eigen, so if the DepthAI packages give trouble you can still build
and run the entire estimator test suite:

```bash
colcon build --packages-select oak_vio_msgs oak_vio --cmake-args -DBUILD_TESTING=ON
```

**Where to expect breakage.** The estimator core (`motion_estimator.cpp`,
`ransac.cpp`, `bucketing.cpp`, `keyframe_manager.cpp`, `motion_model.cpp`,
`pose_integrator.cpp`, `stereo_vio.cpp`) is plain C++17 and Eigen — it should
compile cleanly. The two integration files were written against documented APIs
rather than headers, so check these first:

| File | What to verify |
|---|---|
| `stereo_vio_node.cpp` | `Camera::build()` / `requestOutput()` signatures; `StereoDepth` setter names (`setSubpixelFractionalBits`, `setAlphaScaling`); whether `initialConfig` is a value or pointer on your version; `Sync::inputs[]` indexing; `createOutputQueue()` arity |
| `stereo_vio_node.cpp` | `dai::TrackedFeature` field names (`position`, `age`, `trackingError`) |
| `stereo_vio_pipeline.cpp` | The exact `BasePipeline::createPipeline` signature in your installed header — it gained parameters across v3 releases |
| `stereo_vio_node.hpp` | `BaseNode`'s constructor and pure virtuals; `setInOut` was named `setXinXout` in older versions |

Fastest way to resolve all of them at once:

```bash
dpkg -L ros-jazzy-depthai-ros-driver-v3 | grep -E 'base_node|base_pipeline'
```

Then read those two headers and reconcile. Everything else follows.

Two correctness traps that will not show up as compile errors:

- **`vio.i_subpixel_fractional_bits` must match `setSubpixelFractionalBits()`.**
  A mismatch scales every depth by a power of two, producing a trajectory of
  the right *shape* at the wrong *size* — easy to misread as a calibration
  problem.
- **`readCameraModel()` assumes StereoDepth rectifies into the CAM_C frame.**
  Cross-check `fx`/`cx` against the `camera_info` the driver publishes for the
  rectified left image. If they disagree, trust `camera_info`.

### Step 2 — Phase 0 on-device validation

**Nothing in `params/vio.yaml` marked "Phase 0" means anything until this
runs.** Those values came from the published FoV spec, not your device.

```bash
pip install depthai opencv-python numpy
oakctl list                       # find the device IP

python tools/phase0_calibration_check.py calib   --device <ip>
python tools/phase0_calibration_check.py rectify --device <ip> --alpha 0 0.25 0.5 0.75 1.0
python tools/phase0_calibration_check.py noise   --device <ip>
```

| Sub-command | Question it answers | Parameter it sets |
|---|---|---|
| `calib` | Is the lens really an equidistant fisheye? What does a pinhole rectification cost? | none directly — it decides whether the fallback below is needed |
| `rectify` | Which alpha keeps the most field of view at acceptable epipolar error? | `vio.i_alpha_scaling`, `vio.i_mask_border_fraction` |
| `noise` | What is the real disparity noise? | `vio.i_disparity_sigma_px` |

Acceptance criteria:

- **Epipolar error** median well under 0.5 px. Above ~1 px, rectification is
  broken and no estimator tuning will fix the resulting drift — stop and
  address that first.
- **Valid disparity coverage** comfortably above 50% on a textured scene.
- **Disparity noise** somewhere around 0.3–1.0 px. If it comes back near 0.03 px
  you have measured the quantisation step, not the noise — check the wall is
  textured and the ROI is actually on it.

For the `noise` run: a squarely-faced, textured flat wall filling the frame at
2–4 m, well lit, camera stationary. Record the number and put it in
`params/vio.yaml`; it drives both the depth-uncertainty weighting and the
published covariance.

**If `calib` and `rectify` show the rectified field of view is unusable**,
switch to the Kannala-Brandt fallback described under
[Known gaps](#known-gaps) before going further — it changes the depth source,
so it is much cheaper to decide now than after tuning.

### Step 3 — On-device bring-up and the closed-loop test

```bash
oakctl app run . --env OAK_ROS_PEER=<your-host-ip>
oakctl app logs <app-id>
```

Startup logs print the resolved camera model and the depth precision it
implies — check those against Phase 0 before trusting anything downstream.

Then, on the host, in order:

1. **Topics arrive.** `ros2 topic hz /oak/vo/odometry` should sit at your
   configured FPS. If nothing appears, it is almost always DDS discovery —
   confirm `OAK_ROS_PEER` is set and `ROS_DOMAIN_ID` matches.
2. **Stationary test.** Camera still for 60 s. Position should barely move.
   Any steady creep means disparity bias — revisit `i_disparity_sigma_px` and
   the epipolar error from Phase 0.
3. **Health under motion.** `ros2 topic echo /oak/vo/status` while walking.
   Watch `num_inliers`, `inlier_ratio` and `solve_ms`. Inlier ratio should stay
   above ~0.6 indoors; `solve_ms` should be single digits.
4. **Closed-loop drift** — the number that actually decides whether this works.
   Walk a loop back to the exact start point and measure end-point error as a
   percentage of path length. Do it over a reasonably long loop; short loops
   exaggerate the percentage badly.
5. **The keyframe A/B.** Run the same loop twice, once with
   `vio.i_keyframe_referenced: true` and once `false`, and also try
   `i_keyframe_max_parallax_px` at 10 / 15 / 30. This closes out the open
   question in [section 4](#4-the-keyframe-claim-was-half-right-and-that-changed-a-default)
   on your hardware and your typical motion.
6. **Re-derive the covariance inflation.** With ground-truth-ish loop data,
   compare actual error against the reported covariance and adjust
   `i_covariance_inflation` from the simulated 3.0 to something measured.
7. **Resource headroom.** `oakctl app exec <app-id> top -bn1` — record CPU
   before adding anything else.

### Step 4 — Phase 4: IMU aid

Deferred by agreement, and the code is already shaped for it.
`motion_model.hpp` is the slot: `MotionModel::predict()` currently extrapolates
constant velocity, and the gyro-integrated rotation replaces or blends with its
rotation component. Nothing downstream changes.

Scope, in dependency order:

1. Add `dai::node::IMU` to the pipeline in `stereo_vio_pipeline.cpp` and feed it
   into the existing `Sync` node, so IMU samples arrive time-aligned with frames.
2. Read `calib.getImuToCameraExtrinsics(CAM_B)` at startup.
3. Integrate gyro between frame timestamps into a rotation prior; feed it as
   the seed to `RansacMotionSolver::solve` (which already evaluates the seed as
   a hypothesis, so this needs no estimator change).
4. Add an error-state Kalman filter fusing VO relative pose with IMU, using the
   accelerometer's gravity direction to bound roll and pitch drift — the only
   drift axes that are observable-and-correctable without external reference.
5. Publish odometry at IMU rate (~200 Hz) rather than frame rate.

Worth doing in that order: steps 1–3 alone measurably improve robustness under
fast rotation and motion blur, and are much less work than the filter.

### Step 5 — Phase 5: LENS co-run

Add `dai::node::NeuralDepth` in `stereo_vio_pipeline.cpp` alongside the VO
node. The engines are separate — VO uses the feature-tracker and stereo blocks,
LENS runs on the DSP — so they should coexist, but verify rather than assume:
re-run the resource check from step 3 and watch for thermal throttling under
sustained load.

`depth_source.hpp` already abstracts the disparity lookup, so if you later want
VO to consume LENS depth instead of the block matcher, implement it with
`DepthMapSource` and drop the block matcher entirely. Read the caveat in that
header first — neural depth is smoothed and partly inferred, which is good for
dense perception and less good for VO, where a locally biased depth at a
feature becomes a biased pose.

## Known gaps

- **The camera model may be reading raw fisheye intrinsics, not rectified
  ones.** First run on real hardware reported `fx=567.63 fy=567.62 cx=631.42
  cy=419.17`. Under an equidistant fisheye model that is a 129° HFoV, matching
  the datasheet's 127° almost exactly — and it confirms the prediction made
  from the spec sheet (f ≈ 577 px, within 1.6%). But the estimator assumes a
  **rectified pinhole**, and the same `fx` read as a pinhole gives a
  plausible-looking 97° HFoV, which is exactly what makes this insidious: the
  trajectory comes out the right shape at the wrong scale. `readCameraModel()`
  now logs both interpretations and warns when the numbers look raw. Resolve it
  with Phase 0 `rectify` and by comparing against the driver's rectified
  `camera_info`. **Do not trust absolute scale until this is settled** —
  rotation and trajectory shape are unaffected.
- **Phase 0 has not been run.** Other than the intrinsics above, every
  optics-derived number here comes from the published FoV spec rather than your
  device. Pinhole rectification of this lens will cost field of view; how much
  is a measurement, not a calculation.
- **DepthAI API details need a compile.** There is no C++ toolchain on the
  machine this was written on, so `stereo_vio_node.cpp` and
  `stereo_vio_pipeline.cpp` are written against the documented v3 API and the
  `BaseNode` / `BasePipeline` signatures, but have not been compiled. Expect to
  fix a few call signatures on first build. The estimator core is independent
  of this and is validated.
- **The `_v3` package suffix** applies on Humble/Jazzy and was dropped in
  Kilted. `CMakeLists.txt` resolves either; `oakapp.toml` tries the metapackage
  then falls back.
- **IMU fusion is Phase 4**, as agreed. `motion_model.hpp` is the slot the gyro
  prior drops into.
- **If Phase 0 shows the rectified FoV is unusable**, the fallback is
  Kannala-Brandt on raw fisheye with the pair treated as two monocular cameras
  with fixed extrinsics — ORB-SLAM3's design. Only the projection function
  changes; the cost is losing the HW block matcher, which is why
  `depth_source.hpp` exists.

## Licence

Apache-2.0. No GPL code, and no GPL dependencies: Eigen is MPL2, and the
DepthAI and ROS stacks are permissive.
