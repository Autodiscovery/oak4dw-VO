#!/usr/bin/env python3
"""Host-side smoke test for the on-device VO.

Subscribes to what the camera publishes, collects for a fixed window, and
reports pass/fail against concrete criteria. Run it on the Jazzy machine with
the camera connected and the app running.

    # Camera should be STATIONARY for this one -- it is the strongest early
    # test, because any drift while still is a real bias, not accumulation.
    python3 tools/verify_vo_output.py --duration 20 --stationary

    # Then pick the camera up and move it around.
    python3 tools/verify_vo_output.py --duration 20

Only nav_msgs is required. If oak_vio_msgs is also on the host it reports the
richer estimator health as well -- see --help for how to build it.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
except ImportError:
    sys.exit("rclpy not found. Source your ROS 2 setup first:\n"
             "    source /opt/ros/jazzy/setup.bash")

from nav_msgs.msg import Odometry

# oak_vio_msgs is built on the DEVICE, so the host will not have it unless you
# build it too. Everything essential works without it.
try:
    from oak_vio_msgs.msg import VioStatus
    HAVE_STATUS_MSGS = True
except ImportError:
    VioStatus = None
    HAVE_STATUS_MSGS = False

STATE_NAMES = {0: "INITIALISING", 1: "TRACKING", 2: "LOW_INLIERS", 3: "LOST"}


class Collector(Node):
    def __init__(self, namespace: str):
        super().__init__("oak_vio_verifier")
        self.odom: list[tuple[float, Odometry]] = []
        self.status: list = []

        self.create_subscription(Odometry, f"{namespace}/vo/odometry", self._on_odom, 50)
        if HAVE_STATUS_MSGS:
            self.create_subscription(VioStatus, f"{namespace}/vo/status", self._on_status, 50)

    def _on_odom(self, msg: Odometry) -> None:
        self.odom.append((time.time(), msg))

    def _on_status(self, msg) -> None:
        self.status.append(msg)


RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool | None, name: str, detail: str = "") -> None:
    """ok=None means informational only, never fails the run."""
    tag = "INFO" if ok is None else ("PASS" if ok else "FAIL")
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    if ok is not None:
        RESULTS.append((ok, name, detail))


def position(msg: Odometry) -> tuple[float, float, float]:
    p = msg.pose.pose.position
    return p.x, p.y, p.z


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="To get the richer status checks, build the message package on this host:\n"
               "    git clone https://github.com/Autodiscovery/oak4dw-VO.git\n"
               "    cd oak4dw-VO/ros_ws && colcon build --packages-select oak_vio_msgs\n"
               "    source install/setup.bash\n")
    parser.add_argument("--namespace", default="/oak", help="Driver node namespace (default: /oak)")
    parser.add_argument("--duration", type=float, default=20.0, help="Collection window, seconds")
    # 10, not 30. This device is in external FSYNC slave mode, so its rate is set
    # by an external sync source and vio.i_fps has no effect -- see the frame-rate
    # section of the README. Defaulting to the configured-but-unreachable 30 made
    # every run report a failure for a known, documented, software-unfixable
    # condition, which is the fastest way to teach someone to ignore failures.
    parser.add_argument("--expected-fps", type=float, default=10.0,
                        help="Rate to expect. Match this to the camera's ACTUAL rate (see the app's "
                             "periodic log), not to vio.i_fps, which the sensor may ignore.")
    parser.add_argument("--stationary", action="store_true",
                        help="Camera is not moving: assert drift stays near zero")
    parser.add_argument("--max-static-drift", type=float, default=0.05,
                        help="Metres of drift tolerated over the window when --stationary")
    args = parser.parse_args()

    print("=" * 70)
    print("OAK 4 D W -- on-device VO output check")
    print("=" * 70)
    if not HAVE_STATUS_MSGS:
        print("\nNote: oak_vio_msgs not found on this host, so estimator health checks")
        print("      are skipped. Odometry checks below are unaffected. See --help.")

    rclpy.init()
    node = Collector(args.namespace)

    # --- Discovery ------------------------------------------------------
    print(f"\nWaiting for {args.namespace}/vo/odometry ...")
    deadline = time.time() + 10.0
    while time.time() < deadline and not node.odom:
        rclpy.spin_once(node, timeout_sec=0.1)

    if not node.odom:
        check(False, "odometry topic is publishing", "nothing received in 10 s")
        print("\nNothing arrived. In order of likelihood:")
        print("  1. DDS discovery. Confirm the app was started with")
        print("     OAK_ROS_PEER=<this host's IP>, and that ROS_DOMAIN_ID matches.")
        print("  2. Wrong namespace. Try:  ros2 topic list | grep vo")
        print("  3. RMW mismatch. The app uses CycloneDDS; this host must too:")
        print("     export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print(f"Collecting for {args.duration:.0f} s"
          + (" (keep the camera STILL)" if args.stationary else " (move the camera around)") + " ...")
    end = time.time() + args.duration
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)

    node.destroy_node()
    rclpy.shutdown()

    # --- Rate -----------------------------------------------------------
    print("\nPublishing")
    n = len(node.odom)
    wall = node.odom[-1][0] - node.odom[0][0]
    rate = (n - 1) / wall if wall > 0 else 0.0
    check(n > 10, "received a usable number of messages", f"{n} messages in {wall:.1f} s")
    rate_ok = rate > 0.5 * args.expected_fps
    check(rate_ok, "rate is close to the configured FPS",
          f"{rate:.1f} Hz vs {args.expected_fps:.0f} configured")
    if not rate_ok:
        print("         Compare against the 'camera N Hz' figure in the app's periodic log. If the")
        print("         camera itself is below the requested rate then the sensor is the limit, not")
        print("         the pipeline -- widening the sync window will not help. If the camera is at")
        print("         the requested rate but this is not, then Sync is dropping pairs whose")
        print("         timestamps skew outside its window.")

    # Gaps point at dropped frames or a stalled estimator.
    gaps = [b[0] - a[0] for a, b in zip(node.odom, node.odom[1:])]
    if gaps:
        worst = max(gaps)
        check(worst < 1.0, "no long gaps in the stream", f"largest gap {worst * 1000:.0f} ms")

    # --- Frames and covariance -------------------------------------------
    print("\nMessage contents")
    first = node.odom[0][1]
    check(bool(first.header.frame_id), "odometry has a frame_id", f"'{first.header.frame_id}'")
    check(bool(first.child_frame_id), "odometry has a child_frame_id", f"'{first.child_frame_id}'")

    last = node.odom[-1][1]
    diag = [last.pose.covariance[i * 6 + i] for i in range(6)]
    check(all(math.isfinite(v) for v in diag), "pose covariance is finite",
          "diag = [" + ", ".join(f"{v:.2e}" for v in diag) + "]")
    check(all(v >= 0.0 for v in diag), "pose covariance diagonal is non-negative")
    # Covariance must grow as VO drifts; a permanently zero diagonal means the
    # integrator is not propagating uncertainty at all.
    first_trace = sum(node.odom[0][1].pose.covariance[i * 6 + i] for i in range(6))
    last_trace = sum(diag)
    check(last_trace >= first_trace, "covariance accumulates over time",
          f"trace {first_trace:.2e} -> {last_trace:.2e}")

    # --- Motion -----------------------------------------------------------
    print("\nMotion")
    start_p = position(node.odom[0][1])
    end_p = position(last)
    displacement = math.dist(start_p, end_p)
    path = sum(math.dist(position(a[1]), position(b[1])) for a, b in zip(node.odom, node.odom[1:]))
    print(f"     net displacement {displacement * 100:.1f} cm, path length {path * 100:.1f} cm")
    print(f"     final position  x={end_p[0]:+.3f}  y={end_p[1]:+.3f}  z={end_p[2]:+.3f}  (m)")

    # A pose that never moves is only meaningful if the estimator was actually
    # tracking. If it never left INITIALISING, "zero drift" is vacuous -- the
    # pose is pinned at the origin because nothing is being solved. Report that
    # honestly instead of banking a free pass.
    ever_tracked = True
    if HAVE_STATUS_MSGS and node.status:
        ever_tracked = any(m.state in (1, 2) for m in node.status)

    if not ever_tracked:
        check(None, "motion checks skipped",
              "the estimator never reached TRACKING, so pose is pinned at the origin "
              "and drift figures are meaningless")
    elif args.stationary:
        check(displacement < args.max_static_drift,
              f"stationary drift under {args.max_static_drift * 100:.0f} cm",
              f"{displacement * 100:.2f} cm over {wall:.0f} s")
        check(path < args.max_static_drift * 5,
              "no jitter accumulation while still", f"path {path * 100:.2f} cm")
    else:
        check(path > 0.02, "the estimator responds to motion",
              f"path {path * 100:.1f} cm -- if ~0, features are being tracked but no motion is solved")

    # --- Estimator health --------------------------------------------------
    if HAVE_STATUS_MSGS and node.status:
        print("\nEstimator health")
        counts: dict[int, int] = {}
        for m in node.status:
            counts[m.state] = counts.get(m.state, 0) + 1
        total = len(node.status)
        summary = ", ".join(f"{STATE_NAMES.get(s, s)} {100 * c / total:.0f}%"
                            for s, c in sorted(counts.items()))
        tracking_frac = counts.get(1, 0) / total
        check(tracking_frac > 0.8, "mostly in TRACKING state", summary)

        ratios = [m.inlier_ratio for m in node.status if m.num_selected > 0]
        if ratios:
            check(statistics.median(ratios) > 0.5, "healthy inlier ratio",
                  f"median {statistics.median(ratios):.2f}")

        inliers = [m.num_inliers for m in node.status]
        check(statistics.median(inliers) > 30, "enough inliers per frame",
              f"median {statistics.median(inliers):.0f}")

        obs = [m.num_observations for m in node.status]
        corr = [m.num_correspondences for m in node.status]
        sel = [m.num_selected for m in node.status]
        check(statistics.median(obs) > 0,
              "features arrive with valid disparity",
              f"funnel: {statistics.median(obs):.0f} -> {statistics.median(corr):.0f} -> "
              f"{statistics.median(sel):.0f} -> {statistics.median(inliers):.0f} "
              f"(observed -> matched -> bucketed -> inlier)")
        if statistics.median(obs) == 0:
            print("         Zero observations means either the feature tracker produced no")
            print("         corners, or every disparity lookup failed. The app log prints a")
            print("         throttled diagnostic that distinguishes the two -- check it with")
            print("         'oakctl app logs <app-id>'.")

        solve = sorted(m.solve_ms for m in node.status)
        p95 = solve[int(0.95 * (len(solve) - 1))]
        check(p95 < 15.0, "solve time leaves headroom for LENS",
              f"median {statistics.median(solve):.2f} ms, p95 {p95:.2f} ms")

        kf = sum(1 for m in node.status if m.keyframe_promoted)
        par = [m.median_parallax_px for m in node.status]
        check(None, "keyframe rate", f"{kf} promotions in {total} frames "
              f"(~{total / max(kf, 1):.0f} frame spans), median parallax "
              f"{statistics.median(par):.1f} px")

        notes = {m.note for m in node.status if m.note}
        if notes:
            check(None, "notes seen", "; ".join(sorted(notes)[:5]))

    # --- Verdict -----------------------------------------------------------
    failed = [name for ok, name, _ in RESULTS if not ok]
    print("\n" + "=" * 70)
    if failed:
        print(f"FAILED ({len(failed)}/{len(RESULTS)}): " + "; ".join(failed))
        print("\nInterpreting failures:")
        print("  Stationary drift        -> disparity bias. Run Phase 0 'noise' and check")
        print("                             epipolar error from Phase 0 'rectify'.")
        print("  Low inlier ratio        -> scene texture, exposure, or a rectification problem.")
        print("  Mostly LOST             -> too few features with valid disparity. Check the")
        print("                             feature funnel above to see where they are dropping.")
        print("  No response to motion   -> solves are being rejected; read the notes line.")
        return 1

    print(f"All {len(RESULTS)} checks passed.")
    if not args.stationary:
        print("\nNext: the closed-loop drift test. Walk a loop back to the exact start point,")
        print("then compare the final position printed above against the start. Drift as a")
        print("percentage of path length is the number that actually matters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
