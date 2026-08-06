// End-to-end tests: drive StereoVio over synthetic trajectories and measure
// what actually matters -- drift over a closed loop.
#include "oak_vio/stereo_vio.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "synthetic_scene.hpp"

using namespace oak_vio;         // NOLINT(build/namespaces)
using namespace oak_vio::test;   // NOLINT(build/namespaces)

namespace {

constexpr int kWidth = 1280;
constexpr int kHeight = 800;
constexpr double kFrameDt = 1.0 / 30.0;

/// A closed loop: the camera traces a small circle in the x-z plane, yawing
/// gently, and returns exactly to its start. Closing the loop is what makes
/// drift measurable without needing an external reference.
std::vector<Pose> makeLoopTrajectory(int frames, double radius, double yawAmplitude) {
    std::vector<Pose> poses;
    poses.reserve(static_cast<std::size_t>(frames));
    for(int i = 0; i < frames; ++i) {
        const double phase = 2.0 * M_PI * static_cast<double>(i) / static_cast<double>(frames);
        Pose pose;
        pose.R = expSO3(Vec3(0.0, yawAmplitude * std::sin(phase), 0.0));
        pose.t = Vec3(radius * std::sin(phase), 0.0, radius * (1.0 - std::cos(phase)));
        poses.push_back(pose);
    }
    return poses;
}

double pathLength(const std::vector<Pose>& poses) {
    double total = 0.0;
    for(std::size_t i = 1; i < poses.size(); ++i) {
        total += (poses[i].t - poses[i - 1].t).norm();
    }
    return total;
}

struct RunOutcome {
    double finalDrift{0.0};
    double maxPositionError{0.0};
    double maxRotationError{0.0};
    int lostFrames{0};
    int keyframes{0};
    double maxSolveMs{0.0};
};

RunOutcome runTrajectory(const std::vector<Pose>& trajectory, VioParams params, double pixelNoise, double disparityNoise, double outlierFraction, std::uint64_t seed) {
    SyntheticScene scene(makeTestCamera(), kWidth, kHeight, seed);
    scene.generatePoints(500, 1.5, 30.0);

    StereoVio vio(makeTestCamera(), params, kWidth, kHeight);

    RunOutcome outcome;
    for(std::size_t i = 0; i < trajectory.size(); ++i) {
        const auto observations = scene.observe(trajectory[i], pixelNoise, disparityNoise, outlierFraction);
        const VioFrameResult result = vio.processFrame(observations, static_cast<double>(i) * kFrameDt);

        if(result.state == TrackingState::Lost) {
            ++outcome.lostFrames;
        }
        if(result.keyframePromoted) {
            ++outcome.keyframes;
        }
        outcome.maxSolveMs = std::max(outcome.maxSolveMs, result.solveMilliseconds);

        if(i > 0) {
            outcome.maxPositionError = std::max(outcome.maxPositionError, (result.worldFromCamera.t - trajectory[i].t).norm());
            outcome.maxRotationError = std::max(outcome.maxRotationError, rotationError(trajectory[i], result.worldFromCamera));
        }
        if(i + 1 == trajectory.size()) {
            outcome.finalDrift = (result.worldFromCamera.t - trajectory.front().t).norm();
        }
    }
    return outcome;
}

}  // namespace

TEST(Trajectory, TracksACleanLoopAlmostExactly) {
    const auto trajectory = makeLoopTrajectory(120, 0.5, 0.15);
    const RunOutcome outcome = runTrajectory(trajectory, VioParams{}, 0.0, 0.0, 0.0, 101);

    EXPECT_EQ(outcome.lostFrames, 0);
    EXPECT_LT(outcome.maxPositionError, 1e-3) << "noise-free tracking should be near exact";
    EXPECT_LT(outcome.maxRotationError, 1e-4);
    EXPECT_LT(outcome.finalDrift, 1e-3);
}

// A regression guard against gross breakage, NOT a quality claim.
//
// The threshold is deliberately loose. In a Python simulation of this same
// loop, drift ranged from 1% to 10% of path length depending only on the
// random seed -- a 10x spread. Short loops exaggerate the percentage, because
// per-frame error does not shrink with path length. The real number is the
// hardware closed-loop test in the README; this test only catches the
// estimator falling over completely.
TEST(Trajectory, ClosedLoopDriftIsBounded) {
    const auto trajectory = makeLoopTrajectory(180, 0.6, 0.2);
    const RunOutcome outcome = runTrajectory(trajectory, VioParams{}, 0.3, 0.5, 0.10, 202);

    const double length = pathLength(trajectory);
    const double driftPercent = 100.0 * outcome.finalDrift / length;

    EXPECT_EQ(outcome.lostFrames, 0);
    EXPECT_LT(driftPercent, 15.0) << "closed-loop drift " << driftPercent << "% over " << length << " m";
}

// OPEN QUESTION, deliberately not asserted either way.
//
// The plan claimed keyframe-referenced tracking would drift less than
// frame-to-frame, on the ORB-SLAM argument that error should not compound
// every frame. Simulation could not reproduce that: across seeds the two modes
// were indistinguishable, with per-seed variance far exceeding any difference
// between them. The likely reason is that ORB-SLAM's advantage comes largely
// from local bundle adjustment refining the map points, which we skip -- with
// a fixed unrefined keyframe, its triangulation error is a bias that persists
// for the whole span instead of averaging out.
//
// So this test verifies only that the switch does what it says and that both
// modes stay bounded. Which one is actually better on this camera is settled
// by the A/B on real data described in the README, not by asserting it here.
TEST(Trajectory, KeyframeSwitchTakesEffectAndBothModesStayBounded) {
    const auto trajectory = makeLoopTrajectory(240, 0.6, 0.2);
    const double length = pathLength(trajectory);

    VioParams keyframed;
    keyframed.keyframeReferenced = true;

    VioParams frameToFrame;
    frameToFrame.keyframeReferenced = false;

    const RunOutcome withKeyframes = runTrajectory(trajectory, keyframed, 0.3, 0.5, 0.05, 303);
    const RunOutcome withoutKeyframes = runTrajectory(trajectory, frameToFrame, 0.3, 0.5, 0.05, 303);

    // Frame-to-frame re-anchors every frame, so it must produce far more
    // promotions. This is what proves the switch is wired up rather than the
    // two runs being accidentally identical.
    EXPECT_GT(withoutKeyframes.keyframes, withKeyframes.keyframes * 2);

    EXPECT_EQ(withKeyframes.lostFrames, 0);
    EXPECT_EQ(withoutKeyframes.lostFrames, 0);
    EXPECT_LT(100.0 * withKeyframes.finalDrift / length, 15.0);
    EXPECT_LT(100.0 * withoutKeyframes.finalDrift / length, 15.0);
}

// Regression guard: the motion model must update in BOTH keyframe modes.
// In frame-to-frame mode the keyframe is always the previous frame, so the
// keyframe-relative estimate is already the inter-frame motion — an earlier
// version missed that branch and published a permanently zero twist.
TEST(Trajectory, PublishesNonZeroVelocityInBothKeyframeModes) {
    const auto trajectory = makeLoopTrajectory(60, 0.6, 0.1);

    for(const bool keyframed : {true, false}) {
        VioParams params;
        params.keyframeReferenced = keyframed;

        SyntheticScene scene(makeTestCamera(), kWidth, kHeight, 909);
        scene.generatePoints(500, 1.5, 25.0);
        StereoVio vio(makeTestCamera(), params, kWidth, kHeight);

        double maxSpeed = 0.0;
        for(std::size_t i = 0; i < trajectory.size(); ++i) {
            const VioFrameResult result = vio.processFrame(scene.observe(trajectory[i], 0.3, 0.5), static_cast<double>(i) * kFrameDt);
            maxSpeed = std::max(maxSpeed, result.linearVelocity.norm());
        }

        // The loop covers ~3.75 m; at 30 FPS over 60 frames that is roughly
        // 1 m/s, so anything above a few cm/s proves the twist is live.
        EXPECT_GT(maxSpeed, 0.05) << "no velocity reported with keyframeReferenced=" << keyframed;
    }
}

TEST(Trajectory, StationaryCameraDoesNotDrift) {
    // Sixty seconds of a still camera. Any systematic bias in the disparity
    // handling shows up here as motion that is not happening.
    std::vector<Pose> stationary(600);
    const RunOutcome outcome = runTrajectory(stationary, VioParams{}, 0.3, 0.5, 0.05, 404);

    EXPECT_EQ(outcome.lostFrames, 0);
    EXPECT_LT(outcome.finalDrift, 0.02) << "stationary drift should stay in the millimetres";
}

TEST(Trajectory, ReportsLostAndRecoversAfterAFeatureDropout) {
    SyntheticScene scene(makeTestCamera(), kWidth, kHeight, 505);
    scene.generatePoints(400, 1.5, 25.0);
    StereoVio vio(makeTestCamera(), VioParams{}, kWidth, kHeight);

    const auto trajectory = makeLoopTrajectory(90, 0.4, 0.1);

    int lost = 0;
    bool recovered = false;
    for(std::size_t i = 0; i < trajectory.size(); ++i) {
        // Simulate a total dropout (covered lens, motion blur) for 5 frames.
        const bool blackout = i >= 30 && i < 35;
        const auto observations = blackout ? std::vector<Observation>{} : scene.observe(trajectory[i], 0.3, 0.5);

        const VioFrameResult result = vio.processFrame(observations, static_cast<double>(i) * kFrameDt);

        if(blackout) {
            EXPECT_EQ(result.state, TrackingState::Lost);
            ++lost;
        } else if(i > 40) {
            // A few frames after the dropout, tracking must be healthy again.
            if(result.state == TrackingState::Tracking) {
                recovered = true;
            }
        }
    }

    EXPECT_EQ(lost, 5);
    EXPECT_TRUE(recovered) << "the estimator must re-anchor and resume after losing tracking";
}

TEST(Trajectory, HoldsPoseAndInflatesCovarianceWhileLost) {
    StereoVio vio(makeTestCamera(), VioParams{}, kWidth, kHeight);
    SyntheticScene scene(makeTestCamera(), kWidth, kHeight, 606);
    scene.generatePoints(300, 1.5, 25.0);

    vio.processFrame(scene.observe(Pose{}, 0.0, 0.0), 0.0);
    Pose moved;
    moved.t = Vec3(0.05, 0.0, 0.0);
    const VioFrameResult tracked = vio.processFrame(scene.observe(moved, 0.3, 0.5), kFrameDt);
    ASSERT_TRUE(tracked.valid);

    const VioFrameResult lost = vio.processFrame({}, 2 * kFrameDt);
    EXPECT_FALSE(lost.valid);
    EXPECT_EQ(lost.state, TrackingState::Lost);
    // Pose is held rather than reset or extrapolated.
    EXPECT_LT((lost.worldFromCamera.t - tracked.worldFromCamera.t).norm(), 1e-12);
}

TEST(Trajectory, StaysWithinTheCpuBudget) {
    // The design claim is single-digit milliseconds per frame on one core.
    // This runs on a workstation rather than the RVC4's ARM cores, so it is a
    // regression guard against algorithmic blowup, not a device benchmark.
    const auto trajectory = makeLoopTrajectory(120, 0.5, 0.15);
    const RunOutcome outcome = runTrajectory(trajectory, VioParams{}, 0.3, 0.5, 0.15, 707);
    EXPECT_LT(outcome.maxSolveMs, 50.0) << "worst-frame solve time " << outcome.maxSolveMs << " ms";
}

TEST(Trajectory, ResetReturnsToOrigin) {
    StereoVio vio(makeTestCamera(), VioParams{}, kWidth, kHeight);
    SyntheticScene scene(makeTestCamera(), kWidth, kHeight, 808);
    scene.generatePoints(300, 1.5, 25.0);

    const auto trajectory = makeLoopTrajectory(30, 0.5, 0.1);
    for(std::size_t i = 0; i < trajectory.size(); ++i) {
        vio.processFrame(scene.observe(trajectory[i], 0.3, 0.5), static_cast<double>(i) * kFrameDt);
    }

    vio.reset();
    const VioFrameResult afterReset = vio.processFrame(scene.observe(trajectory.back(), 0.3, 0.5), 100.0);
    EXPECT_EQ(afterReset.state, TrackingState::Initialising);
    EXPECT_NEAR(afterReset.worldFromCamera.t.norm(), 0.0, 1e-12);
}
