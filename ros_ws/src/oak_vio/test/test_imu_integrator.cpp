// Gyro integration and the inter-frame rotation prior.
//
// The test that matters most here is the sign convention. A transposed prior is
// silent: RANSAC simply prefers the constant-velocity hypothesis and the IMU
// appears to make no difference, which is indistinguishable from "the IMU does
// not help on this platform". So the first test pins the direction against an
// analytic rotation rather than against the implementation.
#include "oak_vio/imu_integrator.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "oak_vio/stereo_vio.hpp"
#include "synthetic_scene.hpp"

using namespace oak_vio;        // NOLINT(build/namespaces)
using namespace oak_vio::test;  // NOLINT(build/namespaces)

namespace {

ImuParams permissiveParams() {
    ImuParams params;
    params.enabled = true;
    params.reportRateHz = 200.0;
    params.minSampleCoverage = 0.6;
    params.estimateGyroBias = false;  // most tests want the raw integration
    return params;
}

/// Feed a constant angular rate at `rateHz` over [t0, t1], inclusive of both
/// ends so windows inside it are fully covered.
void feedConstantRate(ImuIntegrator& integrator, const Vec3& omega, double t0, double t1, double rateHz, double gravityAlongZ = 9.80665) {
    const double step = 1.0 / rateHz;
    for(double t = t0; t <= t1 + 1e-12; t += step) {
        ImuSample sample;
        sample.timestampSeconds = t;
        sample.angularVelocity = omega;
        sample.hasGyro = true;
        sample.acceleration = Vec3(0.0, 0.0, gravityAlongZ);
        sample.hasAccel = true;
        integrator.addSample(sample);
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Convention
// ---------------------------------------------------------------------------

TEST(ImuIntegrator, DeltaRotationMapsPreviousFrameIntoCurrent) {
    // A camera yawing at +0.5 rad/s about its own z axis. After dt the camera
    // has rotated by +theta, so a point fixed in the world appears to have
    // rotated by -theta in the camera's coordinates. deltaRotation() returns
    // that current-from-previous transform, so it must be exp(-[omega]x dt) --
    // NOT exp(+[omega]x dt), which is the world-from-body integration.
    ImuIntegrator integrator(permissiveParams());
    const Vec3 omega(0.0, 0.0, 0.5);
    feedConstantRate(integrator, omega, 0.0, 1.0, 200.0);

    const double dt = 0.1;
    const RotationPrior prior = integrator.deltaRotation(0.2, 0.2 + dt);
    ASSERT_TRUE(prior.valid) << prior.rejection;

    const Mat3 expected = expSO3(-omega * dt);
    EXPECT_LT(logSO3(prior.deltaRotation.transpose() * expected).norm(), 1e-9);
    EXPECT_NEAR(prior.angleRad, omega.norm() * dt, 1e-9);

    // And the transposed answer is genuinely different, so the test above has
    // teeth rather than passing on a near-identity rotation.
    EXPECT_GT(logSO3(prior.deltaRotation.transpose() * expSO3(omega * dt)).norm(), 0.09);
}

TEST(ImuIntegrator, RotationScalesWithTheWindowNotWithSampleBoundaries) {
    // Windows that start and end between samples must still integrate the
    // exact elapsed time. If the implementation snapped to sample boundaries
    // the answer would quantise at 5 ms, which at 0.5 rad/s is 2.5 mrad of
    // avoidable seed error every frame.
    ImuIntegrator integrator(permissiveParams());
    const Vec3 omega(0.0, 0.3, 0.0);
    feedConstantRate(integrator, omega, 0.0, 1.0, 200.0);

    const double t0 = 0.30123;  // deliberately off-grid
    const double t1 = 0.37777;
    const RotationPrior prior = integrator.deltaRotation(t0, t1);
    ASSERT_TRUE(prior.valid) << prior.rejection;
    EXPECT_NEAR(prior.angleRad, omega.norm() * (t1 - t0), 1e-6);
}

TEST(ImuIntegrator, ComposesNonCommutingRotationsInTimeOrder) {
    // Two successive rotations about different axes do not commute, so getting
    // the multiplication order wrong shows up here and nowhere else.
    ImuIntegrator integrator(permissiveParams());
    const Vec3 first(0.8, 0.0, 0.0);
    const Vec3 second(0.0, 0.8, 0.0);
    feedConstantRate(integrator, first, 0.0, 0.2, 400.0);
    feedConstantRate(integrator, second, 0.2 + 0.0025, 0.4, 400.0);

    const RotationPrior prior = integrator.deltaRotation(0.0, 0.4);
    ASSERT_TRUE(prior.valid) << prior.rejection;

    // World-from-body composes right-multiplied in time order; the prior is its
    // transpose.
    const Mat3 worldFromBody = expSO3(first * 0.2) * expSO3(second * 0.2);
    EXPECT_LT(logSO3(prior.deltaRotation.transpose() * worldFromBody.transpose()).norm(), 2e-3);

    // The reversed composition is a different rotation, so the check above is
    // not satisfied by both orders. The gap is the commutator, of order
    // |a||b| = 0.16 * 0.16 = 0.026 rad -- an order of magnitude above the 2e-3
    // tolerance used above, so the assertion discriminates rather than passing
    // on slack.
    const Mat3 reversed = expSO3(second * 0.2) * expSO3(first * 0.2);
    EXPECT_GT(logSO3(worldFromBody.transpose() * reversed).norm(), 0.02);
}

// ---------------------------------------------------------------------------
// Extrinsics
// ---------------------------------------------------------------------------

TEST(ImuIntegrator, RotatesAngularRateIntoTheCameraFrame) {
    // IMU frame rotated +90 deg about z relative to the camera: a rate about
    // the IMU's x axis is a rate about the camera's y axis.
    ImuIntegrator integrator(permissiveParams());
    const Mat3 imuToCamera = expSO3(Vec3(0.0, 0.0, M_PI / 2.0));
    integrator.setImuToCameraRotation(imuToCamera);

    const Vec3 omegaImu(0.4, 0.0, 0.0);
    feedConstantRate(integrator, omegaImu, 0.0, 0.5, 200.0);

    const RotationPrior prior = integrator.deltaRotation(0.1, 0.2);
    ASSERT_TRUE(prior.valid) << prior.rejection;

    const Vec3 expectedAxisCamera = imuToCamera * omegaImu;  // ~ (0, 0.4, 0)
    EXPECT_NEAR(expectedAxisCamera.x(), 0.0, 1e-12);
    EXPECT_NEAR(expectedAxisCamera.y(), 0.4, 1e-12);
    EXPECT_LT(logSO3(prior.deltaRotation.transpose() * expSO3(-expectedAxisCamera * 0.1)).norm(), 1e-9);
}

TEST(ImuIntegrator, NormalisesANearOrthonormalExtrinsic) {
    // Calibration arrives as floats through a JSON round trip, so it is only
    // nearly orthonormal. Left as-is the error compounds through every step.
    ImuIntegrator integrator(permissiveParams());
    Mat3 sloppy = expSO3(Vec3(0.1, -0.2, 0.3));
    sloppy *= 1.003;  // scaled, so no longer a rotation
    integrator.setImuToCameraRotation(sloppy);

    const Mat3 stored = integrator.imuToCameraRotation();
    EXPECT_NEAR((stored * stored.transpose() - Mat3::Identity()).norm(), 0.0, 1e-12);
    EXPECT_NEAR(stored.determinant(), 1.0, 1e-12);
}

// ---------------------------------------------------------------------------
// Refusing to answer
// ---------------------------------------------------------------------------

TEST(ImuIntegrator, RejectsAWindowWithTooFewSamples) {
    // A dropped transport burst leaves a gap, and integrating over it silently
    // attributes zero rotation to the missing time. Falling back to the
    // constant-velocity seed is better than a prior that is confidently too
    // small.
    ImuIntegrator integrator(permissiveParams());
    feedConstantRate(integrator, Vec3(0.0, 0.0, 0.5), 0.0, 0.10, 200.0);

    const RotationPrior prior = integrator.deltaRotation(0.0, 0.5);
    EXPECT_FALSE(prior.valid);
    EXPECT_LT(prior.coverage, 0.6);
    EXPECT_STRNE(prior.rejection, "");
}

TEST(ImuIntegrator, HoldsTheRateAcrossSampleAlignmentAtTheWindowEdges) {
    // Frame timestamps do not land on sample timestamps, so a window normally
    // opens before its first sample and closes after its last. Dropping those
    // slivers understates every rotation by up to two sample intervals, always
    // in the same direction -- at 2.5 rad/s and 200 Hz that is 2.5% of the
    // rotation, every frame, which is a bias rather than noise.
    ImuIntegrator integrator(permissiveParams());
    const Vec3 omega(0.0, 0.0, 2.5);
    feedConstantRate(integrator, omega, 0.100, 0.300, 200.0);

    // Window straddles the buffered range by 4 ms at the start and 3 ms at the
    // end, both within one nominal sample interval.
    const double t0 = 0.096;
    const double t1 = 0.303;
    const RotationPrior prior = integrator.deltaRotation(t0, t1);
    ASSERT_TRUE(prior.valid) << prior.rejection;
    EXPECT_NEAR(prior.angleRad, omega.norm() * (t1 - t0), 1e-6);
    EXPECT_GT(prior.coverage, 0.99);
}

TEST(ImuIntegrator, RejectsAWindowWithADroppedBurstInTheMiddle) {
    // The case a naive coverage measure misses entirely: samples exist at both
    // ends of the window, so "were there samples?" says yes, while 60 ms in the
    // middle are missing and the trapezoid quietly spans the hole.
    ImuIntegrator integrator(permissiveParams());
    const Vec3 omega(0.0, 0.0, 1.0);
    feedConstantRate(integrator, omega, 0.00, 0.04, 200.0);
    feedConstantRate(integrator, omega, 0.10, 0.14, 200.0);

    const RotationPrior prior = integrator.deltaRotation(0.0, 0.14);
    EXPECT_FALSE(prior.valid);
    EXPECT_LT(prior.coverage, 0.6);
    EXPECT_GT(prior.samplesUsed, 10);  // samples were present, just not throughout
}

TEST(ImuIntegrator, RejectsAnImplausibleRotation) {
    ImuParams params = permissiveParams();
    params.maxRotationPerIntervalRad = 0.5;
    ImuIntegrator integrator(params);
    feedConstantRate(integrator, Vec3(0.0, 0.0, 20.0), 0.0, 0.5, 400.0);

    const RotationPrior prior = integrator.deltaRotation(0.1, 0.2);
    EXPECT_FALSE(prior.valid);
}

TEST(ImuIntegrator, RejectsEverythingWhenDisabled) {
    ImuParams params = permissiveParams();
    params.enabled = false;
    ImuIntegrator integrator(params);
    feedConstantRate(integrator, Vec3(0.0, 0.0, 0.5), 0.0, 0.5, 200.0);
    EXPECT_FALSE(integrator.deltaRotation(0.1, 0.2).valid);
}

TEST(ImuIntegrator, DropsOutOfOrderSamplesInsteadOfSubtractingRotation) {
    ImuIntegrator integrator(permissiveParams());
    feedConstantRate(integrator, Vec3(0.0, 0.0, 0.5), 0.0, 0.5, 200.0);
    const std::size_t before = integrator.bufferedSamples();

    ImuSample stale;
    stale.timestampSeconds = 0.25;  // already passed
    stale.angularVelocity = Vec3(0.0, 0.0, 99.0);
    stale.hasGyro = true;
    integrator.addSample(stale);

    EXPECT_EQ(integrator.bufferedSamples(), before);
    EXPECT_EQ(integrator.droppedOutOfOrder(), 1U);
}

TEST(ImuIntegrator, BoundsTheBuffer) {
    ImuParams params = permissiveParams();
    params.bufferSeconds = 0.25;
    ImuIntegrator integrator(params);
    feedConstantRate(integrator, Vec3(0.0, 0.0, 0.1), 0.0, 5.0, 200.0);

    EXPECT_LE(integrator.bufferedSpanSeconds(), 0.26);
    EXPECT_LT(integrator.bufferedSamples(), 60U);
}

// ---------------------------------------------------------------------------
// Bias
// ---------------------------------------------------------------------------

TEST(ImuIntegrator, LearnsGyroBiasWhileStationaryAndRemovesIt) {
    ImuParams params = permissiveParams();
    params.estimateGyroBias = true;
    params.biasMinSamples = 100;
    ImuIntegrator integrator(params);

    const Vec3 bias(0.004, -0.002, 0.003);  // ~0.2 deg/s, a realistic MEMS bias
    feedConstantRate(integrator, bias, 0.0, 1.0, 200.0);

    ASSERT_TRUE(integrator.biasReady());
    EXPECT_LT((integrator.gyroBias() - bias).norm(), 1e-6);

    // With the bias removed, a stationary window must report no rotation.
    const RotationPrior prior = integrator.deltaRotation(0.5, 0.6);
    ASSERT_TRUE(prior.valid) << prior.rejection;
    EXPECT_LT(prior.angleRad, 1e-6);
}

TEST(ImuIntegrator, DoesNotLearnBiasFromASmoothTurn) {
    // A constant-rate turn holds |a| at gravity, so an accelerometer-only test
    // would call it static and learn the turn rate as bias -- then subtract
    // that rate from every future frame. Both tests together must refuse.
    ImuParams params = permissiveParams();
    params.estimateGyroBias = true;
    params.biasMinSamples = 50;
    ImuIntegrator integrator(params);

    feedConstantRate(integrator, Vec3(0.0, 0.0, 0.6), 0.0, 1.0, 200.0);

    EXPECT_FALSE(integrator.biasReady());
    EXPECT_EQ(integrator.staticSamples(), 0);
    EXPECT_LT(integrator.gyroBias().norm(), 1e-12);
}

TEST(ImuIntegrator, DoesNotLearnBiasUnderLinearAcceleration) {
    // Free fall or hard acceleration: the gyro reads near zero but the
    // accelerometer is nowhere near gravity, so this is not stillness.
    ImuParams params = permissiveParams();
    params.estimateGyroBias = true;
    params.biasMinSamples = 50;
    ImuIntegrator integrator(params);

    feedConstantRate(integrator, Vec3(0.001, 0.0, 0.0), 0.0, 1.0, 200.0, /*gravityAlongZ=*/2.0);

    EXPECT_FALSE(integrator.biasReady());
    EXPECT_EQ(integrator.staticSamples(), 0);
}

TEST(ImuIntegrator, ReportsAccelerationInTheCameraFrame) {
    ImuIntegrator integrator(permissiveParams());
    const Mat3 imuToCamera = expSO3(Vec3(M_PI / 2.0, 0.0, 0.0));
    integrator.setImuToCameraRotation(imuToCamera);

    ImuSample sample;
    sample.timestampSeconds = 0.0;
    sample.acceleration = Vec3(0.0, 0.0, 9.80665);
    sample.hasAccel = true;
    integrator.addSample(sample);

    ASSERT_TRUE(integrator.hasAcceleration());
    EXPECT_LT((integrator.lastAccelerationCamera() - imuToCamera * sample.acceleration).norm(), 1e-9);
}

// ---------------------------------------------------------------------------
// Blending
// ---------------------------------------------------------------------------

TEST(BlendRotation, EndpointsAndMidpoint) {
    const Mat3 from = expSO3(Vec3(0.0, 0.0, 0.0));
    const Mat3 to = expSO3(Vec3(0.0, 0.0, 0.4));

    EXPECT_LT(logSO3(blendRotation(from, to, 0.0).transpose() * from).norm(), 1e-12);
    EXPECT_LT(logSO3(blendRotation(from, to, 1.0).transpose() * to).norm(), 1e-12);

    const Mat3 half = blendRotation(from, to, 0.5);
    EXPECT_NEAR(logSO3(half).norm(), 0.2, 1e-9);
    // Still a rotation, which elementwise interpolation would not guarantee.
    EXPECT_NEAR((half * half.transpose() - Mat3::Identity()).norm(), 0.0, 1e-12);
}

TEST(BlendRotation, ClampsOutOfRangeWeights) {
    const Mat3 from = Mat3::Identity();
    const Mat3 to = expSO3(Vec3(0.3, 0.0, 0.0));
    EXPECT_LT(logSO3(blendRotation(from, to, -1.0).transpose() * from).norm(), 1e-12);
    EXPECT_LT(logSO3(blendRotation(from, to, 4.0).transpose() * to).norm(), 1e-12);
}

// ---------------------------------------------------------------------------
// End to end: does the prior actually buy anything?
// ---------------------------------------------------------------------------

namespace {

/// A yaw shake about the camera's y axis: theta(t) = A sin(2 pi f t), plus a
/// little forward translation so translation stays observable.
///
/// A shake rather than a sweep, for a mundane but important reason: a sustained
/// sweep rotates the synthetic landmarks out of the field of view within a few
/// frames, after which tracking is lost for want of a scene rather than for
/// want of a prior, and the test measures nothing. A shake reaches the same
/// per-frame rotation while the total stays inside the FoV — and it is also
/// what a legged platform's camera actually does.
struct YawShake {
    double amplitudeRad{0.265};  // ~15 deg, peak rate ~2.5 rad/s at 1.5 Hz
    double frequencyHz{1.5};
    double forwardMetresPerSec{0.3};
    double fps{10.0};  // the rate this rig runs at, FSYNC-limited

    [[nodiscard]] double angleAt(double t) const { return amplitudeRad * std::sin(2.0 * M_PI * frequencyHz * t); }
    /// Fixed-axis rotation, so the body-frame rate is just the derivative.
    [[nodiscard]] Vec3 bodyRateAt(double t) const {
        return Vec3(0.0, amplitudeRad * 2.0 * M_PI * frequencyHz * std::cos(2.0 * M_PI * frequencyHz * t), 0.0);
    }
    [[nodiscard]] Pose poseAt(double t) const {
        Pose pose;
        pose.R = expSO3(Vec3(0.0, angleAt(t), 0.0));
        pose.t = Vec3(0.0, 0.0, forwardMetresPerSec * t);
        return pose;
    }
};

}  // namespace

TEST(GyroPrior, IsProducedConsumedAndDoesNotDegradeTracking) {
    // What this test can and cannot establish, stated plainly: it pins the
    // plumbing (a prior is produced every frame and reaches the estimator) and
    // non-regression (with it, tracking and rotation error are no worse). It
    // does NOT try to prove the prior improves accuracy. On synthetic data with
    // 300 clean corners RANSAC succeeds from any seed, so a simulated win would
    // be an artefact of how hard the scene was made rather than evidence. The
    // A/B that settles the benefit is the one on hardware, under real motion
    // blur — see the Phase 4 section of the README.
    const YawShake shake;
    const RectifiedCamera camera = makeTestCamera();
    const int width = 640;
    const int height = 400;
    const double frameInterval = 1.0 / shake.fps;
    const int frames = 20;

    struct Outcome {
        int tracked{0};
        int priorsOffered{0};
        int priorsUsed{0};
        double worstRotationError{0.0};
    };

    const auto runTrial = [&](bool useGyro) {
        SyntheticScene scene(camera, width, height, 4242);
        scene.generatePoints(300, 2.0, 25.0);

        VioParams params;
        params.keyframeReferenced = true;
        params.gyroPriorWeight = 1.0;
        StereoVio vio(camera, params, width, height);

        ImuParams imuParams;
        imuParams.enabled = true;
        imuParams.reportRateHz = 200.0;
        imuParams.estimateGyroBias = false;
        ImuIntegrator integrator(imuParams);

        Outcome outcome;
        double maxPerFrameRotation = 0.0;

        for(int frame = 0; frame < frames; ++frame) {
            const double stamp = static_cast<double>(frame) * frameInterval;

            // Gyro arrives continuously between frames, at its own rate.
            const double imuStep = 1.0 / imuParams.reportRateHz;
            for(double t = std::max(0.0, stamp - frameInterval) + imuStep; t <= stamp + 1e-12; t += imuStep) {
                ImuSample sample;
                sample.timestampSeconds = t;
                sample.angularVelocity = shake.bodyRateAt(t);
                sample.hasGyro = true;
                sample.acceleration = Vec3(0.0, 0.0, 9.80665);
                sample.hasAccel = true;
                integrator.addSample(sample);
            }

            const Pose truth = shake.poseAt(stamp);
            const std::vector<Observation> observations = scene.observe(truth, 0.3, 0.3);

            RotationPrior prior;
            if(useGyro && frame > 0) {
                prior = integrator.deltaRotation(stamp - frameInterval, stamp);
                if(prior.valid) {
                    ++outcome.priorsOffered;
                    maxPerFrameRotation = std::max(maxPerFrameRotation, prior.angleRad);

                    // The prior must agree with the truth it was generated
                    // from, or the rest of this test is measuring nothing.
                    const Mat3 truthDelta = shake.poseAt(stamp).R.transpose() * shake.poseAt(stamp - frameInterval).R;
                    EXPECT_LT(logSO3(prior.deltaRotation.transpose() * truthDelta).norm(), 2e-3);
                }
            }

            const VioFrameResult result = vio.processFrame(observations, stamp, prior);
            if(result.usedGyroPrior) {
                ++outcome.priorsUsed;
            }
            if(result.state == TrackingState::Tracking) {
                ++outcome.tracked;
                outcome.worstRotationError = std::max(outcome.worstRotationError, rotationError(result.worldFromCamera, truth));
            }
        }

        // Confirm the trajectory really does exercise fast rotation, so this is
        // not quietly testing a near-stationary camera.
        if(useGyro) {
            EXPECT_GT(maxPerFrameRotation, 0.2) << "per-frame rotation too small for this test to be meaningful";
        }
        return outcome;
    };

    const Outcome withGyro = runTrial(true);
    const Outcome withoutGyro = runTrial(false);

    // Plumbing: a prior on every frame after the first, and every valid prior
    // reaching the estimator's seed.
    EXPECT_EQ(withGyro.priorsOffered, frames - 1);
    EXPECT_EQ(withGyro.priorsUsed, withGyro.priorsOffered);
    EXPECT_EQ(withoutGyro.priorsUsed, 0);

    // Non-regression. A correct prior can only help RANSAC, since the seed
    // competes as one hypothesis among many and is discarded if it loses.
    EXPECT_GE(withGyro.tracked, withoutGyro.tracked);
    EXPECT_LE(withGyro.worstRotationError, withoutGyro.worstRotationError + 1e-6);
}

TEST(GyroPrior, AWrongPriorDoesNotDestroyTracking) {
    // The failure mode to guard against: a bad extrinsic, or a transposed
    // convention, making the prior confidently wrong. RANSAC evaluates the seed
    // as one hypothesis among many, so tracking should survive on the strength
    // of the data.
    const RectifiedCamera camera = makeTestCamera();
    const int width = 640;
    const int height = 400;
    SyntheticScene scene(camera, width, height, 99);
    scene.generatePoints(300, 2.0, 25.0);

    VioParams params;
    params.gyroPriorWeight = 1.0;
    StereoVio vio(camera, params, width, height);

    int tracked = 0;
    for(int frame = 0; frame < 10; ++frame) {
        const double stamp = 0.1 * frame;
        Pose truth;
        truth.t = Vec3(0.0, 0.0, 0.05 * frame);  // gentle forward motion only

        RotationPrior nonsense;
        nonsense.valid = true;
        nonsense.deltaRotation = expSO3(Vec3(0.3, -0.25, 0.2));  // ~26 deg of invented rotation
        nonsense.angleRad = logSO3(nonsense.deltaRotation).norm();

        const VioFrameResult result = vio.processFrame(scene.observe(truth, 0.3, 0.3), stamp, nonsense);
        if(result.state == TrackingState::Tracking) {
            ++tracked;
            EXPECT_LT(rotationError(result.worldFromCamera, truth), 0.02);
        }
    }
    EXPECT_GE(tracked, 8);
}
