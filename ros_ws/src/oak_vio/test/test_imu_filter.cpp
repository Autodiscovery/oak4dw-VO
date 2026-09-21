// The Phase 4 error-state filter.
//
// The filter is off by default and unvalidated on hardware, so these tests are
// doing a specific job: pinning the parts whose correctness is a matter of
// arithmetic rather than tuning. Propagation must not invent motion, the
// gravity update must correct roll and pitch and must NOT pretend to correct
// yaw, the covariance must stay a covariance, and the biases must be
// observable. Anything beyond that — whether fusing helps the loop-drift number
// — is a hardware question these tests cannot answer.
#include "oak_vio/imu_filter.hpp"

#include <gtest/gtest.h>

#include <Eigen/Eigenvalues>

#include <cmath>

using namespace oak_vio;  // NOLINT(build/namespaces)

namespace {

constexpr double kGravity = 9.80665;

ImuFilterParams enabledParams() {
    ImuFilterParams params;
    params.enabled = true;
    return params;
}

/// Attitude error of the filter against truth, as a rotation vector in the
/// body frame. Near level, x and y are the gravity-observable axes (roll and
/// pitch) and z is yaw, which gravity cannot see.
Vec3 attitudeError(const ImuFilter& filter, const Mat3& truth) {
    return logSO3(filter.state().worldFromBody.R.transpose() * truth);
}

/// The accelerometer reading a static platform at attitude `worldFromBody`
/// produces: gravity, rotated into the body frame.
Vec3 staticAccelerometer(const Mat3& worldFromBody, double gravity = kGravity) {
    return worldFromBody.transpose() * Vec3(0.0, 0.0, gravity);
}

}  // namespace

// ---------------------------------------------------------------------------
// Propagation
// ---------------------------------------------------------------------------

TEST(ImuFilter, StaticPlatformDoesNotDriftUnderPropagationAlone) {
    // Gravity has to cancel exactly against the accelerometer reading. If the
    // sign of the gravity vector is wrong the filter accelerates at 2g and the
    // error is unmissable; if the frame is wrong it drifts sideways, which is
    // not.
    ImuFilter filter(enabledParams());
    filter.initialise(Pose{}, 0.0);

    const double step = 1.0 / 200.0;
    for(int i = 1; i <= 400; ++i) {  // two seconds
        ASSERT_TRUE(filter.propagate(Vec3::Zero(), staticAccelerometer(Mat3::Identity()), step * i));
    }

    EXPECT_LT(filter.state().velocity.norm(), 1e-9);
    EXPECT_LT(filter.state().worldFromBody.t.norm(), 1e-9);
    EXPECT_LT(logSO3(filter.state().worldFromBody.R).norm(), 1e-12);
    EXPECT_EQ(filter.propagations(), 400U);
}

TEST(ImuFilter, PropagatesRotationAtTheGyroRate) {
    ImuFilter filter(enabledParams());
    filter.initialise(Pose{}, 0.0);

    const Vec3 omega(0.0, 0.0, 0.4);
    const double step = 1.0 / 200.0;
    const int steps = 100;  // half a second
    for(int i = 1; i <= steps; ++i) {
        // Accelerometer follows the attitude, so the platform is turning in
        // place rather than tumbling.
        const Vec3 accel = staticAccelerometer(filter.state().worldFromBody.R);
        ASSERT_TRUE(filter.propagate(omega, accel, step * i));
    }

    const double expectedAngle = omega.norm() * step * steps;
    EXPECT_NEAR(logSO3(filter.state().worldFromBody.R).norm(), expectedAngle, 1e-9);
}

TEST(ImuFilter, RefusesBackwardsAndOverlongSteps) {
    ImuFilterParams params = enabledParams();
    params.maxPropagationStepSeconds = 0.02;
    ImuFilter filter(params);
    filter.initialise(Pose{}, 1.0);

    EXPECT_FALSE(filter.propagate(Vec3::Zero(), staticAccelerometer(Mat3::Identity()), 0.99));  // backwards
    EXPECT_FALSE(filter.propagate(Vec3::Zero(), staticAccelerometer(Mat3::Identity()), 1.0));   // no elapsed time
    EXPECT_FALSE(filter.propagate(Vec3::Zero(), staticAccelerometer(Mat3::Identity()), 1.5));   // too long
    EXPECT_TRUE(filter.propagate(Vec3::Zero(), staticAccelerometer(Mat3::Identity()), 1.01));
    EXPECT_EQ(filter.rejectedSteps(), 3U);
    EXPECT_EQ(filter.propagations(), 1U);
}

TEST(ImuFilter, DoesNothingBeforeInitialisation) {
    ImuFilter filter(enabledParams());
    EXPECT_FALSE(filter.propagate(Vec3::Zero(), Vec3(0.0, 0.0, kGravity), 0.01));
    EXPECT_FALSE(filter.updateGravity(Vec3(0.0, 0.0, kGravity)));
    EXPECT_FALSE(filter.updatePose(Pose{}, Mat6::Identity()));
}

TEST(ImuFilter, PropagationGrowsUncertaintyMonotonically) {
    ImuFilter filter(enabledParams());
    filter.initialise(Pose{}, 0.0);
    const double before = filter.covariance().trace();

    const double step = 1.0 / 200.0;
    for(int i = 1; i <= 200; ++i) {
        filter.propagate(Vec3::Zero(), staticAccelerometer(Mat3::Identity()), step * i);
    }

    EXPECT_GT(filter.covariance().trace(), before);
}

// ---------------------------------------------------------------------------
// Gravity: what it can and cannot fix
// ---------------------------------------------------------------------------

TEST(ImuFilter, GravityUpdateCorrectsRollAndPitchButNotYaw) {
    // The whole justification for the accelerometer in this filter. Roll and
    // pitch are observable against an absolute direction and therefore
    // bounded; yaw is not, and a filter that appeared to correct yaw from an
    // accelerometer would be fitting noise.
    ImuFilter filter(enabledParams());

    Pose wrong;
    wrong.R = expSO3(Vec3(0.10, -0.08, 0.20));  // roll, pitch and yaw all off
    filter.initialise(wrong, 0.0);

    const Vec3 initialError = attitudeError(filter, Mat3::Identity());
    ASSERT_GT(std::hypot(initialError.x(), initialError.y()), 0.1);

    const double step = 1.0 / 200.0;
    for(int i = 1; i <= 2000; ++i) {  // ten seconds of stillness
        const Vec3 accel = staticAccelerometer(Mat3::Identity());  // truth is level
        filter.propagate(Vec3::Zero(), accel, step * i);
        filter.updateGravity(accel);
    }

    const Vec3 finalError = attitudeError(filter, Mat3::Identity());
    const double tiltBefore = std::hypot(initialError.x(), initialError.y());
    const double tiltAfter = std::hypot(finalError.x(), finalError.y());

    EXPECT_LT(tiltAfter, 0.2 * tiltBefore) << "gravity failed to bound roll and pitch";
    // Yaw is untouched: unobservable from gravity, so it must neither be
    // corrected nor corrupted.
    EXPECT_NEAR(std::abs(finalError.z()), std::abs(initialError.z()), 0.03);
    EXPECT_GT(filter.gravityUpdates(), 1000U);
}

TEST(ImuFilter, GravityUpdateIsGatedWhileAccelerating) {
    // The measurement model says "the only acceleration is gravity". Under
    // linear acceleration that is false, and using the reading anyway rotates
    // the attitude into the direction of travel -- the classic failure of a
    // naive accelerometer-aided attitude filter.
    ImuFilterParams params = enabledParams();
    params.gravityGateTolerance = 0.5;
    ImuFilter filter(params);
    filter.initialise(Pose{}, 0.0);

    EXPECT_TRUE(filter.updateGravity(Vec3(0.0, 0.0, kGravity)));
    EXPECT_FALSE(filter.updateGravity(Vec3(4.0, 0.0, kGravity)));   // |a| well over g
    EXPECT_FALSE(filter.updateGravity(Vec3(0.0, 0.0, 2.0)));        // free fall
    EXPECT_EQ(filter.gravityUpdates(), 1U);
}

TEST(ImuFilter, GravityUpdateCanBeDisabled) {
    ImuFilterParams params = enabledParams();
    params.useGravityUpdate = false;
    ImuFilter filter(params);
    filter.initialise(Pose{}, 0.0);
    EXPECT_FALSE(filter.updateGravity(Vec3(0.0, 0.0, kGravity)));
}

TEST(ImuFilter, GyroBiasIsObservableThroughGravity) {
    // A constant gyro bias tips the attitude away from level; gravity keeps
    // pulling it back; the only self-consistent explanation is a bias, so the
    // filter should find it. This is what makes the bias state worth carrying
    // rather than trusting the factory value.
    ImuFilterParams params = enabledParams();
    params.gravitySigma = 0.5;  // still, so the model genuinely holds here
    ImuFilter filter(params);
    filter.initialise(Pose{}, 0.0);

    const Vec3 trueBias(0.01, -0.006, 0.0);  // ~0.5 deg/s
    const double step = 1.0 / 200.0;
    for(int i = 1; i <= 6000; ++i) {  // thirty seconds
        const Vec3 measuredGyro = trueBias;  // platform still, so all we read is bias
        const Vec3 accel = staticAccelerometer(Mat3::Identity());
        filter.propagate(measuredGyro, accel, step * i);
        filter.updateGravity(accel);
    }

    // Only the two gravity-observable axes are expected to converge; bias about
    // the gravity axis is not observable while still, which is why trueBias has
    // a zero there.
    EXPECT_NEAR(filter.state().gyroBias.x(), trueBias.x(), 0.004);
    EXPECT_NEAR(filter.state().gyroBias.y(), trueBias.y(), 0.004);
}

// ---------------------------------------------------------------------------
// VO fusion
// ---------------------------------------------------------------------------

TEST(ImuFilter, PoseUpdatePullsTowardAConfidentMeasurement) {
    ImuFilter filter(enabledParams());
    filter.initialise(Pose{}, 0.0);

    Pose measured;
    measured.R = expSO3(Vec3(0.0, 0.0, 0.05));
    measured.t = Vec3(0.3, -0.2, 0.1);

    // A very confident measurement: the filter should move nearly all the way.
    const Mat6 tight = Mat6::Identity() * 1e-8;
    for(int i = 0; i < 5; ++i) {
        ASSERT_TRUE(filter.updatePose(measured, tight));
    }

    EXPECT_LT((filter.state().worldFromBody.t - measured.t).norm(), 1e-3);
    EXPECT_LT(logSO3(filter.state().worldFromBody.R.transpose() * measured.R).norm(), 1e-3);
    EXPECT_EQ(filter.poseUpdates(), 5U);
}

TEST(ImuFilter, PoseUpdateIgnoresAnUninformativeMeasurement) {
    ImuFilter filter(enabledParams());
    filter.initialise(Pose{}, 0.0);

    Pose measured;
    measured.t = Vec3(5.0, 0.0, 0.0);  // absurd, and declared absurdly uncertain

    ASSERT_TRUE(filter.updatePose(measured, Mat6::Identity() * 1e6));
    EXPECT_LT(filter.state().worldFromBody.t.norm(), 0.05);
}

TEST(ImuFilter, PoseCovarianceUsesTranslationThenRotationOrdering) {
    // ROS covariance arrays are (x, y, z, rx, ry, rz). Getting this backwards
    // publishes a plausible-looking matrix that tells a downstream filter the
    // wrong thing about which axes are uncertain.
    ImuFilterParams params = enabledParams();
    params.initialPositionSigma = 0.5;   // variance 0.25
    params.initialAttitudeSigma = 0.02;  // variance 4e-4
    ImuFilter filter(params);
    filter.initialise(Pose{}, 0.0);

    const Mat6 covariance = filter.poseCovariance();
    EXPECT_NEAR(covariance(0, 0), 0.25, 1e-12);
    EXPECT_NEAR(covariance(3, 3), 4e-4, 1e-12);
}

// ---------------------------------------------------------------------------
// The covariance has to stay a covariance
// ---------------------------------------------------------------------------

TEST(ImuFilter, CovarianceStaysSymmetricAndPositiveSemiDefinite) {
    // Over a long run the textbook (I - KH)P update loses symmetry and then
    // definiteness to rounding, and a filter that has quietly gone indefinite
    // produces nonsense that reads as a modelling error. The Joseph form plus
    // explicit symmetrisation is the guard; this is the test that it works.
    ImuFilter filter(enabledParams());
    filter.initialise(Pose{}, 0.0);

    const double step = 1.0 / 200.0;
    for(int i = 1; i <= 4000; ++i) {
        const Vec3 omega(0.05 * std::sin(0.01 * i), 0.03, -0.02);
        const Vec3 accel = staticAccelerometer(filter.state().worldFromBody.R);
        filter.propagate(omega, accel, step * i);
        filter.updateGravity(accel);
        if(i % 20 == 0) {
            Pose vo = filter.state().worldFromBody;
            vo.t += Vec3(1e-3, -1e-3, 5e-4);
            filter.updatePose(vo, Mat6::Identity() * 1e-4);
        }
    }

    const Mat15& covariance = filter.covariance();
    EXPECT_LT((covariance - covariance.transpose()).norm(), 1e-12);

    const Eigen::SelfAdjointEigenSolver<Mat15> solver(covariance);
    ASSERT_EQ(solver.info(), Eigen::Success);
    EXPECT_GT(solver.eigenvalues().minCoeff(), -1e-12);
    EXPECT_TRUE(covariance.allFinite());
    EXPECT_TRUE(filter.state().worldFromBody.R.allFinite());
}

TEST(ImuFilter, ResetClearsEverything) {
    ImuFilter filter(enabledParams());
    filter.initialise(Pose{}, 0.0);
    filter.propagate(Vec3::Zero(), Vec3(0.0, 0.0, kGravity), 0.005);
    ASSERT_GT(filter.propagations(), 0U);

    filter.reset();
    EXPECT_FALSE(filter.state().initialised);
    EXPECT_EQ(filter.propagations(), 0U);
    EXPECT_FALSE(filter.propagate(Vec3::Zero(), Vec3(0.0, 0.0, kGravity), 0.01));
}
