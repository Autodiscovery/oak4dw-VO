// Error-state Kalman filter fusing VO with the IMU — Phase 4, steps 4-5.
//
// SCAFFOLD. Off by default (vio.i_imu_filter_enabled), and deliberately so:
// propagation and the gravity update below are pinned by unit tests, but the
// filter as a whole has never run on hardware and its tuning is guesswork until
// it has. Enabling it changes what `~/vo/odometry` means, so it stays behind a
// switch until the loop-drift numbers say it helps.
//
// What it is for
// ---------------------------------------------------------------------------
// Roll and pitch are the only drift axes that are both observable and
// correctable without an external reference: gravity gives an absolute
// direction, so the accelerometer bounds them indefinitely. Yaw and position
// have no such anchor and will still drift — an IMU cannot fix that, and any
// claim otherwise is a claim about a magnetometer or a map. The honest benefit
// here is therefore bounded attitude, smoother output between frames, and
// odometry at IMU rate rather than frame rate, which at the 10 Hz this rig
// currently runs at is the difference between usable and not for a controller.
//
// State
// ---------------------------------------------------------------------------
// Nominal: R (world<-body), v (world), p (world), gyro bias, accel bias.
// Error state, 15 dimensions, in this order:
//
//     [0:3)   dTheta   attitude error, LOCAL: R_true = R_nominal * exp([dTheta]x)
//     [3:6)   dv       velocity error, world frame
//     [6:9)   dp       position error, world frame
//     [9:12)  dbg      gyro bias error
//     [12:15) dba      accelerometer bias error
//
// The local (right) attitude perturbation is chosen to match the estimator's
// SE(3) left-perturbation convention in spirit — no Euler angles anywhere, no
// gimbal lock — and because it makes the gravity Jacobian a clean skew matrix.
//
// "body" here is the RECTIFIED left camera frame, not the IMU frame: samples are
// rotated into it by ImuIntegrator before they arrive, so everything downstream
// of this filter shares one frame with the VO.
#pragma once

#include <cstdint>

#include "oak_vio/imu_integrator.hpp"
#include "oak_vio/types.hpp"

namespace oak_vio {

using Mat15 = Eigen::Matrix<double, 15, 15>;
using Vec15 = Eigen::Matrix<double, 15, 1>;

struct ImuFilterParams {
    /// Master switch. False leaves the VO output exactly as it is today.
    bool enabled{false};

    // ---- Process noise ---------------------------------------------------
    // Continuous-time densities, which is how an IMU datasheet and
    // CalibrationHandler::getImuNoiseParameters() both report them. The node
    // reads the device's own values and only falls back to these.
    double gyroNoiseDensity{2.0e-4};       ///< rad/s/sqrt(Hz)
    double accelNoiseDensity{2.0e-3};      ///< m/s^2/sqrt(Hz)
    double gyroBiasRandomWalk{2.0e-5};     ///< rad/s^2/sqrt(Hz)
    double accelBiasRandomWalk{2.0e-4};    ///< m/s^3/sqrt(Hz)

    // ---- Gravity update --------------------------------------------------
    /// Apply the accelerometer as a gravity-direction measurement.
    bool useGravityUpdate{true};
    /// Standard deviation of that measurement, m/s^2.
    ///
    /// Deliberately much larger than the sensor's own noise: the measurement
    /// model assumes the platform is not accelerating, and on a walking or
    /// driving robot that assumption is wrong most of the time. The residual is
    /// dominated by unmodelled linear acceleration, not by sensor noise, so
    /// sizing this from the datasheet would make the filter trust a tilted
    /// reading and rotate the whole state into it.
    double gravitySigma{1.5};
    /// Skip the update when |a| departs from gravity by more than this, which
    /// is the cheap test for "the platform is accelerating, so this reading is
    /// not gravity".
    double gravityGateTolerance{1.0};

    // ---- VO pose update --------------------------------------------------
    /// Scale applied to the VO covariance before using it as measurement
    /// noise. VO's absolute-pose error is correlated frame to frame (see the
    /// caveat on updatePose), so its reported covariance understates the
    /// independent information each frame carries.
    double voCovarianceScale{1.0};

    // ---- Initial uncertainty ---------------------------------------------
    double initialAttitudeSigma{0.1};      ///< rad
    double initialVelocitySigma{0.1};      ///< m/s
    double initialPositionSigma{0.01};     ///< m
    double initialGyroBiasSigma{0.01};     ///< rad/s
    double initialAccelBiasSigma{0.1};     ///< m/s^2

    double gravityMagnitude{9.80665};

    /// Refuse a propagation step longer than this, rather than extrapolating
    /// across a gap the filter knows nothing about.
    double maxPropagationStepSeconds{0.05};
};

struct ImuFilterState {
    Pose worldFromBody;                 ///< R, t == position
    Vec3 velocity{Vec3::Zero()};        ///< world frame, m/s
    Vec3 gyroBias{Vec3::Zero()};        ///< rad/s, body frame
    Vec3 accelBias{Vec3::Zero()};       ///< m/s^2, body frame
    double timestampSeconds{0.0};
    bool initialised{false};
};

/// 15-state error-state Kalman filter. Not thread safe.
class ImuFilter {
   public:
    explicit ImuFilter(ImuFilterParams params = {});

    void setParams(const ImuFilterParams& params);
    [[nodiscard]] const ImuFilterParams& params() const { return params_; }

    /// Start the filter at a known pose. Velocity starts at zero, biases at the
    /// integrator's current estimate if one is passed in.
    void initialise(const Pose& worldFromBody, double timestampSeconds, const Vec3& gyroBias = Vec3::Zero(), const Vec3& accelBias = Vec3::Zero());

    /// Strapdown propagation to `timestampSeconds` using one IMU sample whose
    /// angular rate and acceleration are already in the BODY (rectified left
    /// camera) frame. Returns false when the step is rejected as too long or
    /// out of order.
    bool propagate(const Vec3& angularVelocityBody, const Vec3& accelerationBody, double timestampSeconds);

    /// Accelerometer as a gravity-direction measurement. Bounds roll and pitch;
    /// carries no yaw information by construction. Returns false when gated out
    /// because the platform is accelerating.
    bool updateGravity(const Vec3& accelerationBody);

    /// VO absolute pose as a measurement.
    ///
    /// CAVEAT, and the reason this filter is not on by default: VO's absolute
    /// pose error is a random walk, so consecutive measurements are strongly
    /// correlated, while a Kalman update assumes they are independent. Feeding
    /// integrated VO pose in as if it were an independent fix makes the filter
    /// over-confident, which is the dangerous direction. The correct treatment
    /// is a relative-pose update against a cloned anchor state; that is the
    /// next piece of work and is why the hook below exists. Until then,
    /// voCovarianceScale is the blunt instrument that keeps this honest.
    bool updatePose(const Pose& worldFromBody, const Mat6& covariance);

    [[nodiscard]] const ImuFilterState& state() const { return state_; }
    [[nodiscard]] const Mat15& covariance() const { return covariance_; }
    /// Pose covariance in the estimator's (translation, rotation) ordering, so
    /// it can go straight onto a ROS message.
    [[nodiscard]] Mat6 poseCovariance() const;

    void reset();
    [[nodiscard]] std::uint64_t propagations() const { return propagations_; }
    [[nodiscard]] std::uint64_t gravityUpdates() const { return gravityUpdates_; }
    [[nodiscard]] std::uint64_t poseUpdates() const { return poseUpdates_; }
    [[nodiscard]] std::uint64_t rejectedSteps() const { return rejectedSteps_; }

   private:
    /// Fold an error-state correction into the nominal state and zero it.
    void applyCorrection(const Vec15& correction);
    /// Keep the covariance symmetric. Asymmetry accumulates from rounding and
    /// eventually makes the filter diverge in a way that looks like a modelling
    /// error rather than arithmetic.
    void symmetrise();

    ImuFilterParams params_;
    ImuFilterState state_;
    Mat15 covariance_{Mat15::Identity()};

    std::uint64_t propagations_{0};
    std::uint64_t gravityUpdates_{0};
    std::uint64_t poseUpdates_{0};
    std::uint64_t rejectedSteps_{0};
};

}  // namespace oak_vio
