// Gyro integration into an inter-frame rotation prior — Phase 4, steps 1-3.
//
// The estimator already evaluates its seed as a RANSAC hypothesis in its own
// right, so a good rotation prior is worth more than its cost: it is exactly
// what fails under fast rotation, where the constant-velocity model has nothing
// useful to say and motion blur has thinned the corners at the same moment.
//
// This header is deliberately free of ROS and DepthAI so it can be unit tested
// against analytic trajectories on a workstation, like the rest of the core.
// The node converts dai::IMUPacket into ImuSample and hands it over.
//
// ---------------------------------------------------------------------------
// Frames and sign conventions — the one place this is easy to get wrong
// ---------------------------------------------------------------------------
//
// The gyro reports the body's angular rate omega expressed in the BODY (IMU)
// frame. Orientation therefore integrates by RIGHT multiplication:
//
//     R_world_body(t + dt) = R_world_body(t) * exp([omega]x dt)
//
// Over a frame interval that accumulates to Q = prod exp([omega]x dt), in time
// order, right-multiplied. The estimator wants the transform that maps a point
// from the PREVIOUS camera frame into the CURRENT one:
//
//     p_cur = R_cur_prev * p_prev,      R_cur_prev = R_wc(t1)^T R_wc(t0) = Q^T
//
// so deltaRotation() returns Q TRANSPOSED. That single transpose is the
// difference between a prior that helps and one that doubles the rotation
// error, and it is silent — RANSAC simply rejects the hypothesis and falls back
// to the constant-velocity seed, so the symptom is "the IMU changed nothing"
// rather than an error. test_imu_integrator.cpp pins it against an analytic
// yaw sweep for that reason.
//
// A rotation increment measured in the IMU frame maps into the camera frame
// exactly, not approximately:
//
//     R_ci exp([omega_i]x dt) R_ci^T = exp([R_ci omega_i]x dt)
//
// so each sample's angular rate is rotated into the camera frame first and the
// integration happens entirely there. R_ci must take the IMU frame all the way
// to the RECTIFIED left camera frame, which is the frame the estimator's pixels
// live in — see setImuToCameraRotation.
#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>

#include "oak_vio/types.hpp"

namespace oak_vio {

/// One IMU report, already converted out of the DepthAI types.
///
/// Gyro and accelerometer arrive as separate reports on this device, so a
/// sample may carry either or both.
struct ImuSample {
    /// Device timestamp, same clock as ImgFrame::getTimestamp(). Mixing the
    /// device and host clocks here would produce a plausible-looking prior with
    /// a constant offset, which is worse than no prior at all.
    double timestampSeconds{0.0};
    Vec3 angularVelocity{Vec3::Zero()};  ///< rad/s, IMU frame
    Vec3 acceleration{Vec3::Zero()};     ///< m/s^2, IMU frame
    bool hasGyro{false};
    bool hasAccel{false};
};

struct ImuParams {
    /// Master switch. False means the IMU node is never created, so this costs
    /// nothing when off.
    bool enabled{false};

    /// Requested report rate, used only to judge whether an integration window
    /// has enough samples to be trustworthy.
    double reportRateHz{200.0};

    /// Reject the prior when less than this fraction of the frame interval is
    /// covered by closely-spaced samples. See maxSampleGapFactor for what
    /// "covered" means, and why the obvious definition is not enough.
    double minSampleCoverage{0.6};

    /// Longest sample spacing still counted as covered, as a multiple of the
    /// nominal interval 1/reportRateHz.
    ///
    /// This is what makes the coverage figure mean something. Integrating
    /// between two samples 50 ms apart produces a number either way — the
    /// trapezoid just spans the gap — so measuring only "did samples exist
    /// somewhere in the window" would report full coverage for a stream that
    /// had dropped a burst mid-interval. Spacings beyond this are still
    /// integrated (a held rate beats no prior) but do not count as covered, so
    /// a dropped burst shows up as low coverage and the prior is refused.
    double maxSampleGapFactor{3.0};

    /// How much of the prior's rotation to use, against the constant-velocity
    /// prediction: 1.0 is gyro only, 0.0 ignores it, in between is a geodesic
    /// blend. Default 1.0 — the gyro is far better at rotation than a
    /// constant-velocity extrapolation, and this is only a seed.
    double rotationPriorWeight{1.0};

    // ---- Bias ------------------------------------------------------------
    /// Estimate gyro bias from intervals where the platform is judged static.
    ///
    /// Bias is the dominant gyro error over the ~30-100 ms this integrates, and
    /// on a consumer MEMS part it drifts with temperature, so the factory value
    /// alone goes stale. Left uncorrected, a 0.5 deg/s bias is 0.017 deg per
    /// 33 ms frame — negligible as a seed, which is why this is a refinement
    /// rather than a prerequisite, and why the prior still works before the
    /// estimate is ready.
    bool estimateGyroBias{true};
    /// Static test: angular rate magnitude below this (rad/s)...
    double staticGyroThreshold{0.02};
    /// ...and acceleration magnitude within this of gravity (m/s^2). Both are
    /// required, because a platform in free fall or in a smooth turn can pass
    /// either one alone.
    double staticAccelTolerance{0.35};
    /// Static samples needed before the estimate is applied. At 200 Hz this is
    /// one second of stillness.
    int biasMinSamples{200};
    /// Exponential forgetting factor for the bias estimate, per static sample.
    /// 1.0 would be a plain running mean, which cannot follow thermal drift.
    double biasForgettingFactor{0.999};
    double gravityMagnitude{9.80665};

    // ---- Sanity ----------------------------------------------------------
    /// Discard a prior implying more rotation than this over one interval. A
    /// 1.5 rad inter-frame rotation is 86 degrees, which no camera on a robot
    /// does between frames; seeing one means the timestamps or the samples are
    /// wrong, and seeding RANSAC with it would be worse than seeding identity.
    double maxRotationPerIntervalRad{1.5};

    /// How much history to keep. Needs to cover the frame interval plus the
    /// transport jitter of the IMU queue; 1 s at 200 Hz is 200 samples.
    double bufferSeconds{1.0};
};

/// The rotation prior for one frame interval.
struct RotationPrior {
    bool valid{false};
    /// Maps points from the previous camera frame into the current one.
    Mat3 deltaRotation{Mat3::Identity()};
    /// Rotation magnitude, radians. Reported so the health line can show
    /// whether the prior is doing anything.
    double angleRad{0.0};
    int samplesUsed{0};
    /// Samples present over samples expected. Below params.minSampleCoverage
    /// the prior is rejected.
    double coverage{0.0};
    /// Why the prior was rejected, for the status topic. Empty when valid.
    const char* rejection{""};
};

/// Buffers gyro/accelerometer reports and integrates them between frame
/// timestamps.
///
/// Not thread safe. The node's IMU callback and its frame callback run on
/// different DepthAI threads, so the node guards this with a mutex rather than
/// paying for one on every sample here.
class ImuIntegrator {
   public:
    explicit ImuIntegrator(ImuParams params = {});

    void setParams(const ImuParams& params);
    [[nodiscard]] const ImuParams& params() const { return params_; }

    /// Rotation taking a vector from the IMU frame into the RECTIFIED left
    /// camera frame.
    ///
    /// Not simply the extrinsic from calibration: getImuToCameraExtrinsics()
    /// gives IMU to the RAW CAM_B frame, while the estimator's pixels live in
    /// the rectified frame, which differs by the stereo rectification rotation.
    /// The node composes the two. That rotation is usually small — a few
    /// degrees — so getting it wrong degrades the prior quietly instead of
    /// breaking it, which is the sort of error that survives for months.
    void setImuToCameraRotation(const Mat3& rotation);
    [[nodiscard]] const Mat3& imuToCameraRotation() const { return imuToCamera_; }

    /// Add one report. Samples older than the buffer window are dropped here.
    void addSample(const ImuSample& sample);

    /// Integrate over the half-open interval (t0, t1].
    ///
    /// Samples are used at their own timestamps, and the first and last
    /// intervals are clipped to the window bounds, so the result does not
    /// depend on where samples happen to fall relative to frame boundaries.
    [[nodiscard]] RotationPrior deltaRotation(double t0, double t1) const;

    /// Drop buffered samples. Keeps the bias estimate, which is a property of
    /// the sensor rather than of the track.
    void reset();
    /// Drop the bias estimate too.
    void resetBias();

    [[nodiscard]] Vec3 gyroBias() const { return gyroBias_; }
    [[nodiscard]] bool biasReady() const { return staticSamples_ >= params_.biasMinSamples; }
    [[nodiscard]] int staticSamples() const { return staticSamples_; }

    /// Most recent accelerometer reading, rotated into the camera frame.
    /// While the platform is not accelerating this is the gravity direction,
    /// which is what bounds roll and pitch in the Phase 4 filter.
    [[nodiscard]] Vec3 lastAccelerationCamera() const { return lastAccelCamera_; }
    [[nodiscard]] bool hasAcceleration() const { return haveAccel_; }

    [[nodiscard]] std::size_t bufferedSamples() const { return buffer_.size(); }
    /// Timestamp span currently buffered, seconds. Zero when fewer than two.
    [[nodiscard]] double bufferedSpanSeconds() const;
    [[nodiscard]] std::uint64_t totalSamples() const { return totalSamples_; }
    [[nodiscard]] std::uint64_t droppedOutOfOrder() const { return droppedOutOfOrder_; }

   private:
    void updateBias(const ImuSample& sample);

    ImuParams params_;
    Mat3 imuToCamera_{Mat3::Identity()};

    /// Gyro samples only, ascending in time. Accelerometer reports update the
    /// bias logic and the gravity direction but are not integrated.
    std::deque<ImuSample> buffer_;

    Vec3 gyroBias_{Vec3::Zero()};
    int staticSamples_{0};
    Vec3 lastAccelCamera_{Vec3::Zero()};
    Vec3 lastAccelImu_{Vec3::Zero()};
    bool haveAccel_{false};

    std::uint64_t totalSamples_{0};
    std::uint64_t droppedOutOfOrder_{0};
};

/// Geodesic interpolation from `from` to `to` on SO(3): weight 0 gives `from`,
/// 1 gives `to`. Used to blend the gyro prior against the constant-velocity
/// prediction.
[[nodiscard]] Mat3 blendRotation(const Mat3& from, const Mat3& to, double weight);

}  // namespace oak_vio
