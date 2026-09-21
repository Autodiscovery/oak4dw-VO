#include "oak_vio/imu_filter.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace oak_vio {
namespace {

constexpr int kTheta = 0;
constexpr int kVel = 3;
constexpr int kPos = 6;
constexpr int kGyroBias = 9;
constexpr int kAccelBias = 12;

}  // namespace

ImuFilter::ImuFilter(ImuFilterParams params) : params_(std::move(params)) {
    reset();
}

void ImuFilter::setParams(const ImuFilterParams& params) {
    params_ = params;
}

void ImuFilter::reset() {
    state_ = ImuFilterState{};
    covariance_.setZero();
    propagations_ = 0;
    gravityUpdates_ = 0;
    poseUpdates_ = 0;
    rejectedSteps_ = 0;
}

void ImuFilter::initialise(const Pose& worldFromBody, double timestampSeconds, const Vec3& gyroBias, const Vec3& accelBias) {
    state_.worldFromBody = worldFromBody;
    state_.worldFromBody.normalise();
    state_.velocity.setZero();
    state_.gyroBias = gyroBias;
    state_.accelBias = accelBias;
    state_.timestampSeconds = timestampSeconds;
    state_.initialised = true;

    covariance_.setZero();
    const auto fill = [this](int index, double sigma) {
        const double variance = sigma * sigma;
        for(int i = 0; i < 3; ++i) {
            covariance_(index + i, index + i) = variance;
        }
    };
    fill(kTheta, params_.initialAttitudeSigma);
    fill(kVel, params_.initialVelocitySigma);
    fill(kPos, params_.initialPositionSigma);
    fill(kGyroBias, params_.initialGyroBiasSigma);
    fill(kAccelBias, params_.initialAccelBiasSigma);

    propagations_ = 0;
    gravityUpdates_ = 0;
    poseUpdates_ = 0;
    rejectedSteps_ = 0;
}

bool ImuFilter::propagate(const Vec3& angularVelocityBody, const Vec3& accelerationBody, double timestampSeconds) {
    if(!state_.initialised) {
        return false;
    }
    const double dt = timestampSeconds - state_.timestampSeconds;
    // A step that goes backwards means the samples arrived out of order; one
    // that is too long means samples were dropped. Extrapolating across either
    // invents motion, and the covariance would not reflect that it was
    // invented, so refuse and let the next VO update re-anchor instead.
    if(dt <= 0.0 || dt > params_.maxPropagationStepSeconds) {
        ++rejectedSteps_;
        return false;
    }

    const Vec3 omega = angularVelocityBody - state_.gyroBias;
    const Vec3 accelBody = accelerationBody - state_.accelBias;

    const Mat3 rotation = state_.worldFromBody.R;
    const Vec3 gravityWorld(0.0, 0.0, -params_.gravityMagnitude);
    const Vec3 accelWorld = rotation * accelBody + gravityWorld;

    // ---- Nominal state ---------------------------------------------------
    state_.worldFromBody.t += state_.velocity * dt + 0.5 * accelWorld * dt * dt;
    state_.velocity += accelWorld * dt;
    state_.worldFromBody.R = rotation * expSO3(omega * dt);
    state_.worldFromBody.normalise();
    state_.timestampSeconds = timestampSeconds;

    // ---- Error-state transition -----------------------------------------
    // Local attitude perturbation, R_true = R * exp([dTheta]x):
    //
    //     dTheta' = -[omega]x dTheta - dbg
    //     dv'     = -R [a]x dTheta - R dba
    //     dp'     = dv
    //
    // First-order discretisation is enough at 200 Hz: the neglected terms are
    // O(dt^2) against a step of 5 ms.
    Mat15 transition = Mat15::Identity();
    transition.block<3, 3>(kTheta, kTheta) -= skew(omega) * dt;
    transition.block<3, 3>(kTheta, kGyroBias) = -Mat3::Identity() * dt;
    transition.block<3, 3>(kVel, kTheta) = -rotation * skew(accelBody) * dt;
    transition.block<3, 3>(kVel, kAccelBias) = -rotation * dt;
    transition.block<3, 3>(kPos, kVel) = Mat3::Identity() * dt;

    // Noise densities integrate to variances over the step. The position block
    // picks up the accelerometer's contribution through the double integration,
    // which is the dt^3/3 term; dropping it makes early position uncertainty
    // read as zero.
    Mat15 processNoise = Mat15::Zero();
    const double gyroVariance = params_.gyroNoiseDensity * params_.gyroNoiseDensity * dt;
    const double accelVariance = params_.accelNoiseDensity * params_.accelNoiseDensity * dt;
    const double gyroBiasVariance = params_.gyroBiasRandomWalk * params_.gyroBiasRandomWalk * dt;
    const double accelBiasVariance = params_.accelBiasRandomWalk * params_.accelBiasRandomWalk * dt;
    const double positionVariance = params_.accelNoiseDensity * params_.accelNoiseDensity * dt * dt * dt / 3.0;
    processNoise.block<3, 3>(kTheta, kTheta) = Mat3::Identity() * gyroVariance;
    processNoise.block<3, 3>(kVel, kVel) = Mat3::Identity() * accelVariance;
    processNoise.block<3, 3>(kPos, kPos) = Mat3::Identity() * positionVariance;
    processNoise.block<3, 3>(kGyroBias, kGyroBias) = Mat3::Identity() * gyroBiasVariance;
    processNoise.block<3, 3>(kAccelBias, kAccelBias) = Mat3::Identity() * accelBiasVariance;

    covariance_ = transition * covariance_ * transition.transpose() + processNoise;
    symmetrise();

    ++propagations_;
    return true;
}

bool ImuFilter::updateGravity(const Vec3& accelerationBody) {
    if(!state_.initialised || !params_.useGravityUpdate) {
        return false;
    }

    const Vec3 measured = accelerationBody - state_.accelBias;
    // The model is "the only acceleration is gravity". When that is visibly
    // false the reading is not a gravity measurement, and using it anyway would
    // rotate the whole attitude estimate into the direction of travel.
    if(std::abs(measured.norm() - params_.gravityMagnitude) > params_.gravityGateTolerance) {
        return false;
    }

    // Static platform: a_body = R^T * (0, 0, +g) + bias.
    const Vec3 gravityUp(0.0, 0.0, params_.gravityMagnitude);
    const Vec3 predictedGravityBody = state_.worldFromBody.R.transpose() * gravityUp;
    const Vec3 residual = accelerationBody - (predictedGravityBody + state_.accelBias);

    Eigen::Matrix<double, 3, 15> jacobian = Eigen::Matrix<double, 3, 15>::Zero();
    // d/dTheta [ (R exp([dTheta]x))^T g ] = +[R^T g]x
    jacobian.block<3, 3>(0, kTheta) = skew(predictedGravityBody);
    jacobian.block<3, 3>(0, kAccelBias) = Mat3::Identity();

    const Mat3 noise = Mat3::Identity() * (params_.gravitySigma * params_.gravitySigma);
    const Mat3 innovationCovariance = jacobian * covariance_ * jacobian.transpose() + noise;
    const Eigen::Matrix<double, 15, 3> gain = covariance_ * jacobian.transpose() * innovationCovariance.inverse();

    applyCorrection(gain * residual);

    // Joseph form. The textbook (I - KH)P is algebraically equal but loses
    // symmetry and positive-definiteness to rounding over long runs, and a
    // filter that has quietly gone indefinite looks like a modelling problem.
    const Mat15 identityMinusKH = Mat15::Identity() - gain * jacobian;
    covariance_ = identityMinusKH * covariance_ * identityMinusKH.transpose() + gain * noise * gain.transpose();
    symmetrise();

    ++gravityUpdates_;
    return true;
}

bool ImuFilter::updatePose(const Pose& worldFromBody, const Mat6& covariance) {
    if(!state_.initialised) {
        return false;
    }

    // Measurement ordered (translation, rotation) to match the incoming
    // covariance and the ROS convention, rather than the error state's order.
    Vec6 residual;
    residual.head<3>() = worldFromBody.t - state_.worldFromBody.t;
    residual.tail<3>() = logSO3(state_.worldFromBody.R.transpose() * worldFromBody.R);

    Eigen::Matrix<double, 6, 15> jacobian = Eigen::Matrix<double, 6, 15>::Zero();
    jacobian.block<3, 3>(0, kPos) = Mat3::Identity();
    jacobian.block<3, 3>(3, kTheta) = Mat3::Identity();

    const Mat6 noise = covariance * std::max(1e-9, params_.voCovarianceScale);
    const Mat6 innovationCovariance = jacobian * covariance_ * jacobian.transpose() + noise;
    const Eigen::Matrix<double, 15, 6> gain = covariance_ * jacobian.transpose() * innovationCovariance.inverse();

    applyCorrection(gain * residual);

    const Mat15 identityMinusKH = Mat15::Identity() - gain * jacobian;
    covariance_ = identityMinusKH * covariance_ * identityMinusKH.transpose() + gain * noise * gain.transpose();
    symmetrise();

    ++poseUpdates_;
    return true;
}

void ImuFilter::applyCorrection(const Vec15& correction) {
    // Attitude folds in multiplicatively on the right, matching the local
    // perturbation the Jacobians were derived with. Adding it to anything would
    // leave SO(3).
    state_.worldFromBody.R = state_.worldFromBody.R * expSO3(correction.segment<3>(kTheta));
    state_.worldFromBody.normalise();
    state_.velocity += correction.segment<3>(kVel);
    state_.worldFromBody.t += correction.segment<3>(kPos);
    state_.gyroBias += correction.segment<3>(kGyroBias);
    state_.accelBias += correction.segment<3>(kAccelBias);
}

void ImuFilter::symmetrise() {
    covariance_ = 0.5 * (covariance_ + covariance_.transpose()).eval();
}

Mat6 ImuFilter::poseCovariance() const {
    Mat6 out = Mat6::Zero();
    out.topLeftCorner<3, 3>() = covariance_.block<3, 3>(kPos, kPos);
    out.topRightCorner<3, 3>() = covariance_.block<3, 3>(kPos, kTheta);
    out.bottomLeftCorner<3, 3>() = covariance_.block<3, 3>(kTheta, kPos);
    out.bottomRightCorner<3, 3>() = covariance_.block<3, 3>(kTheta, kTheta);
    return out;
}

}  // namespace oak_vio
