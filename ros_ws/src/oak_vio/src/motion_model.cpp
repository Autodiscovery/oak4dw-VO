#include "oak_vio/motion_model.hpp"

#include <cmath>

namespace oak_vio {

void MotionModel::update(const Pose& deltaPrevToCur, double dtSeconds, std::int64_t frameIndex) {
    lastDelta_ = deltaPrevToCur;
    lastDt_ = dtSeconds;
    lastFrameIndex_ = frameIndex;
    valid_ = dtSeconds > 1e-6;

    if(valid_) {
        // The delta maps previous-camera points into the current camera, so
        // the camera's own motion is the inverse. Report velocity of the
        // camera, which is what a twist message means.
        const Pose cameraMotion = deltaPrevToCur.inverse();
        linearVelocity_ = cameraMotion.t / dtSeconds;
        angularVelocity_ = logSO3(cameraMotion.R) / dtSeconds;
    } else {
        linearVelocity_.setZero();
        angularVelocity_.setZero();
    }
}

Pose MotionModel::predict(double dtSeconds, std::int64_t frameIndex, int maxSeedAgeFrames) const {
    if(!valid_ || lastDt_ <= 1e-6 || dtSeconds <= 0.0) {
        return {};
    }
    // A stale prediction is worse than no prediction: if frames were dropped
    // the platform may have done anything in between.
    if(frameIndex - lastFrameIndex_ > maxSeedAgeFrames) {
        return {};
    }

    // Scale the last motion to the new interval. Rotation scales in the Lie
    // algebra, translation linearly.
    const double scale = dtSeconds / lastDt_;
    Pose predicted;
    predicted.R = expSO3(logSO3(lastDelta_.R) * scale);
    predicted.t = lastDelta_.t * scale;
    return predicted;
}

void MotionModel::reset() {
    lastDelta_ = Pose{};
    lastDt_ = 0.0;
    lastFrameIndex_ = -1;
    valid_ = false;
    linearVelocity_.setZero();
    angularVelocity_.setZero();
}

}  // namespace oak_vio
