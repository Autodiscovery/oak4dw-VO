// Constant-velocity motion prediction.
//
// Seeds the solve from the previous inter-frame motion rather than identity.
// Costs nothing, cuts Gauss-Newton iterations, and materially improves RANSAC
// conditioning under fast motion because the prior is evaluated as a
// hypothesis in its own right.
//
// This is also the slot the Phase 4 gyro prior drops into: replace (or blend)
// the predicted rotation with the IMU-integrated one and nothing downstream
// changes.
#pragma once

#include <cstdint>

#include "oak_vio/types.hpp"

namespace oak_vio {

class MotionModel {
   public:
    /// Record the inter-frame motion just estimated. `deltaPrevToCur` maps
    /// points from the previous frame's camera into the current one.
    void update(const Pose& deltaPrevToCur, double dtSeconds, std::int64_t frameIndex);

    /// Predicted previous->current motion for the next frame, scaled to the
    /// new time step. Returns identity when no usable history exists.
    [[nodiscard]] Pose predict(double dtSeconds, std::int64_t frameIndex, int maxSeedAgeFrames) const;

    void reset();

    [[nodiscard]] bool valid() const { return valid_; }

    /// Linear and angular velocity from the last update, for the twist field
    /// of the published odometry.
    [[nodiscard]] Vec3 linearVelocity() const { return linearVelocity_; }
    [[nodiscard]] Vec3 angularVelocity() const { return angularVelocity_; }

   private:
    Pose lastDelta_;
    double lastDt_{0.0};
    std::int64_t lastFrameIndex_{-1};
    bool valid_{false};
    Vec3 linearVelocity_{Vec3::Zero()};
    Vec3 angularVelocity_{Vec3::Zero()};
};

}  // namespace oak_vio
