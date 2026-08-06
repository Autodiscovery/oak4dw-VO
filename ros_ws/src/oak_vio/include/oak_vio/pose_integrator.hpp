// Accumulates keyframe-relative estimates into a world pose, and propagates
// covariance while doing it.
//
// The estimator produces T_kf_cur (keyframe camera -> current camera). The
// world pose of the current camera is
//
//     T_world_cur = T_world_kf * inverse(T_kf_cur)
//
// and on keyframe promotion T_world_kf takes the current value, carrying its
// accumulated covariance with it. Only keyframe-to-keyframe error compounds;
// within a span the pose is measured directly against the anchor.
//
// Covariance is propagated with first-order Jacobians for the perturbation
// convention used throughout (see types.hpp and motion_estimator.hpp):
//
//     T_true = Pert(delta) * T_est,   Pert((rho, phi)) = (Exp(phi) R, Exp(phi) t + rho)
//
// The ordering (translation, then rotation) matches the ROS covariance
// convention for nav_msgs/Odometry, so no reshuffling is needed downstream.
#pragma once

#include "oak_vio/types.hpp"

namespace oak_vio {

class PoseIntegrator {
   public:
    PoseIntegrator();

    /// Reset to a known world pose with zero uncertainty. Called at startup
    /// and by the reset service.
    void reset(const Pose& worldFromCamera = Pose{});

    /// Fold in a new keyframe-relative estimate.
    void update(const Pose& keyframeFromCurrent, const Mat6& keyframeFromCurrentCovariance);

    /// Make the current pose the new keyframe anchor. Call this *after*
    /// update() on the frame being promoted.
    void anchorAtCurrent();

    [[nodiscard]] const Pose& worldFromCamera() const { return worldFromCamera_; }
    [[nodiscard]] const Mat6& covariance() const { return currentCovariance_; }
    [[nodiscard]] const Pose& worldFromKeyframe() const { return worldFromKeyframe_; }

    /// First-order Jacobians of the perturbation convention. Exposed for unit
    /// tests, which check them against numerical derivatives.
    /// d(delta_c)/d(delta_b) for c = a * b.
    [[nodiscard]] static Mat6 composeJacobianRhs(const Pose& a, const Pose& b, const Pose& c);
    /// d(delta_inv)/d(delta) for inv = inverse(pose).
    [[nodiscard]] static Mat6 inverseJacobian(const Pose& pose);

   private:
    Pose worldFromKeyframe_;
    Mat6 keyframeCovariance_;
    Pose worldFromCamera_;
    Mat6 currentCovariance_;
};

}  // namespace oak_vio
