#include "oak_vio/pose_integrator.hpp"

namespace oak_vio {

PoseIntegrator::PoseIntegrator() {
    reset();
}

void PoseIntegrator::reset(const Pose& worldFromCamera) {
    worldFromKeyframe_ = worldFromCamera;
    worldFromCamera_ = worldFromCamera;
    keyframeCovariance_ = Mat6::Zero();
    currentCovariance_ = Mat6::Zero();
}

Mat6 PoseIntegrator::inverseJacobian(const Pose& pose) {
    // With T_true = Pert(delta) * T and Pert((rho, phi)) = (Exp(phi) R, Exp(phi) t + rho):
    //     phi' = -R^T phi
    //     rho' = -R^T rho + skew(R^T t) R^T phi
    const Mat3 rt = pose.R.transpose();
    Mat6 j = Mat6::Zero();
    j.topLeftCorner<3, 3>() = -rt;
    j.topRightCorner<3, 3>() = skew(rt * pose.t) * rt;
    j.bottomRightCorner<3, 3>() = -rt;
    return j;
}

Mat6 PoseIntegrator::composeJacobianRhs(const Pose& a, const Pose& b, const Pose& c) {
    // For c = a * b, propagating b's error:
    //     phi_c = ... + a.R phi_b
    //     rho_c = ... + a.R rho_b + (skew(c.t) a.R - a.R skew(b.t)) phi_b
    // The corresponding Jacobian in a is exactly the identity, which is why
    // only the right-hand one is needed here.
    Mat6 j = Mat6::Zero();
    j.topLeftCorner<3, 3>() = a.R;
    j.topRightCorner<3, 3>() = skew(c.t) * a.R - a.R * skew(b.t);
    j.bottomRightCorner<3, 3>() = a.R;
    return j;
}

void PoseIntegrator::update(const Pose& keyframeFromCurrent, const Mat6& keyframeFromCurrentCovariance) {
    const Pose currentFromKeyframe = keyframeFromCurrent.inverse();
    const Mat6 inverseJ = inverseJacobian(keyframeFromCurrent);
    const Mat6 currentFromKeyframeCovariance = inverseJ * keyframeFromCurrentCovariance * inverseJ.transpose();

    worldFromCamera_ = worldFromKeyframe_ * currentFromKeyframe;
    worldFromCamera_.normalise();

    const Mat6 composeJ = composeJacobianRhs(worldFromKeyframe_, currentFromKeyframe, worldFromCamera_);

    // The anchor's own error passes through with an identity Jacobian, so the
    // two terms simply add.
    currentCovariance_ = keyframeCovariance_ + composeJ * currentFromKeyframeCovariance * composeJ.transpose();

    // Keep it symmetric — repeated products accumulate asymmetry that some
    // downstream consumers reject outright.
    currentCovariance_ = 0.5 * (currentCovariance_ + currentCovariance_.transpose()).eval();
}

void PoseIntegrator::anchorAtCurrent() {
    worldFromKeyframe_ = worldFromCamera_;
    keyframeCovariance_ = currentCovariance_;
}

}  // namespace oak_vio
