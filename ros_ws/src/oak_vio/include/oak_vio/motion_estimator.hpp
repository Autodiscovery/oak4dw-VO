// Six-DOF motion estimation from stereo correspondences.
//
// Estimates the transform T taking points from the reference keyframe's
// camera frame into the current camera frame, by minimising reprojection
// error in the current frame.
//
// Parameterisation
// ----------------
// The plan called for libviso2's Euler-angle parameterisation. We use a left
// perturbation on SO(3) with decoupled translation instead:
//
//     R <- Exp(phi) * R,   t <- Exp(phi) * t + rho,   delta = (rho, phi)
//
// Euler angles are fine for the small frame-to-frame increments libviso2
// deals with, but we track against a keyframe that can span a whole turn, and
// Euler hits gimbal lock at pitch = +/-90 deg. The Lie formulation has no
// singularity and actually yields a *simpler* Jacobian:
//
//     P' = R * X + t
//     dP'/d(delta) = [ I_3 | -skew(P') ]
//
// Residual
// --------
// Three residuals per point, not four. In a rectified pair the right image's
// v coordinate equals the left's by construction, so a fourth row would be an
// exact duplicate of the second and would silently double-weight v. We use
// (u, v, disparity); see RectifiedCamera::project.
//
// Weighting
// ---------
// The keyframe 3D point is *not* exact — it was triangulated from a noisy
// disparity, with sigma_Z / Z = sigma_d / d. Ignoring that (the naive
// error-in-variables mistake) is fine on near scenes but produces occasional
// catastrophic divergence on far ones: in simulation over a 5-40 m scene the
// 95th-percentile translation error was 12.6 m when the point was treated as
// exact, versus 9.7 mm once its uncertainty was propagated.
//
// The propagation is cheap because the point's uncertainty is essentially
// rank-1 along its viewing ray. Pushing that through the pose and the
// projection gives a per-component variance inflation:
//
//     a = dh/dP' * R * ray_kf
//     var_i = sigma_meas_i^2 + sigma_Z^2 * a_i^2
//
// This also self-corrects: a far point contributes almost nothing under pure
// rotation (where its depth genuinely does not matter) and is heavily
// discounted under translation (where it does).
#pragma once

#include <cstdint>
#include <vector>

#include "oak_vio/rectified_camera.hpp"
#include "oak_vio/types.hpp"
#include "oak_vio/vio_params.hpp"

namespace oak_vio {

/// Outcome of a Gauss-Newton refinement over a fixed correspondence set.
struct RefinementResult {
    Pose pose;                    ///< keyframe -> current
    Mat6 informationMatrix{Mat6::Zero()};  ///< J^T W J, before noise scaling
    double finalCost{0.0};        ///< robustified sum of squared residuals
    int iterations{0};
    bool converged{false};
    /// Set when the normal equations were rank deficient — the geometry did
    /// not constrain all six degrees of freedom.
    bool degenerate{false};
};

class MotionEstimator {
   public:
    MotionEstimator(RectifiedCamera camera, VioParams params);

    /// Closed-form minimal solve from three 3D-3D point pairs (Umeyama /
    /// Arun, without scale). Used for RANSAC hypothesis generation: it is
    /// exact, iteration-free, and cannot diverge, which makes it far better
    /// suited to minimal sets than seeding Gauss-Newton.
    ///
    /// We can do this because we have disparity at *both* the keyframe and
    /// the current frame, so both point clouds are known. Scoring is still
    /// done in 2D reprojection, which is the statistically correct metric.
    ///
    /// Returns false on a degenerate (collinear or coincident) triple.
    [[nodiscard]] bool solveMinimal(const std::vector<Correspondence>& correspondences,
                                    const std::array<int, 3>& indices,
                                    Pose& out) const;

    /// Same closed form over an arbitrary index set. Used to bootstrap the
    /// refinement when no motion prior is available.
    [[nodiscard]] bool solveClosedForm(const std::vector<Correspondence>& correspondences,
                                       const std::vector<int>& indices,
                                       Pose& out) const;

    /// Iteratively reweighted Gauss-Newton over the given correspondence
    /// subset, minimising Huber-robustified reprojection error.
    [[nodiscard]] RefinementResult refine(const std::vector<Correspondence>& correspondences,
                                          const std::vector<int>& indices,
                                          const Pose& seed) const;

    /// Reprojection error, in pixels, of one correspondence under a pose.
    /// This is the RANSAC scoring function. Points that fall behind the
    /// camera return infinity.
    [[nodiscard]] double reprojectionError(const Correspondence& c, const Pose& pose) const;

    /// Everything the inner loop needs for one correspondence at one pose.
    struct PointEvaluation {
        Vec3 residual{Vec3::Zero()};
        Mat36 jacobian{Mat36::Zero()};
        /// Per-component measurement variance, including the propagated
        /// keyframe-depth uncertainty.
        Vec3 variance{Vec3::Ones()};
        bool valid{false};
    };

    /// Residual, Jacobian and variance for one correspondence.
    [[nodiscard]] PointEvaluation evaluate(const Correspondence& c, const Pose& pose) const;

    /// Residual and its 3x6 Jacobian for one correspondence. Exposed for unit
    /// tests, which check it against a numerical derivative.
    /// Returns false if the point is behind the camera.
    [[nodiscard]] bool residualAndJacobian(const Correspondence& c,
                                           const Pose& pose,
                                           Vec3& residual,
                                           Mat36& jacobian) const;

    /// Invert an information matrix into a pose covariance. Weights are
    /// absolute (1/variance), so this is a plain inverse — no extra noise
    /// scaling. Rank-deficient inputs come back with large but finite
    /// diagonal entries rather than infinities, so downstream filters stay
    /// well behaved.
    [[nodiscard]] Mat6 covarianceFromInformation(const Mat6& information, int numInliers) const;

    [[nodiscard]] const RectifiedCamera& camera() const { return camera_; }
    [[nodiscard]] const VioParams& params() const { return params_; }

   private:
    RectifiedCamera camera_;
    VioParams params_;
    /// Base measurement variance (pixel, pixel, disparity), before the
    /// keyframe-depth term is added.
    Vec3 baseVariance_{Vec3::Ones()};
};

}  // namespace oak_vio
