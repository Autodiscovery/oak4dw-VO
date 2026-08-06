// Core value types for the OAK 4 D stereo visual odometry estimator.
//
// This header is deliberately free of ROS and DepthAI dependencies so the
// estimator can be unit tested on a workstation without hardware.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <Eigen/SVD>
#include <Eigen/Geometry>

namespace oak_vio {

using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec6 = Eigen::Matrix<double, 6, 1>;
using Mat3 = Eigen::Matrix3d;
using Mat6 = Eigen::Matrix<double, 6, 6>;
using Mat36 = Eigen::Matrix<double, 3, 6>;

/// A single feature observation in the rectified left image, with the stereo
/// disparity sampled at that pixel. Mirrors dai::TrackedFeature plus a
/// disparity lookup.
struct Observation {
    std::uint32_t id{0};
    float u{0.0F};
    float v{0.0F};
    float disparity{0.0F};  ///< px, subpixel, strictly positive when valid
    float trackingError{0.0F};
    std::uint32_t age{0};
};

/// A feature seen both in the reference keyframe and in the current frame.
/// `pointKf` is already triangulated, so the estimator never touches the
/// camera model during its inner loop.
struct Correspondence {
    std::uint32_t id{0};
    Vec3 pointKf{Vec3::Zero()};  ///< 3D point in the keyframe camera frame, metres
    Vec2 uvKf{Vec2::Zero()};     ///< pixel position in the keyframe, for parallax
    Vec3 zCur{Vec3::Zero()};     ///< measurement in the current frame: (u, v, disparity)
    float trackingError{0.0F};
    std::uint32_t age{0};        ///< frames this feature has survived, from the HW tracker
    /// Depth of pointKf, cached so the weighting can propagate its uncertainty
    /// without recomputing. Equal to pointKf.z().
    double depthKf{0.0};
    /// Disparity measured at the keyframe. Sets the depth uncertainty:
    /// sigma_Z / Z == sigma_d / d.
    double disparityKf{0.0};

    [[nodiscard]] double parallax() const {
        return (zCur.head<2>() - uvKf).norm();
    }
};

/// Rigid transform. Applying it maps a point from the source frame to the
/// target frame: `p_target = R * p_source + t`.
struct Pose {
    Mat3 R{Mat3::Identity()};
    Vec3 t{Vec3::Zero()};

    [[nodiscard]] Vec3 apply(const Vec3& p) const {
        return R * p + t;
    }

    [[nodiscard]] Pose inverse() const {
        Pose out;
        out.R = R.transpose();
        out.t = -out.R * t;
        return out;
    }

    /// this ∘ rhs — apply `rhs` first, then `this`.
    [[nodiscard]] Pose operator*(const Pose& rhs) const {
        Pose out;
        out.R = R * rhs.R;
        out.t = R * rhs.t + t;
        return out;
    }

    /// Re-orthonormalise R. Accumulated products drift off SO(3) over
    /// thousands of frames; call this after every integration step.
    void normalise() {
        const Eigen::JacobiSVD<Mat3> svd(R, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R = svd.matrixU() * svd.matrixV().transpose();
        if(R.determinant() < 0.0) {
            Mat3 correction = Mat3::Identity();
            correction(2, 2) = -1.0;
            R = svd.matrixU() * correction * svd.matrixV().transpose();
        }
    }

    [[nodiscard]] double translationNorm() const {
        return t.norm();
    }

    /// Rotation magnitude in radians, from the trace of R.
    [[nodiscard]] double rotationAngle() const {
        const double cosTheta = std::clamp((R.trace() - 1.0) * 0.5, -1.0, 1.0);
        return std::acos(cosTheta);
    }
};

/// Why a frame's estimate is or is not trustworthy. Published on the status
/// topic so a downstream filter can react rather than guess.
enum class TrackingState : std::uint8_t {
    Initialising = 0,
    Tracking = 1,
    LowInliers = 2,  ///< solved, but the solution is weak — covariance inflated
    Lost = 3,        ///< no usable solution; zero motion published
};

/// Skew-symmetric matrix such that `skew(a) * b == a.cross(b)`.
inline Mat3 skew(const Vec3& v) {
    Mat3 m;
    // clang-format off
    m <<   0.0, -v.z(),  v.y(),
         v.z(),    0.0, -v.x(),
        -v.y(),  v.x(),    0.0;
    // clang-format on
    return m;
}

/// SO(3) exponential map (Rodrigues), numerically safe near zero.
inline Mat3 expSO3(const Vec3& omega) {
    const double theta = omega.norm();
    if(theta < 1e-10) {
        // Second-order expansion; exact enough well inside double precision.
        return Mat3::Identity() + skew(omega);
    }
    const Vec3 axis = omega / theta;
    const Mat3 K = skew(axis);
    return Mat3::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * K * K;
}

/// SO(3) logarithm, inverse of expSO3.
inline Vec3 logSO3(const Mat3& R) {
    const double cosTheta = std::clamp((R.trace() - 1.0) * 0.5, -1.0, 1.0);
    const double theta = std::acos(cosTheta);
    if(theta < 1e-10) {
        return Vec3(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1)) * 0.5;
    }
    const double scale = theta / (2.0 * std::sin(theta));
    return Vec3(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1)) * scale;
}

}  // namespace oak_vio
