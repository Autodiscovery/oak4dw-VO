// Ideal rectified pinhole stereo model.
//
// After StereoDepth rectification the pair behaves as two ideal pinhole
// cameras with identical intrinsics, coplanar image planes and a pure
// horizontal baseline. Every equation in the estimator assumes that, so this
// is the single place the assumption is encoded.
//
// Populate from the rectified CameraInfo the driver publishes, not from raw
// sensor intrinsics — the rectified P matrix is what actually describes the
// images the estimator sees. See fromProjectionMatrices().
#pragma once

#include <array>
#include <cmath>
#include <stdexcept>

#include "oak_vio/types.hpp"

namespace oak_vio {

class RectifiedCamera {
   public:
    RectifiedCamera() = default;

    RectifiedCamera(double fx, double fy, double cx, double cy, double baselineM)
        : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baselineM) {
        if(fx_ <= 0.0 || fy_ <= 0.0 || baseline_ <= 0.0) {
            throw std::invalid_argument("RectifiedCamera: fx, fy and baseline must be positive");
        }
        fxBaseline_ = fx_ * baseline_;
    }

    /// Build from the left and right rectified 3x4 projection matrices as
    /// published in sensor_msgs/CameraInfo. The baseline is recovered from the
    /// right camera's Tx term: P_right(0,3) == -fx * baseline.
    static RectifiedCamera fromProjectionMatrices(const std::array<double, 12>& pLeft, const std::array<double, 12>& pRight) {
        const double fx = pLeft[0];
        const double fy = pLeft[5];
        const double cx = pLeft[2];
        const double cy = pLeft[6];
        if(fx <= 0.0) {
            throw std::invalid_argument("RectifiedCamera: left P matrix has non-positive fx");
        }
        const double baseline = -pRight[3] / fx;
        if(baseline <= 0.0) {
            throw std::invalid_argument(
                "RectifiedCamera: recovered a non-positive baseline from the right P matrix. "
                "Check that the right camera_info is the RECTIFIED one and that left/right are not swapped.");
        }
        return {fx, fy, cx, cy, baseline};
    }

    /// Triangulate a rectified-left pixel with its disparity into a 3D point
    /// in the left camera frame. Caller must have checked disparity > 0.
    [[nodiscard]] Vec3 unproject(double u, double v, double disparity) const {
        const double z = fxBaseline_ / disparity;
        return {(u - cx_) * z / fx_, (v - cy_) * z / fy_, z};
    }

    /// Project a 3D point to the measurement vector the estimator compares
    /// against: (u, v, disparity).
    ///
    /// Note we use disparity as the third component rather than the right
    /// image's u. They are algebraically equivalent (u_r = u_l - d) but the
    /// disparity form gives a sparser Jacobian and keeps the residual
    /// components closer to independent. The right image's v carries no
    /// information at all in a rectified pair — it equals the left v by
    /// construction — which is why there are three residuals here, not four.
    [[nodiscard]] Vec3 project(const Vec3& p) const {
        const double invZ = 1.0 / p.z();
        return {fx_ * p.x() * invZ + cx_, fy_ * p.y() * invZ + cy_, fxBaseline_ * invZ};
    }

    /// d(project)/d(point), evaluated at p. Rows follow project()'s ordering.
    [[nodiscard]] Mat3 projectionJacobian(const Vec3& p) const {
        const double invZ = 1.0 / p.z();
        const double invZ2 = invZ * invZ;
        Mat3 j = Mat3::Zero();
        j(0, 0) = fx_ * invZ;
        j(0, 2) = -fx_ * p.x() * invZ2;
        j(1, 1) = fy_ * invZ;
        j(1, 2) = -fy_ * p.y() * invZ2;
        j(2, 2) = -fxBaseline_ * invZ2;
        return j;
    }

    /// A point is only usable if it is in front of the camera by a sane margin.
    [[nodiscard]] bool isInFront(const Vec3& p) const {
        return p.z() > 1e-3;
    }

    [[nodiscard]] double fx() const { return fx_; }
    [[nodiscard]] double fy() const { return fy_; }
    [[nodiscard]] double cx() const { return cx_; }
    [[nodiscard]] double cy() const { return cy_; }
    [[nodiscard]] double baseline() const { return baseline_; }

    /// Depth at a given disparity — handy for logging and gating.
    [[nodiscard]] double depthAtDisparity(double disparity) const {
        return fxBaseline_ / disparity;
    }

    /// One-sigma depth error at a given depth, for a given disparity noise.
    /// dZ = Z^2 * sigma_d / (fx * b). Used to sanity check the configuration
    /// at startup and to log what precision the current setup actually has.
    [[nodiscard]] double depthSigmaAt(double depthM, double disparitySigmaPx) const {
        return depthM * depthM * disparitySigmaPx / fxBaseline_;
    }

   private:
    double fx_{0.0};
    double fy_{0.0};
    double cx_{0.0};
    double cy_{0.0};
    double baseline_{0.0};
    double fxBaseline_{0.0};
};

}  // namespace oak_vio
