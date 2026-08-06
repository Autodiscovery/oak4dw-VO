// Synthetic stereo scene for the unit tests.
//
// Generates 3D landmarks and renders them as tracked-feature observations from
// an arbitrary camera pose, with configurable pixel noise, disparity noise and
// outlier contamination. Deterministic given a seed, so a failure is always
// reproducible.
//
// The formulas here mirror tools/validate_estimator_math.py, which has been
// checked numerically against finite differences and ground truth. Keep the
// three in sync.
#pragma once

#include <cmath>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

#include "oak_vio/rectified_camera.hpp"
#include "oak_vio/types.hpp"

namespace oak_vio::test {

/// A rectified configuration representative of the OAK 4 D W after alpha
/// scaling to roughly 100 deg HFoV.
inline RectifiedCamera makeTestCamera() {
    return {537.0, 537.0, 640.0, 400.0, 0.075};
}

class SyntheticScene {
   public:
    SyntheticScene(RectifiedCamera camera, int width, int height, std::uint64_t seed)
        : camera_(camera), width_(width), height_(height), rng_(seed) {}

    /// Scatter landmarks in front of the origin camera, log-uniform in depth
    /// so near and far points are both well represented.
    void generatePoints(int count, double minDepth, double maxDepth) {
        points_.clear();
        points_.reserve(static_cast<std::size_t>(count));
        std::uniform_real_distribution<double> uDist(0.05 * width_, 0.95 * width_);
        std::uniform_real_distribution<double> vDist(0.05 * height_, 0.95 * height_);
        std::uniform_real_distribution<double> logDepth(std::log(minDepth), std::log(maxDepth));

        while(static_cast<int>(points_.size()) < count) {
            const double depth = std::exp(logDepth(rng_));
            const double disparity = camera_.fx() * camera_.baseline() / depth;
            if(disparity < 1.0) {
                continue;
            }
            points_.push_back(camera_.unproject(uDist(rng_), vDist(rng_), disparity));
        }
        ages_.clear();
    }

    /// Observe the landmarks from `worldFromCamera` (camera pose in the world
    /// frame, where the world frame is the initial camera frame).
    ///
    /// `outlierFraction` corrupts that share of observations with large random
    /// pixel displacements, simulating mistracked corners.
    std::vector<Observation> observe(const Pose& worldFromCamera, double pixelNoise, double disparityNoise, double outlierFraction = 0.0) {
        const Pose cameraFromWorld = worldFromCamera.inverse();
        std::normal_distribution<double> pixelDist(0.0, std::max(pixelNoise, 1e-12));
        std::normal_distribution<double> disparityDist(0.0, std::max(disparityNoise, 1e-12));
        std::normal_distribution<double> outlierDist(0.0, 25.0);
        std::uniform_real_distribution<double> unit(0.0, 1.0);

        std::vector<Observation> observations;
        observations.reserve(points_.size());

        for(std::size_t i = 0; i < points_.size(); ++i) {
            const Vec3 inCamera = cameraFromWorld.apply(points_[i]);
            if(inCamera.z() <= 0.1) {
                continue;
            }
            Vec3 measured = camera_.project(inCamera);
            if(measured.z() < 1.0) {
                continue;  // beyond usable disparity
            }
            if(pixelNoise > 0.0) {
                measured.x() += pixelDist(rng_);
                measured.y() += pixelDist(rng_);
            }
            if(disparityNoise > 0.0) {
                measured.z() += disparityDist(rng_);
            }
            if(measured.z() < 1.0) {
                continue;
            }
            const bool isOutlier = outlierFraction > 0.0 && unit(rng_) < outlierFraction;
            if(isOutlier) {
                measured.x() += outlierDist(rng_);
                measured.y() += outlierDist(rng_);
            }
            if(measured.x() < 0.0 || measured.y() < 0.0 || measured.x() >= width_ || measured.y() >= height_) {
                continue;
            }

            const auto id = static_cast<std::uint32_t>(i);
            Observation observation;
            observation.id = id;
            observation.u = static_cast<float>(measured.x());
            observation.v = static_cast<float>(measured.y());
            observation.disparity = static_cast<float>(measured.z());
            observation.trackingError = 1.0F;
            observation.age = ++ages_[id];
            observations.push_back(observation);
        }
        return observations;
    }

    /// Build correspondences directly, bypassing KeyframeManager. Used by the
    /// estimator-level tests that want full control of the input.
    std::vector<Correspondence> correspondences(const std::vector<Observation>& keyframe, const std::vector<Observation>& current) const {
        std::unordered_map<std::uint32_t, const Observation*> index;
        index.reserve(keyframe.size() * 2);
        for(const auto& observation : keyframe) {
            index[observation.id] = &observation;
        }

        std::vector<Correspondence> out;
        out.reserve(current.size());
        for(const auto& now : current) {
            const auto it = index.find(now.id);
            if(it == index.end()) {
                continue;
            }
            const Observation& then = *it->second;
            Correspondence c;
            c.id = now.id;
            c.pointKf = camera_.unproject(then.u, then.v, then.disparity);
            c.uvKf = Vec2(then.u, then.v);
            c.zCur = Vec3(now.u, now.v, now.disparity);
            c.trackingError = now.trackingError;
            c.age = now.age;
            c.depthKf = c.pointKf.z();
            c.disparityKf = then.disparity;
            out.push_back(c);
        }
        return out;
    }

    [[nodiscard]] const std::vector<Vec3>& points() const { return points_; }
    [[nodiscard]] const RectifiedCamera& camera() const { return camera_; }
    std::mt19937_64& rng() { return rng_; }

   private:
    RectifiedCamera camera_;
    int width_;
    int height_;
    std::mt19937_64 rng_;
    std::vector<Vec3> points_;
    std::unordered_map<std::uint32_t, std::uint32_t> ages_;
};

/// Pose from an axis-angle rotation and a translation, for readable tests.
inline Pose makePose(const Vec3& rotationVector, const Vec3& translation) {
    Pose pose;
    pose.R = expSO3(rotationVector);
    pose.t = translation;
    return pose;
}

/// Geodesic rotation error between two poses, in radians.
inline double rotationError(const Pose& a, const Pose& b) {
    return logSO3(a.R.transpose() * b.R).norm();
}

}  // namespace oak_vio::test
