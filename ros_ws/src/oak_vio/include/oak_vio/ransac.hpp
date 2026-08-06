// RANSAC wrapper around MotionEstimator.
//
// Hypotheses come from the closed-form three-point solve rather than from
// Gauss-Newton on a minimal set: it is exact, iteration-free and cannot
// diverge. Scoring is done on image-plane reprojection error, which is the
// statistically meaningful metric even though the hypothesis was generated in
// 3D.
#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "oak_vio/motion_estimator.hpp"
#include "oak_vio/types.hpp"
#include "oak_vio/vio_params.hpp"

namespace oak_vio {

struct RansacResult {
    Pose pose;                            ///< keyframe -> current
    Mat6 covariance{Mat6::Identity()};
    std::vector<int> inliers;
    int numCorrespondences{0};
    double inlierRatio{0.0};
    int iterationsUsed{0};
    bool success{false};
    /// True when the geometry did not constrain all six DOF, or when the
    /// solution failed the motion plausibility check.
    bool rejected{false};
    const char* rejectionReason{""};
};

class RansacMotionSolver {
   public:
    RansacMotionSolver(MotionEstimator estimator, VioParams params, std::uint64_t seed = 0x0AC4'171DULL);

    /// Estimate keyframe->current motion. `seed` is the prior (constant
    /// velocity, or identity); it is evaluated as a hypothesis in its own
    /// right, which usually wins outright when the platform moves smoothly and
    /// saves the whole sampling loop from mattering.
    [[nodiscard]] RansacResult solve(const std::vector<Correspondence>& correspondences, const Pose& seed);

    /// Deterministic re-seeding, so a recorded sequence replays identically.
    void reseed(std::uint64_t seed) { rng_.seed(seed); }

   private:
    /// Count and collect inliers for a hypothesis.
    [[nodiscard]] std::vector<int> scoreHypothesis(const std::vector<Correspondence>& correspondences, const Pose& pose) const;

    /// Adaptive stopping: how many samples are still needed to see one clean
    /// triple with the configured confidence, given the best inlier ratio so far.
    [[nodiscard]] int requiredIterations(double inlierRatio) const;

    [[nodiscard]] bool motionIsPlausible(const Pose& pose) const;

    MotionEstimator estimator_;
    VioParams params_;
    std::mt19937_64 rng_;
};

}  // namespace oak_vio
