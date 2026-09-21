#include "oak_vio/ransac.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace oak_vio {

RansacMotionSolver::RansacMotionSolver(MotionEstimator estimator, VioParams params, std::uint64_t seed)
    : estimator_(std::move(estimator)), params_(std::move(params)), rng_(seed) {}

std::vector<int> RansacMotionSolver::scoreHypothesis(const std::vector<Correspondence>& correspondences, const Pose& pose) const {
    std::vector<int> inliers;
    inliers.reserve(correspondences.size());
    const double threshold = params_.ransacInlierThresholdPx;
    for(std::size_t i = 0; i < correspondences.size(); ++i) {
        if(estimator_.reprojectionError(correspondences[i], pose) < threshold) {
            inliers.push_back(static_cast<int>(i));
        }
    }
    return inliers;
}

int RansacMotionSolver::requiredIterations(double inlierRatio) const {
    if(inlierRatio <= 0.0) {
        return params_.ransacMaxIterations;
    }
    if(inlierRatio >= 1.0) {
        return params_.ransacMinIterations;
    }
    // Probability that a random triple is all-inlier.
    const double pAllInliers = inlierRatio * inlierRatio * inlierRatio;
    if(pAllInliers >= 1.0 - 1e-12) {
        return params_.ransacMinIterations;
    }
    const double numerator = std::log(1.0 - params_.ransacConfidence);
    const double denominator = std::log(1.0 - pAllInliers);
    if(denominator >= -1e-12) {
        return params_.ransacMaxIterations;
    }
    const double needed = std::ceil(numerator / denominator);
    return std::clamp(static_cast<int>(needed), params_.ransacMinIterations, params_.ransacMaxIterations);
}

bool RansacMotionSolver::motionIsPlausible(const Pose& pose) const {
    if(!pose.R.allFinite() || !pose.t.allFinite()) {
        return false;
    }
    return pose.translationNorm() <= params_.maxTranslationPerSpanM && pose.rotationAngle() <= params_.maxRotationPerSpanRad;
}

RansacResult RansacMotionSolver::solve(const std::vector<Correspondence>& correspondences, const Pose& seed) {
    RansacResult result;
    result.numCorrespondences = static_cast<int>(correspondences.size());

    if(correspondences.size() < 3) {
        result.rejected = true;
        result.rejectionReason = "fewer than 3 correspondences";
        return result;
    }

    Pose bestPose = seed;
    std::vector<int> bestInliers;

    std::uniform_int_distribution<std::size_t> pick(0, correspondences.size() - 1);
    const auto total = static_cast<double>(correspondences.size());

    int iterations = 0;
    int budget = params_.ransacMaxIterations;

    // Evaluate the prior first. When the platform moves smoothly this is
    // usually already the best hypothesis, and starting from a strong inlier
    // ratio collapses the adaptive iteration count to the minimum.
    //
    // The budget update below is the part that makes that true, and it used to
    // be missing. The budget was only ever lowered inside the loop, when a
    // RANDOM hypothesis beat the current best -- so in precisely the case this
    // is for, where the seed is already the best hypothesis and nothing beats
    // it, the budget stayed at ransacMaxIterations and the solve ran the full
    // 200 iterations. A good seed produced the WORST cost, exactly inverting
    // the intent, and it went unnoticed because the test that catches it
    // (Ransac.GoodSeedReducesIterations) had never been run: the machine this
    // was written on had no C++ toolchain.
    //
    // It matters twice over. The estimator is the only part of this pipeline
    // that costs real ARM cycles, so this was a ~10x overspend on every frame
    // that tracked smoothly; and the Phase 4 gyro prior's entire rationale is
    // that a better seed is cheap because RANSAC exploits it. It could not.
    if(motionIsPlausible(seed)) {
        bestInliers = scoreHypothesis(correspondences, seed);
        if(bestInliers.size() >= 3) {
            budget = requiredIterations(static_cast<double>(bestInliers.size()) / total);
        }
    }
    while(iterations < budget) {
        ++iterations;

        // Sample three distinct indices.
        std::array<int, 3> sample{};
        sample[0] = static_cast<int>(pick(rng_));
        do {
            sample[1] = static_cast<int>(pick(rng_));
        } while(sample[1] == sample[0]);
        do {
            sample[2] = static_cast<int>(pick(rng_));
        } while(sample[2] == sample[0] || sample[2] == sample[1]);

        Pose hypothesis;
        if(!estimator_.solveMinimal(correspondences, sample, hypothesis)) {
            continue;  // degenerate triple
        }
        if(!motionIsPlausible(hypothesis)) {
            continue;
        }

        std::vector<int> inliers = scoreHypothesis(correspondences, hypothesis);
        if(inliers.size() > bestInliers.size()) {
            bestInliers = std::move(inliers);
            bestPose = hypothesis;
            budget = std::min(budget, std::max(iterations, requiredIterations(static_cast<double>(bestInliers.size()) / total)));
        }
    }

    result.iterationsUsed = iterations;

    if(bestInliers.size() < 3) {
        result.rejected = true;
        result.rejectionReason = "no hypothesis reached 3 inliers";
        result.inlierRatio = 0.0;
        return result;
    }

    // Polish on the full inlier set, then re-score once. The refined pose
    // usually captures a few more inliers than the minimal hypothesis did.
    RefinementResult refined = estimator_.refine(correspondences, bestInliers, bestPose);
    if(!refined.degenerate && motionIsPlausible(refined.pose)) {
        std::vector<int> refinedInliers = scoreHypothesis(correspondences, refined.pose);
        if(refinedInliers.size() >= bestInliers.size()) {
            bestInliers = std::move(refinedInliers);
            bestPose = refined.pose;
            // One more pass now that the inlier set has grown.
            RefinementResult second = estimator_.refine(correspondences, bestInliers, bestPose);
            if(!second.degenerate && motionIsPlausible(second.pose)) {
                bestPose = second.pose;
                refined = second;
            }
        }
    }

    result.pose = bestPose;
    result.inliers = std::move(bestInliers);
    result.inlierRatio = static_cast<double>(result.inliers.size()) / total;
    result.covariance = estimator_.covarianceFromInformation(refined.informationMatrix, static_cast<int>(result.inliers.size()));

    if(refined.degenerate) {
        result.rejected = true;
        result.rejectionReason = "normal equations rank deficient";
        return result;
    }
    if(!motionIsPlausible(result.pose)) {
        result.rejected = true;
        result.rejectionReason = "motion exceeds plausibility bounds";
        return result;
    }

    result.success = true;
    return result;
}

}  // namespace oak_vio
