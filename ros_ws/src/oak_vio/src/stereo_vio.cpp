#include "oak_vio/stereo_vio.hpp"

#include <chrono>
#include <utility>

#include "oak_vio/bucketing.hpp"

namespace oak_vio {

StereoVio::StereoVio(RectifiedCamera camera, VioParams params, int imageWidth, int imageHeight)
    : camera_(std::move(camera)),
      params_(std::move(params)),
      imageWidth_(imageWidth),
      imageHeight_(imageHeight),
      keyframes_(params_),
      solver_(MotionEstimator(camera_, params_), params_) {
    reset();
}

void StereoVio::setParams(const VioParams& params) {
    params_ = params;
    keyframes_.setParams(params_);
    solver_ = RansacMotionSolver(MotionEstimator(camera_, params_), params_);
}

void StereoVio::reset(const Pose& worldFromCamera) {
    keyframes_.reset();
    motionModel_.reset();
    integrator_.reset(worldFromCamera);
    keyframeFromPrevious_ = Pose{};
    haveKeyframeFromPrevious_ = false;
    frameIndex_ = -1;
    lastTimestamp_ = 0.0;
    haveLastTimestamp_ = false;
    consecutiveFailures_ = 0;
    everTracked_ = false;
    lastSelected_.clear();
    lastInlierIndices_.clear();
}

void StereoVio::promoteKeyframe(const std::vector<Observation>& observations, bool anchorPose) {
    keyframes_.setKeyframe(observations, frameIndex_);
    if(anchorPose) {
        integrator_.anchorAtCurrent();
    }
    keyframeFromPrevious_ = Pose{};
    haveKeyframeFromPrevious_ = false;
}

VioFrameResult StereoVio::processFrame(const std::vector<Observation>& observations, double timestampSeconds) {
    return processFrame(observations, timestampSeconds, RotationPrior{});
}

VioFrameResult StereoVio::processFrame(const std::vector<Observation>& observations, double timestampSeconds, const RotationPrior& rotationPrior) {
    const auto wallStart = std::chrono::steady_clock::now();

    ++frameIndex_;

    VioFrameResult result;
    result.frameIndex = frameIndex_;
    result.timestampSeconds = timestampSeconds;
    result.numObservations = static_cast<int>(observations.size());
    result.worldFromCamera = integrator_.worldFromCamera();
    result.poseCovariance = integrator_.covariance();

    const double dt = haveLastTimestamp_ ? (timestampSeconds - lastTimestamp_) : 0.0;
    lastTimestamp_ = timestampSeconds;
    haveLastTimestamp_ = true;

    const auto finish = [&](VioFrameResult& r) {
        r.solveMilliseconds = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wallStart).count();
        return r;
    };

    // ---- Bootstrap, or a frame with nothing in it ------------------------
    if(!keyframes_.hasKeyframe()) {
        promoteKeyframe(observations, false);
        // Only actually promoted if the frame had observations to promote --
        // KeyframeManager treats an empty keyframe as no keyframe.
        result.keyframePromoted = keyframes_.hasKeyframe();

        if(everTracked_) {
            // This branch is reached two ways, and reporting them the same is a
            // defect in the one signal consumers are told to trust. Before
            // anything has ever tracked, no keyframe means starting up. AFTER
            // tracking has worked, no keyframe means the frames arriving carry
            // nothing usable -- a covered lens, a blackout, severe blur -- which
            // is LOST, not initialising. A downstream filter reading
            // Initialising concludes "no pose yet, wait"; reading Lost it
            // concludes "hold the last pose and inflate", which is the correct
            // response and the opposite behaviour.
            result.state = TrackingState::Lost;
            result.note = "no usable observations, so no keyframe could be anchored";
            ++consecutiveFailures_;
            motionModel_.reset();
        } else {
            result.state = TrackingState::Initialising;
            result.note = "keyframe initialised";
        }
        result.poseCovariance = integrator_.covariance();
        return finish(result);
    }

    // ---- Correspondences -------------------------------------------------
    const std::vector<Correspondence> all = keyframes_.buildCorrespondences(observations, camera_, imageWidth_, imageHeight_);
    result.numCorrespondences = static_cast<int>(all.size());

    const std::vector<int> selectedIndices = selectBucketed(all, imageWidth_, imageHeight_, params_);
    result.numSelected = static_cast<int>(selectedIndices.size());
    result.medianParallaxPx = medianParallax(all, selectedIndices);

    std::vector<Correspondence> selected;
    selected.reserve(selectedIndices.size());
    for(const int idx : selectedIndices) {
        selected.push_back(all[static_cast<std::size_t>(idx)]);
    }

    if(selected.size() < static_cast<std::size_t>(params_.minInliers)) {
        ++consecutiveFailures_;
        result.state = TrackingState::Lost;
        result.note = "too few correspondences to attempt a solve";
        motionModel_.reset();
        if(consecutiveFailures_ >= kFailuresBeforeReanchor) {
            promoteKeyframe(observations, true);
            result.keyframePromoted = true;
            consecutiveFailures_ = 0;
        }
        result.poseCovariance = integrator_.covariance();
        return finish(result);
    }

    // ---- Seed ------------------------------------------------------------
    // Constant velocity predicts the *inter-frame* motion; compose it onto the
    // last keyframe-relative estimate to get a keyframe-relative seed.
    //
    // The gyro prior, when present, overrides that prediction's ROTATION and
    // leaves its translation alone. Two reasons it is worth the plumbing:
    // rotation is the component the constant-velocity model predicts worst
    // (angular rate changes far faster than linear velocity on a walking or
    // legged platform), and fast rotation is simultaneously when motion blur
    // thins the corners — so the seed matters most exactly when the vision is
    // weakest. It also applies on frames where constant velocity has nothing to
    // offer, which is the first frame of every keyframe span.
    Pose seed = keyframeFromPrevious_;
    Pose predictedDelta;  // identity
    bool havePrediction = false;

    if(params_.useConstantVelocitySeed && haveKeyframeFromPrevious_ && dt > 0.0) {
        predictedDelta = motionModel_.predict(dt, frameIndex_, params_.maxSeedAgeFrames);
        havePrediction = true;
    }

    if(rotationPrior.valid) {
        predictedDelta.R = blendRotation(predictedDelta.R, rotationPrior.deltaRotation, params_.gyroPriorWeight);
        havePrediction = true;
        result.usedGyroPrior = true;
        result.gyroPriorAngleRad = rotationPrior.angleRad;
    } else {
        result.gyroPriorRejection = rotationPrior.rejection;
    }

    if(havePrediction) {
        // Composing onto keyframeFromPrevious_ is right in both cases: when the
        // keyframe is the previous frame that member is identity, so this
        // collapses to the inter-frame prediction itself.
        seed = predictedDelta * keyframeFromPrevious_;
    }

    // ---- Solve -----------------------------------------------------------
    const RansacResult solved = solver_.solve(selected, seed);
    result.numInliers = static_cast<int>(solved.inliers.size());
    result.inlierRatio = solved.inlierRatio;

    if(retainDebug_) {
        lastSelected_ = selected;
        lastInlierIndices_ = solved.inliers;
    }

    if(!solved.success) {
        ++consecutiveFailures_;
        result.state = TrackingState::Lost;
        result.note = solved.rejectionReason;
        motionModel_.reset();
        if(consecutiveFailures_ >= kFailuresBeforeReanchor) {
            promoteKeyframe(observations, true);
            result.keyframePromoted = true;
            consecutiveFailures_ = 0;
        }
        result.poseCovariance = integrator_.covariance();
        return finish(result);
    }

    consecutiveFailures_ = 0;

    // ---- Integrate -------------------------------------------------------
    if(dt > 0.0) {
        if(haveKeyframeFromPrevious_) {
            // previous -> current, via the keyframe.
            motionModel_.update(solved.pose * keyframeFromPrevious_.inverse(), dt, frameIndex_);
        } else if(keyframes_.keyframeIndex() == frameIndex_ - 1) {
            // The keyframe *is* the previous frame — which is always the case
            // in frame-to-frame mode, and on the first frame after any
            // promotion. Then the keyframe-relative estimate is already the
            // inter-frame motion. Without this branch the motion model never
            // updates in frame-to-frame mode, so the published twist would sit
            // at zero and the constant-velocity seed would never engage.
            motionModel_.update(solved.pose, dt, frameIndex_);
        }
    }

    integrator_.update(solved.pose, solved.covariance);
    keyframeFromPrevious_ = solved.pose;
    haveKeyframeFromPrevious_ = true;

    result.valid = true;
    everTracked_ = true;
    result.keyframeFromCurrent = solved.pose;
    result.worldFromCamera = integrator_.worldFromCamera();
    result.poseCovariance = integrator_.covariance();
    result.linearVelocity = motionModel_.linearVelocity();
    result.angularVelocity = motionModel_.angularVelocity();

    const bool weak = result.numInliers < params_.minInliers || solved.inlierRatio < params_.minInlierRatio;
    result.state = weak ? TrackingState::LowInliers : TrackingState::Tracking;
    if(weak) {
        result.note = "solved, but inlier support is weak";
    }

    // ---- Keyframe promotion ---------------------------------------------
    const KeyframeManager::PromotionDecision decision = keyframes_.shouldPromote(result.numInliers, result.medianParallaxPx, frameIndex_);
    if(decision.promote) {
        promoteKeyframe(observations, true);
        result.keyframePromoted = true;
        if(result.note[0] == '\0') {
            result.note = decision.reason;
        }
    }

    return finish(result);
}

}  // namespace oak_vio
