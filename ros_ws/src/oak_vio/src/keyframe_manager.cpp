#include "oak_vio/keyframe_manager.hpp"

#include <algorithm>
#include <cmath>
#include <utility>

namespace oak_vio {

void FeatureFrame::reset(const std::vector<Observation>& observations, std::int64_t frameIndex) {
    observations_.clear();
    observations_.reserve(observations.size() * 2);
    for(const Observation& obs : observations) {
        observations_.emplace(obs.id, obs);
    }
    frameIndex_ = frameIndex;
}

void FeatureFrame::clear() {
    observations_.clear();
    frameIndex_ = -1;
}

const Observation* FeatureFrame::find(std::uint32_t id) const {
    const auto it = observations_.find(id);
    return it == observations_.end() ? nullptr : &it->second;
}

KeyframeManager::KeyframeManager(VioParams params) : params_(std::move(params)) {}

void KeyframeManager::setKeyframe(const std::vector<Observation>& observations, std::int64_t frameIndex) {
    keyframe_.reset(observations, frameIndex);
}

void KeyframeManager::reset() {
    keyframe_.clear();
}

bool KeyframeManager::insideMask(float u, float v, int imageWidth, int imageHeight) const {
    if(params_.maskBorderFraction <= 0.0) {
        return u >= 0.0F && v >= 0.0F && u < static_cast<float>(imageWidth) && v < static_cast<float>(imageHeight);
    }
    const double marginX = params_.maskBorderFraction * imageWidth;
    const double marginY = params_.maskBorderFraction * imageHeight;
    return u >= marginX && v >= marginY && u <= imageWidth - marginX && v <= imageHeight - marginY;
}

std::vector<Correspondence> KeyframeManager::buildCorrespondences(const std::vector<Observation>& current,
                                                                  const RectifiedCamera& camera,
                                                                  int imageWidth,
                                                                  int imageHeight) const {
    std::vector<Correspondence> correspondences;
    if(keyframe_.empty()) {
        return correspondences;
    }
    correspondences.reserve(std::min(current.size(), keyframe_.size()));

    for(const Observation& now : current) {
        const Observation* then = keyframe_.find(now.id);
        if(then == nullptr) {
            continue;  // feature appeared after the keyframe
        }

        // Disparity must be valid and in range at *both* ends. A feature with
        // good disparity now but none at the keyframe cannot be triangulated
        // into the reference frame.
        if(now.disparity < params_.minDisparityPx || now.disparity > params_.maxDisparityPx) {
            continue;
        }
        if(then->disparity < params_.minDisparityPx || then->disparity > params_.maxDisparityPx) {
            continue;
        }
        if(now.trackingError > params_.maxTrackingError) {
            continue;
        }
        if(!insideMask(now.u, now.v, imageWidth, imageHeight) || !insideMask(then->u, then->v, imageWidth, imageHeight)) {
            continue;
        }

        // Depth consistency. A feature whose depth changed by more than a
        // large factor between the keyframe and now is almost certainly a
        // mistracked corner or a disparity outlier, not real motion. The bound
        // is deliberately loose — it is a sanity filter, not a motion model,
        // and RANSAC handles the rest.
        const double depthRatio = static_cast<double>(then->disparity) / static_cast<double>(now.disparity);
        if(depthRatio < 0.2 || depthRatio > 5.0) {
            continue;
        }

        Correspondence c;
        c.id = now.id;
        c.pointKf = camera.unproject(then->u, then->v, then->disparity);
        c.uvKf = Vec2(then->u, then->v);
        c.zCur = Vec3(now.u, now.v, now.disparity);
        c.trackingError = now.trackingError;
        c.age = now.age;
        c.depthKf = c.pointKf.z();
        c.disparityKf = then->disparity;

        if(!camera.isInFront(c.pointKf)) {
            continue;
        }

        correspondences.push_back(c);
    }

    return correspondences;
}

KeyframeManager::PromotionDecision KeyframeManager::shouldPromote(int numInliers, double medianParallaxPx, std::int64_t currentFrameIndex) const {
    if(!params_.keyframeReferenced) {
        // Frame-to-frame mode: every frame becomes the reference. Kept as a
        // runtime switch so the drift benefit of keyframing can be measured
        // on the same recorded sequence rather than argued about.
        return {true, "frame-to-frame mode"};
    }
    if(keyframe_.empty()) {
        return {true, "no keyframe"};
    }
    if(numInliers < params_.keyframeMinTracked) {
        return {true, "tracked features below threshold"};
    }
    if(medianParallaxPx > params_.keyframeMaxParallaxPx) {
        return {true, "parallax exceeds threshold"};
    }
    if(currentFrameIndex - keyframe_.frameIndex() >= params_.keyframeMaxAgeFrames) {
        return {true, "keyframe age limit"};
    }
    return {false, ""};
}

}  // namespace oak_vio
