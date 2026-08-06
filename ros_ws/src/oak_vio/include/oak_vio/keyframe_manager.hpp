// Reference keyframe storage and promotion policy.
//
// This is the one idea worth taking from ORB-SLAM. Plain frame-to-frame VO
// (libviso2) compounds estimation error at every frame. Estimating instead
// against a reference keyframe means only keyframe-to-keyframe error
// compounds, and within a span the pose is measured directly.
//
// The HW feature tracker makes this nearly free: it already gives stable
// feature IDs across frames, so a feature tracked continuously from keyframe K
// to frame k needs no re-matching — it is the same ID. A "keyframe" here is
// just a snapshot of a few hundred (id -> u, v, disparity) tuples, not an
// ORB-SLAM keyframe with descriptors and a covisibility graph.
#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "oak_vio/rectified_camera.hpp"
#include "oak_vio/types.hpp"
#include "oak_vio/vio_params.hpp"

namespace oak_vio {

/// Immutable snapshot of the observations at one frame.
class FeatureFrame {
   public:
    void reset(const std::vector<Observation>& observations, std::int64_t frameIndex);
    void clear();

    [[nodiscard]] const Observation* find(std::uint32_t id) const;
    [[nodiscard]] std::size_t size() const { return observations_.size(); }
    [[nodiscard]] bool empty() const { return observations_.empty(); }
    [[nodiscard]] std::int64_t frameIndex() const { return frameIndex_; }

   private:
    std::unordered_map<std::uint32_t, Observation> observations_;
    std::int64_t frameIndex_{-1};
};

class KeyframeManager {
   public:
    explicit KeyframeManager(VioParams params);

    [[nodiscard]] bool hasKeyframe() const { return !keyframe_.empty(); }
    [[nodiscard]] std::int64_t keyframeIndex() const { return keyframe_.frameIndex(); }
    [[nodiscard]] std::size_t keyframeSize() const { return keyframe_.size(); }

    /// Make `observations` the new reference keyframe.
    void setKeyframe(const std::vector<Observation>& observations, std::int64_t frameIndex);

    void reset();

    /// Pair every current observation that also exists in the keyframe,
    /// triangulate the keyframe point, and apply the gating rules
    /// (disparity bounds, tracking error, border mask, depth consistency).
    [[nodiscard]] std::vector<Correspondence> buildCorrespondences(const std::vector<Observation>& current,
                                                                   const RectifiedCamera& camera,
                                                                   int imageWidth,
                                                                   int imageHeight) const;

    struct PromotionDecision {
        bool promote{false};
        const char* reason{""};
    };

    /// Decide whether the current frame should become the new keyframe.
    /// Called after the solve, so it can use the inlier count.
    [[nodiscard]] PromotionDecision shouldPromote(int numInliers, double medianParallaxPx, std::int64_t currentFrameIndex) const;

    void setParams(const VioParams& params) { params_ = params; }

   private:
    /// Is this pixel inside the usable (unmasked) region?
    [[nodiscard]] bool insideMask(float u, float v, int imageWidth, int imageHeight) const;

    FeatureFrame keyframe_;
    VioParams params_;
};

}  // namespace oak_vio
