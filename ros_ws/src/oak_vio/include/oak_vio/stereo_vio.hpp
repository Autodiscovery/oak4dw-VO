// The estimator's front door.
//
// Feed it the tracked features (with disparity already sampled) for one frame;
// get back a world pose, a covariance and a health report. It owns the
// keyframe, the motion model and the pose integrator, and knows nothing about
// ROS or DepthAI — which is what makes the whole thing testable on a
// workstation against synthetic trajectories.
#pragma once

#include <cstdint>
#include <vector>

#include "oak_vio/keyframe_manager.hpp"
#include "oak_vio/motion_model.hpp"
#include "oak_vio/pose_integrator.hpp"
#include "oak_vio/ransac.hpp"
#include "oak_vio/rectified_camera.hpp"
#include "oak_vio/types.hpp"
#include "oak_vio/vio_params.hpp"

namespace oak_vio {

struct VioFrameResult {
    /// True when this frame produced a usable motion estimate. When false the
    /// pose is held at its last value and the covariance is inflated.
    bool valid{false};
    TrackingState state{TrackingState::Initialising};

    Pose worldFromCamera;
    Mat6 poseCovariance{Mat6::Zero()};

    /// The raw keyframe-relative estimate, before integration.
    Pose keyframeFromCurrent;

    Vec3 linearVelocity{Vec3::Zero()};
    Vec3 angularVelocity{Vec3::Zero()};

    int numObservations{0};
    int numCorrespondences{0};
    int numSelected{0};
    int numInliers{0};
    double inlierRatio{0.0};
    double medianParallaxPx{0.0};

    bool keyframePromoted{false};
    const char* note{""};

    double solveMilliseconds{0.0};
    std::int64_t frameIndex{-1};
    double timestampSeconds{0.0};
};

class StereoVio {
   public:
    StereoVio(RectifiedCamera camera, VioParams params, int imageWidth, int imageHeight);

    /// Process one frame. `observations` are the tracked features in the
    /// rectified left image with disparity already sampled; see
    /// DepthSource for where that comes from.
    VioFrameResult processFrame(const std::vector<Observation>& observations, double timestampSeconds);

    /// Drop all state and restart from the given world pose.
    void reset(const Pose& worldFromCamera = Pose{});

    void setParams(const VioParams& params);
    [[nodiscard]] const VioParams& params() const { return params_; }
    [[nodiscard]] const RectifiedCamera& camera() const { return camera_; }

    /// Correspondences used on the last frame, and which of them were
    /// inliers. For the debug feature topic; empty unless
    /// setRetainDebugData(true) has been called.
    void setRetainDebugData(bool enabled) { retainDebug_ = enabled; }
    [[nodiscard]] const std::vector<Correspondence>& lastSelected() const { return lastSelected_; }
    [[nodiscard]] const std::vector<int>& lastInlierIndices() const { return lastInlierIndices_; }

    [[nodiscard]] std::int64_t frameIndex() const { return frameIndex_; }
    [[nodiscard]] int consecutiveFailures() const { return consecutiveFailures_; }

   private:
    /// Re-anchor the keyframe on the current observations.
    void promoteKeyframe(const std::vector<Observation>& observations, bool anchorPose);

    RectifiedCamera camera_;
    VioParams params_;
    int imageWidth_;
    int imageHeight_;

    KeyframeManager keyframes_;
    RansacMotionSolver solver_;
    MotionModel motionModel_;
    PoseIntegrator integrator_;

    /// Last accepted keyframe->current estimate, used both as the constant-
    /// velocity composition base and to derive the inter-frame delta.
    Pose keyframeFromPrevious_;
    bool haveKeyframeFromPrevious_{false};

    /// Has any frame ever produced a usable estimate?
    ///
    /// Distinguishes "starting up" from "lost": without it, a frame carrying no
    /// observations after minutes of good tracking reports Initialising, which
    /// tells a downstream filter to wait for a first pose rather than to hold
    /// the last one.
    bool everTracked_{false};

    std::int64_t frameIndex_{-1};
    double lastTimestamp_{0.0};
    bool haveLastTimestamp_{false};
    int consecutiveFailures_{0};

    bool retainDebug_{false};
    std::vector<Correspondence> lastSelected_;
    std::vector<int> lastInlierIndices_;

    /// After this many consecutive failures, re-anchor the keyframe on the
    /// current frame so tracking can restart. The world pose is left where it
    /// was — we have lost track, not teleported.
    static constexpr int kFailuresBeforeReanchor = 3;
};

}  // namespace oak_vio
