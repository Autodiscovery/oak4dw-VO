// CPU feature detection and tracking, replacing dai::node::FeatureTracker.
//
// The RVC4 hardware tracker is non-functional on this firmware -- it returns
// zero corners for every input and asserts on its hardware session. See
// docs/luxonis-bug-featuretracker-rvc4.md. This does the same job on the ARM
// cores with OpenCV, which is Apache-2.0 and therefore fine commercially.
//
// The contract is deliberately identical to what the hardware node was supposed
// to provide: a list of observations carrying a STABLE ID across frames, an age,
// and a tracking error. That is the whole interface the estimator depends on, so
// nothing downstream of here changes -- keyframing, RANSAC, the Jacobians and the
// covariance work are all untouched.
//
// Cost: roughly 8-12 ms per frame on one core at 640x400 with ~300 features,
// against the ~0 the hardware block would have cost. That is the price of the
// firmware bug, and it is why tracking runs at quarter resolution.
#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "oak_vio/types.hpp"

namespace oak_vio {

struct CpuTrackerParams {
    // ---- Detection (goodFeaturesToTrack) ---------------------------------
    int maxFeatures{320};
    /// Corner strength floor, relative to the best corner in the image.
    double qualityLevel{0.01};
    /// Minimum separation between corners, in pixels. Doubles as the redetection
    /// mask radius so new corners are not piled onto existing tracks.
    double minDistance{12.0};
    int blockSize{7};
    /// Harris rather than Shi-Tomasi, matching what the hardware block did and
    /// what the StereoScan paper assumes.
    bool useHarris{true};
    double harrisK{0.04};
    /// Refine detections to subpixel accuracy. Cheap at these counts and worth
    /// it: corner localisation error propagates straight into pose error.
    bool refineSubPixel{true};

    // ---- Tracking (calcOpticalFlowPyrLK) ---------------------------------
    int lkWindowSize{21};
    int lkPyramidLevels{3};
    int lkMaxIterations{30};
    double lkEpsilon{0.01};
    /// Drop tracks whose LK residual exceeds this. Same units and role as the
    /// hardware node's trackingError.
    float maxTrackingError{30.0F};
    /// Reject tracks that jump further than this between frames; at 30 FPS no
    /// real feature moves this far, so it is a mistracking filter.
    double maxDisplacementPx{80.0};
    /// Bidirectional consistency check: track forward then back, and drop
    /// points that do not return to where they started. Catches the mistracks
    /// that a low LK residual alone will happily accept.
    bool forwardBackwardCheck{true};
    double forwardBackwardThresholdPx{1.0};

    // ---- Redetection ------------------------------------------------------
    /// Top up with fresh corners once surviving tracks fall below this.
    int redetectBelow{200};
    /// Ignore a border of this many pixels. LK is unreliable at the edges, and
    /// on the wide lens the rectified periphery is stretched.
    int borderMargin{12};
};

/// Detects corners and tracks them frame to frame, maintaining stable IDs.
///
/// Not thread safe: call track() from one thread only.
class CpuFeatureTracker {
   public:
    explicit CpuFeatureTracker(CpuTrackerParams params = {});

    /// Track into `gray` (CV_8UC1) and return the surviving observations.
    /// `disparity` is left at zero — the caller samples it.
    std::vector<Observation> track(const cv::Mat& gray);

    /// Drop all tracks. IDs are not reused.
    void reset();

    [[nodiscard]] std::size_t activeTracks() const { return points_.size(); }
    [[nodiscard]] std::uint64_t totalDetected() const { return totalDetected_; }
    void setParams(const CpuTrackerParams& params) { params_ = params; }

   private:
    /// Top up to maxFeatures, masking out the neighbourhood of existing tracks.
    void detect(const cv::Mat& gray);

    CpuTrackerParams params_;
    cv::Mat previousGray_;
    std::vector<cv::Point2f> points_;
    std::vector<std::uint32_t> ids_;
    std::vector<std::uint32_t> ages_;
    std::vector<float> errors_;
    /// IDs start at 1 so 0 can mean "unset".
    std::uint32_t nextId_{1};
    std::uint64_t totalDetected_{0};
};

}  // namespace oak_vio
