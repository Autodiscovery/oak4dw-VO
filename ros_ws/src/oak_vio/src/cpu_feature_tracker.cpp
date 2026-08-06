#include "oak_vio/cpu_feature_tracker.hpp"

#include <algorithm>
#include <utility>

#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace oak_vio {

CpuFeatureTracker::CpuFeatureTracker(CpuTrackerParams params) : params_(std::move(params)) {}

void CpuFeatureTracker::reset() {
    previousGray_.release();
    points_.clear();
    ids_.clear();
    ages_.clear();
    errors_.clear();
}

void CpuFeatureTracker::detect(const cv::Mat& gray) {
    const int wanted = params_.maxFeatures - static_cast<int>(points_.size());
    if(wanted <= 0) {
        return;
    }

    // Mask out the border and the neighbourhood of every existing track, so a
    // top-up adds corners in genuinely new places rather than duplicating the
    // ones already being followed.
    cv::Mat mask(gray.size(), CV_8UC1, cv::Scalar(255));
    const int margin = std::max(0, params_.borderMargin);
    if(margin > 0) {
        mask(cv::Rect(0, 0, gray.cols, margin)).setTo(0);
        mask(cv::Rect(0, gray.rows - margin, gray.cols, margin)).setTo(0);
        mask(cv::Rect(0, 0, margin, gray.rows)).setTo(0);
        mask(cv::Rect(gray.cols - margin, 0, margin, gray.rows)).setTo(0);
    }
    for(const cv::Point2f& point : points_) {
        cv::circle(mask, point, static_cast<int>(params_.minDistance), cv::Scalar(0), -1);
    }

    std::vector<cv::Point2f> fresh;
    cv::goodFeaturesToTrack(gray, fresh, wanted, params_.qualityLevel, params_.minDistance, mask, params_.blockSize, params_.useHarris, params_.harrisK);

    if(fresh.empty()) {
        return;
    }

    if(params_.refineSubPixel) {
        // Corner localisation error propagates directly into pose error, and
        // this is cheap at these counts.
        cv::cornerSubPix(gray,
                         fresh,
                         cv::Size(5, 5),
                         cv::Size(-1, -1),
                         cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, 20, 0.03));
    }

    for(const cv::Point2f& point : fresh) {
        points_.push_back(point);
        ids_.push_back(nextId_++);
        ages_.push_back(1);
        errors_.push_back(0.0F);
    }
    totalDetected_ += fresh.size();
}

std::vector<Observation> CpuFeatureTracker::track(const cv::Mat& gray) {
    std::vector<Observation> observations;
    if(gray.empty() || gray.type() != CV_8UC1) {
        return observations;
    }

    const cv::TermCriteria criteria(cv::TermCriteria::EPS | cv::TermCriteria::COUNT, params_.lkMaxIterations, params_.lkEpsilon);
    const cv::Size window(params_.lkWindowSize, params_.lkWindowSize);

    if(!previousGray_.empty() && !points_.empty()) {
        std::vector<cv::Point2f> tracked;
        std::vector<std::uint8_t> status;
        std::vector<float> error;
        cv::calcOpticalFlowPyrLK(previousGray_, gray, points_, tracked, status, error, window, params_.lkPyramidLevels, criteria);

        // Track back again and require the round trip to return to where it
        // started. A mistracked corner can have a perfectly small forward
        // residual, so this catches failures the error threshold does not.
        std::vector<std::uint8_t> backStatus;
        std::vector<cv::Point2f> backTracked;
        if(params_.forwardBackwardCheck) {
            std::vector<float> backError;
            cv::calcOpticalFlowPyrLK(gray, previousGray_, tracked, backTracked, backStatus, backError, window, params_.lkPyramidLevels, criteria);
        }

        std::vector<cv::Point2f> keptPoints;
        std::vector<std::uint32_t> keptIds;
        std::vector<std::uint32_t> keptAges;
        std::vector<float> keptErrors;
        keptPoints.reserve(points_.size());
        keptIds.reserve(points_.size());
        keptAges.reserve(points_.size());
        keptErrors.reserve(points_.size());

        const double margin = params_.borderMargin;
        for(std::size_t i = 0; i < tracked.size(); ++i) {
            if(status[i] == 0) {
                continue;
            }
            if(error[i] > params_.maxTrackingError) {
                continue;
            }
            const cv::Point2f& point = tracked[i];
            if(point.x < margin || point.y < margin || point.x >= static_cast<float>(gray.cols) - margin || point.y >= static_cast<float>(gray.rows) - margin) {
                continue;
            }
            if(cv::norm(point - points_[i]) > params_.maxDisplacementPx) {
                continue;
            }
            if(params_.forwardBackwardCheck) {
                if(backStatus[i] == 0 || cv::norm(backTracked[i] - points_[i]) > params_.forwardBackwardThresholdPx) {
                    continue;
                }
            }

            keptPoints.push_back(point);
            keptIds.push_back(ids_[i]);
            keptAges.push_back(ages_[i] + 1);
            keptErrors.push_back(error[i]);
        }

        points_ = std::move(keptPoints);
        ids_ = std::move(keptIds);
        ages_ = std::move(keptAges);
        errors_ = std::move(keptErrors);
    }

    // Top up when tracks have thinned, and always on the first frame.
    if(points_.empty() || static_cast<int>(points_.size()) < params_.redetectBelow) {
        detect(gray);
    }

    observations.reserve(points_.size());
    for(std::size_t i = 0; i < points_.size(); ++i) {
        Observation observation;
        observation.id = ids_[i];
        observation.u = points_[i].x;
        observation.v = points_[i].y;
        observation.age = ages_[i];
        observation.trackingError = errors_[i];
        observation.disparity = 0.0F;  // the caller samples this
        observations.push_back(observation);
    }

    gray.copyTo(previousGray_);
    return observations;
}

}  // namespace oak_vio
