#include "oak_vio/stereo_vio_node.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <utility>

#include "oak_vio/depth_source.hpp"

namespace oak_vio {
namespace {

/// Declare a parameter only if it is not already declared, and return its value.
template <typename T>
T declareOnce(const std::shared_ptr<rclcpp::Node>& node, const std::string& name, const T& fallback) {
    if(!node->has_parameter(name)) {
        return node->declare_parameter<T>(name, fallback);
    }
    return node->get_parameter(name).get_value<T>();
}

geometry_msgs::msg::Quaternion toQuaternion(const Mat3& rotation) {
    const Eigen::Quaterniond q(rotation);
    geometry_msgs::msg::Quaternion out;
    out.x = q.x();
    out.y = q.y();
    out.z = q.z();
    out.w = q.w();
    return out;
}

/// ROS covariance arrays are row-major 6x6 with (x, y, z, rx, ry, rz)
/// ordering, which matches our (rho, phi) parameterisation directly.
void fillCovariance(const Mat6& source, std::array<double, 36>& target) {
    for(int row = 0; row < 6; ++row) {
        for(int col = 0; col < 6; ++col) {
            target[static_cast<std::size_t>(row) * 6 + static_cast<std::size_t>(col)] = source(row, col);
        }
    }
}

}  // namespace

Mat3 StereoVioNode::opticalToRos() {
    // x_ros = z_opt, y_ros = -x_opt, z_ros = -y_opt
    Mat3 r;
    // clang-format off
    r <<  0.0,  0.0, 1.0,
         -1.0,  0.0, 0.0,
          0.0, -1.0, 0.0;
    // clang-format on
    return r;
}

StereoVioNode::StereoVioNode(const std::string& daiNodeName,
                             std::shared_ptr<rclcpp::Node> node,
                             std::shared_ptr<dai::Pipeline> pipeline,
                             const std::string& deviceName,
                             bool rsCompatibility)
    : BaseNode(daiNodeName, node, pipeline, deviceName, rsCompatibility) {
    RCLCPP_DEBUG(getLogger(), "Creating stereo VO node %s", daiNodeName.c_str());
    setNames();
    declareParams();
    setInOut(pipeline);
}

StereoVioNode::~StereoVioNode() = default;

void StereoVioNode::setNames() {
    syncQueueName_ = getName() + "_sync";
    disparityKey_ = "disparity";
    rectifiedKey_ = "rectifiedLeft";
}

void StereoVioNode::declareParams() {
    auto node = getROSNode();

    // 640x400, not 1280x800. Corner detection and tracking now run on the ARM
    // cores because the hardware block is broken, so resolution costs CPU
    // directly -- roughly 8-12 ms/frame here against 25-40 at full resolution.
    // The price is halving the focal length, which doubles depth sigma (about
    // 29 cm to 59 cm at 5 m). Raise these if you would rather spend the CPU.
    width_ = declareOnce<int>(node, "vio.i_width", 640);
    height_ = declareOnce<int>(node, "vio.i_height", 400);
    fps_ = declareOnce<double>(node, "vio.i_fps", 30.0);

    // Sensor mode, set explicitly rather than left to auto-selection. Must be a
    // resolution the OV9282 actually supports; 1280x800 is its native full frame.
    // Keep the same aspect ratio as i_width/i_height so the downscale is a pure
    // resize -- see the note on STRETCH in setInOut.
    sensorWidth_ = declareOnce<int>(node, "vio.i_sensor_width", 1280);
    sensorHeight_ = declareOnce<int>(node, "vio.i_sensor_height", 800);

    // Defaults FALSE because the OAK 4 D W tested here is in external FSYNC slave
    // mode: the rate comes from an external sync source and cannot be set from
    // software. Attempting it aborts the pipeline with "Cannot override fps while
    // using external FSYNC slave mode", with or without an explicit sensor
    // resolution. Set true on a device that owns its own timing.
    declareOnce<bool>(node, "vio.i_set_sensor_fps", false);

    // StereoDepth subpixel setting. This MUST match what DisparityMapSource
    // divides by, or every depth is scaled by a power of two and the
    // trajectory comes out the right shape at the wrong scale.
    subpixelFractionalBits_ = declareOnce<int>(node, "vio.i_subpixel_fractional_bits", 5);

    // Negative means "leave StereoDepth's default alpha alone". Phase 0 sets
    // this: it trades retained field of view against rectification stretch on
    // the wide lens.
    alphaScaling_ = static_cast<float>(declareOnce<double>(node, "vio.i_alpha_scaling", -1.0));

    odomFrame_ = declareOnce<std::string>(node, "vio.i_odom_frame", "odom");
    baseFrame_ = declareOnce<std::string>(node, "vio.i_base_frame", getFrameName("vio"));
    publishTf_ = declareOnce<bool>(node, "vio.i_publish_tf", true);
    rosConvention_ = declareOnce<bool>(node, "vio.i_publish_ros_convention", true);

    const VioParams defaults;
    declareOnce<double>(node, "vio.i_min_disparity_px", defaults.minDisparityPx);
    declareOnce<double>(node, "vio.i_max_disparity_px", defaults.maxDisparityPx);
    declareOnce<double>(node, "vio.i_max_tracking_error", defaults.maxTrackingError);
    declareOnce<double>(node, "vio.i_mask_border_fraction", defaults.maskBorderFraction);
    declareOnce<int>(node, "vio.i_bucket_cols", defaults.bucketCols);
    declareOnce<int>(node, "vio.i_bucket_rows", defaults.bucketRows);
    declareOnce<int>(node, "vio.i_max_features_per_bucket", defaults.maxFeaturesPerBucket);
    declareOnce<int>(node, "vio.i_ransac_min_iterations", defaults.ransacMinIterations);
    declareOnce<int>(node, "vio.i_ransac_max_iterations", defaults.ransacMaxIterations);
    declareOnce<double>(node, "vio.i_ransac_inlier_threshold_px", defaults.ransacInlierThresholdPx);
    declareOnce<double>(node, "vio.i_ransac_confidence", defaults.ransacConfidence);
    declareOnce<int>(node, "vio.i_min_inliers", defaults.minInliers);
    declareOnce<double>(node, "vio.i_min_inlier_ratio", defaults.minInlierRatio);
    declareOnce<int>(node, "vio.i_gn_max_iterations", defaults.gnMaxIterations);
    declareOnce<double>(node, "vio.i_huber_delta_px", defaults.huberDeltaPx);
    declareOnce<bool>(node, "vio.i_keyframe_referenced", defaults.keyframeReferenced);
    declareOnce<int>(node, "vio.i_keyframe_min_tracked", defaults.keyframeMinTracked);
    declareOnce<double>(node, "vio.i_keyframe_max_parallax_px", defaults.keyframeMaxParallaxPx);
    declareOnce<int>(node, "vio.i_keyframe_max_age_frames", defaults.keyframeMaxAgeFrames);
    declareOnce<double>(node, "vio.i_max_translation_per_span_m", defaults.maxTranslationPerSpanM);
    declareOnce<double>(node, "vio.i_max_rotation_per_span_rad", defaults.maxRotationPerSpanRad);
    declareOnce<double>(node, "vio.i_pixel_sigma_px", defaults.pixelSigmaPx);
    declareOnce<double>(node, "vio.i_disparity_sigma_px", defaults.disparitySigmaPx);
    declareOnce<double>(node, "vio.i_covariance_inflation", defaults.covarianceInflation);
    declareOnce<bool>(node, "vio.i_use_constant_velocity_seed", defaults.useConstantVelocitySeed);
    declareOnce<bool>(node, "vio.i_publish_debug_features", false);

    // CPU feature tracker.
    const CpuTrackerParams trackerDefaults;
    declareOnce<int>(node, "vio.i_max_features", trackerDefaults.maxFeatures);
    declareOnce<double>(node, "vio.i_quality_level", trackerDefaults.qualityLevel);
    declareOnce<double>(node, "vio.i_min_feature_distance_px", trackerDefaults.minDistance);
    declareOnce<int>(node, "vio.i_detector_block_size", trackerDefaults.blockSize);
    declareOnce<bool>(node, "vio.i_use_harris", trackerDefaults.useHarris);
    declareOnce<bool>(node, "vio.i_refine_subpixel", trackerDefaults.refineSubPixel);
    declareOnce<int>(node, "vio.i_lk_window_size", trackerDefaults.lkWindowSize);
    declareOnce<int>(node, "vio.i_lk_pyramid_levels", trackerDefaults.lkPyramidLevels);
    declareOnce<double>(node, "vio.i_max_displacement_px", trackerDefaults.maxDisplacementPx);
    declareOnce<bool>(node, "vio.i_forward_backward_check", trackerDefaults.forwardBackwardCheck);
    declareOnce<double>(node, "vio.i_forward_backward_threshold_px", trackerDefaults.forwardBackwardThresholdPx);
    declareOnce<int>(node, "vio.i_redetect_below", trackerDefaults.redetectBelow);
    declareOnce<int>(node, "vio.i_border_margin_px", trackerDefaults.borderMargin);

    params_ = readParams();
}

CpuTrackerParams StereoVioNode::readTrackerParams() {
    auto node = getROSNode();
    CpuTrackerParams p;
    p.maxFeatures = static_cast<int>(node->get_parameter("vio.i_max_features").as_int());
    p.qualityLevel = node->get_parameter("vio.i_quality_level").as_double();
    p.minDistance = node->get_parameter("vio.i_min_feature_distance_px").as_double();
    p.blockSize = static_cast<int>(node->get_parameter("vio.i_detector_block_size").as_int());
    p.useHarris = node->get_parameter("vio.i_use_harris").as_bool();
    p.refineSubPixel = node->get_parameter("vio.i_refine_subpixel").as_bool();
    p.lkWindowSize = static_cast<int>(node->get_parameter("vio.i_lk_window_size").as_int());
    p.lkPyramidLevels = static_cast<int>(node->get_parameter("vio.i_lk_pyramid_levels").as_int());
    p.maxTrackingError = static_cast<float>(node->get_parameter("vio.i_max_tracking_error").as_double());
    p.maxDisplacementPx = node->get_parameter("vio.i_max_displacement_px").as_double();
    p.forwardBackwardCheck = node->get_parameter("vio.i_forward_backward_check").as_bool();
    p.forwardBackwardThresholdPx = node->get_parameter("vio.i_forward_backward_threshold_px").as_double();
    p.redetectBelow = static_cast<int>(node->get_parameter("vio.i_redetect_below").as_int());
    p.borderMargin = static_cast<int>(node->get_parameter("vio.i_border_margin_px").as_int());
    return p;
}

VioParams StereoVioNode::readParams() {
    auto node = getROSNode();
    VioParams p;
    p.minDisparityPx = static_cast<float>(node->get_parameter("vio.i_min_disparity_px").as_double());
    p.maxDisparityPx = static_cast<float>(node->get_parameter("vio.i_max_disparity_px").as_double());
    p.maxTrackingError = static_cast<float>(node->get_parameter("vio.i_max_tracking_error").as_double());
    p.maskBorderFraction = node->get_parameter("vio.i_mask_border_fraction").as_double();
    p.bucketCols = static_cast<int>(node->get_parameter("vio.i_bucket_cols").as_int());
    p.bucketRows = static_cast<int>(node->get_parameter("vio.i_bucket_rows").as_int());
    p.maxFeaturesPerBucket = static_cast<int>(node->get_parameter("vio.i_max_features_per_bucket").as_int());
    p.ransacMinIterations = static_cast<int>(node->get_parameter("vio.i_ransac_min_iterations").as_int());
    p.ransacMaxIterations = static_cast<int>(node->get_parameter("vio.i_ransac_max_iterations").as_int());
    p.ransacInlierThresholdPx = node->get_parameter("vio.i_ransac_inlier_threshold_px").as_double();
    p.ransacConfidence = node->get_parameter("vio.i_ransac_confidence").as_double();
    p.minInliers = static_cast<int>(node->get_parameter("vio.i_min_inliers").as_int());
    p.minInlierRatio = node->get_parameter("vio.i_min_inlier_ratio").as_double();
    p.gnMaxIterations = static_cast<int>(node->get_parameter("vio.i_gn_max_iterations").as_int());
    p.huberDeltaPx = node->get_parameter("vio.i_huber_delta_px").as_double();
    p.keyframeReferenced = node->get_parameter("vio.i_keyframe_referenced").as_bool();
    p.keyframeMinTracked = static_cast<int>(node->get_parameter("vio.i_keyframe_min_tracked").as_int());
    p.keyframeMaxParallaxPx = node->get_parameter("vio.i_keyframe_max_parallax_px").as_double();
    p.keyframeMaxAgeFrames = static_cast<int>(node->get_parameter("vio.i_keyframe_max_age_frames").as_int());
    p.maxTranslationPerSpanM = node->get_parameter("vio.i_max_translation_per_span_m").as_double();
    p.maxRotationPerSpanRad = node->get_parameter("vio.i_max_rotation_per_span_rad").as_double();
    p.pixelSigmaPx = node->get_parameter("vio.i_pixel_sigma_px").as_double();
    p.disparitySigmaPx = node->get_parameter("vio.i_disparity_sigma_px").as_double();
    p.covarianceInflation = node->get_parameter("vio.i_covariance_inflation").as_double();
    p.useConstantVelocitySeed = node->get_parameter("vio.i_use_constant_velocity_seed").as_bool();
    return p;
}

void StereoVioNode::setInOut(std::shared_ptr<dai::Pipeline> pipeline) {
    // ---- Cameras ---------------------------------------------------------
    // Sensor FPS goes to the FSYNC MASTER only.
    //
    // Setting it on both threw at pipeline start:
    //
    //     RPC 'startPipeline' failed: Cannot override fps while using external
    //     FSYNC slave mode
    //
    // On OAK-D-style stereo devices the pair is hardware-synchronised internally:
    // the left sensor has FSYNC as an INPUT and the right drives it as an OUTPUT.
    // So CAM_B is a slave by construction, its rate is dictated by CAM_C, and
    // trying to override it is correctly refused. The master defines the rate;
    // the slave must be left unset.
    //
    // This is also the explanation for the 10 Hz the app has been stuck at all
    // along: requestOutput's fps argument is ignored on this device (as is its
    // pixel-format argument), so nothing was ever actually setting the sensor
    // rate. It was not a throughput limit, which is why it did not move between
    // 1280x800 and 640x400, and not auto-exposure, which sat at 8.3 ms.
    //
    // If this still throws, the device is in *external* FSYNC slave mode -- driven
    // by a signal on the M8 connector rather than by its own right sensor -- and
    // the rate cannot be set from software at all. Set
    // vio.i_set_sensor_fps: false to get back to a running app at whatever rate
    // the external source dictates.
    const bool setSensorFps = getROSNode()->get_parameter("vio.i_set_sensor_fps").as_bool();
    const std::optional<float> sensorFps = setSensorFps ? std::optional<float>(static_cast<float>(fps_)) : std::nullopt;
    const auto sensorResolution = std::make_pair(sensorWidth_, sensorHeight_);

    leftCamera_ = pipeline->create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_B, sensorResolution, sensorFps);
    rightCamera_ = pipeline->create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_C, sensorResolution, sensorFps);

    // Explicit mono format, and STRETCH rather than CROP.
    //
    // GRAY8 is requested explicitly now that the sensor is configured explicitly.
    // An earlier attempt to request it was silently downgraded to NV12, but that
    // was with the sensor resolution left unset.
    //
    // The resize mode matters for correctness, not just framing. CROP takes a
    // centre window out of the sensor image, which would halve the field of view
    // at 640x400 from a 1280x800 sensor -- and, worse, getCameraIntrinsics()
    // returns intrinsics SCALED for the requested size (it reported fx=284.08 at
    // 640x400, exactly half of 568.15 at 1280x800). Scaled intrinsics describe a
    // downscaled image, not a cropped one, so CROP leaves the camera model
    // inconsistent with the pixels. STRETCH is a true resize and matches what the
    // intrinsics assume.
    //
    // 640x400 and 1280x800 share an aspect ratio of 1.6, so STRETCH here is an
    // exact 2x downscale with no distortion. The check below catches the case
    // where someone picks a size that would actually stretch.
    const double sensorAspect = static_cast<double>(sensorWidth_) / static_cast<double>(sensorHeight_);
    const double outputAspect = static_cast<double>(width_) / static_cast<double>(height_);
    if(std::abs(sensorAspect - outputAspect) > 0.01) {
        RCLCPP_WARN(getLogger(),
                    "Output %dx%d (aspect %.3f) does not match sensor %dx%d (aspect %.3f). STRETCH will distort "
                    "the image and the calibration will not describe it. Pick an output size with the sensor's "
                    "aspect ratio.",
                    width_,
                    height_,
                    outputAspect,
                    sensorWidth_,
                    sensorHeight_,
                    sensorAspect);
    }

    const auto requested = std::make_pair(width_, height_);
    auto* leftOutput = leftCamera_->requestOutput(requested, dai::ImgFrame::Type::GRAY8, dai::ImgResizeMode::STRETCH, static_cast<float>(fps_));
    auto* rightOutput = rightCamera_->requestOutput(requested, dai::ImgFrame::Type::GRAY8, dai::ImgResizeMode::STRETCH, static_cast<float>(fps_));

    // ---- Stereo ----------------------------------------------------------
    stereo_ = pipeline->create<dai::node::StereoDepth>();
    stereo_->setDefaultProfilePreset(dai::node::StereoDepth::PresetMode::HIGH_DETAIL);
    stereo_->setRectification(true);
    stereo_->setLeftRightCheck(true);
    stereo_->setExtendedDisparity(false);
    stereo_->setSubpixel(true);
    stereo_->setSubpixelFractionalBits(subpixelFractionalBits_);
    stereo_->setDepthAlign(dai::CameraBoardSocket::CAM_B);
    if(alphaScaling_ >= 0.0F) {
        stereo_->setAlphaScaling(alphaScaling_);
    }

    leftOutput->link(stereo_->left);
    rightOutput->link(stereo_->right);

    // ---- Sync: rectified left image + disparity ---------------------------
    //
    // No FeatureTracker and no ImageManip in this graph. The RVC4 hardware
    // feature tracker is non-functional on this firmware -- it returns zero
    // corners for any input and asserts on its hardware session, see
    // docs/luxonis-bug-featuretracker-rvc4.md -- so corners are found on the CPU
    // in onFrame. That also removes the need for the format conversion, since
    // OpenCV reads RAW8 as single-channel 8-bit directly.
    //
    // Both inputs now come off the same StereoDepth node so their timestamps
    // agree closely, but the window still matters: sampling disparity from a
    // different frame than the corners came from produces a slow scale drift
    // that is very hard to attribute after the fact.
    sync_ = pipeline->create<dai::node::Sync>();
    const auto syncWindowMs = static_cast<int>(1500.0 / std::max(1.0, fps_));
    sync_->setSyncThreshold(std::chrono::milliseconds(syncWindowMs));
    stereo_->rectifiedLeft.link(sync_->inputs[rectifiedKey_]);
    stereo_->disparity.link(sync_->inputs[disparityKey_]);
    outputQueue_ = sync_->out.createOutputQueue(4, false);

    if(setSensorFps) {
        RCLCPP_INFO(getLogger(), "VO pipeline: %dx%d from %dx%d sensor @ %.1f FPS requested, sync window %d ms, CPU feature tracking", width_, height_, sensorWidth_, sensorHeight_, fps_, syncWindowMs);
    } else {
        RCLCPP_WARN(getLogger(),
                    "VO pipeline: %dx%d from %dx%d sensor, sync window %d ms, CPU feature tracking. Sensor FPS is "
                    "NOT being set (vio.i_set_sensor_fps is false), so vio.i_fps=%.1f is not the rate you will get -- "
                    "an external FSYNC source dictates it. Watch the 'camera N Hz' figure below for the real rate.",
                    width_,
                    height_,
                    sensorWidth_,
                    sensorHeight_,
                    syncWindowMs,
                    fps_);
    }

    // Camera rate and exposure. Retained because the sensor delivering 10 Hz
    // against 30 requested is still unexplained, and is independent of the
    // tracker fault -- every stage measured at 10 Hz, so the camera is the limit.
    cameraRateQueue_ = leftOutput->createOutputQueue(1, false);
    cameraRateQueue_->addCallback([this](const std::shared_ptr<dai::ADatatype>& message) {
        ++cameraFrames_;
        if(const auto frame = std::dynamic_pointer_cast<dai::ImgFrame>(message)) {
            lastExposureUs_ = static_cast<long>(std::chrono::duration_cast<std::chrono::microseconds>(frame->getExposureTime()).count());
            lastSensitivityIso_ = frame->getSensitivity();
        }
    });
}

RectifiedCamera StereoVioNode::readCameraModel(const std::shared_ptr<dai::Device>& device) {
    auto calibration = device->readCalibration();

    // Intrinsics of the rectified left image. StereoDepth rectifies the left
    // image into the right camera's rectified frame, so the intrinsics that
    // describe the rectified output come from CAM_C. Phase 0 verifies this
    // against the driver's published camera_info; if they disagree, trust
    // camera_info and change it here.
    const auto intrinsics = calibration.getCameraIntrinsics(dai::CameraBoardSocket::CAM_C, width_, height_);
    const double fx = intrinsics[0][0];
    const double fy = intrinsics[1][1];
    const double cx = intrinsics[0][2];
    const double cy = intrinsics[1][2];

    // getBaselineDistance() returns centimetres.
    const double baselineMetres = static_cast<double>(calibration.getBaselineDistance()) * 0.01;

    RCLCPP_INFO(getLogger(),
                "VO camera model: fx=%.2f fy=%.2f cx=%.2f cy=%.2f baseline=%.4f m (%dx%d)",
                fx,
                fy,
                cx,
                cy,
                baselineMetres,
                width_,
                height_);

    const RectifiedCamera camera(fx, fy, cx, cy, baselineMetres);

    // Raw sensor intrinsics and rectified intrinsics are easy to confuse here,
    // and confusing them yields a trajectory of plausible shape and wrong
    // scale. The tell is which projection model the numbers are consistent
    // with: a fisheye has fx ~= fy and an equidistant FoV matching the
    // datasheet, whereas a rectified pinhole has a noticeably smaller implied
    // FoV. Log both readings so the ambiguity is visible rather than silent.
    const double halfWidth = 0.5 * width_;
    const double hfovPinhole = 2.0 * std::atan(halfWidth / fx) * 180.0 / M_PI;
    const double hfovEquidistant = 2.0 * (halfWidth / fx) * 180.0 / M_PI;
    RCLCPP_INFO(getLogger(),
                "  implied HFoV: %.1f deg if pinhole, %.1f deg if equidistant fisheye",
                hfovPinhole,
                hfovEquidistant);
    if(hfovEquidistant > 110.0 && std::abs(fx - fy) < 0.02 * fx) {
        RCLCPP_WARN(getLogger(),
                    "  These look like RAW fisheye intrinsics (fx ~= fy, equidistant HFoV %.0f deg "
                    "matches the datasheet), not rectified pinhole intrinsics. The estimator assumes "
                    "a rectified pinhole. Run tools/phase0_calibration_check.py rectify and compare "
                    "against the driver's rectified camera_info before trusting absolute scale.",
                    hfovEquidistant);
    }

    // State the depth precision this configuration actually gives, so nobody
    // has to rediscover it from a drifting trajectory. Note this uses the
    // disparity *noise*, not the subpixel quantisation step.
    for(const double range : {2.0, 5.0, 10.0}) {
        RCLCPP_INFO(getLogger(),
                    "  depth sigma at %.0f m: %.2f cm (%.1f%%)",
                    range,
                    camera.depthSigmaAt(range, params_.disparitySigmaPx) * 100.0,
                    100.0 * camera.depthSigmaAt(range, params_.disparitySigmaPx) / range);
    }

    return camera;
}

void StereoVioNode::setupQueues(std::shared_ptr<dai::Device> device) {
    auto node = getROSNode();

    params_ = readParams();
    const RectifiedCamera camera = readCameraModel(device);

    vio_ = std::make_unique<StereoVio>(camera, params_, width_, height_);
    vio_->setRetainDebugData(node->get_parameter("vio.i_publish_debug_features").as_bool());

    // CPU corner detection and tracking, standing in for the broken RVC4
    // hardware block. This is the one part of the pipeline that is not free, so
    // its cost is reported in the periodic health line.
    const CpuTrackerParams trackerParams = readTrackerParams();
    tracker_ = std::make_unique<CpuFeatureTracker>(trackerParams);
    RCLCPP_INFO(getLogger(),
                "VO feature front-end: CPU (OpenCV Harris + Lucas-Kanade), up to %d features, "
                "redetect below %d, %dx%d LK window over %d pyramid levels",
                trackerParams.maxFeatures,
                trackerParams.redetectBelow,
                trackerParams.lkWindowSize,
                trackerParams.lkWindowSize,
                trackerParams.lkPyramidLevels);

    odometryPublisher_ = node->create_publisher<nav_msgs::msg::Odometry>("~/vo/odometry", rclcpp::QoS(10));
    statusPublisher_ = node->create_publisher<oak_vio_msgs::msg::VioStatus>("~/vo/status", rclcpp::QoS(10));
    tfBroadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(node);

    resetService_ = node->create_service<std_srvs::srv::Trigger>(
        "~/vo/reset",
        [this](const std::shared_ptr<std_srvs::srv::Trigger::Request> request, std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
            handleReset(request, response);
        });

    outputQueue_->addCallback([this](const std::shared_ptr<dai::ADatatype>& message) {
        onFrame(std::dynamic_pointer_cast<dai::MessageGroup>(message));
    });

    RCLCPP_INFO(getLogger(), "Stereo VO running: %dx%d @ %.1f FPS, keyframe mode %s", width_, height_, fps_, params_.keyframeReferenced ? "on" : "off (frame-to-frame)");
}

void StereoVioNode::closeQueues() {
    if(outputQueue_) {
        outputQueue_->close();
        outputQueue_.reset();
    }
    if(cameraRateQueue_) {
        cameraRateQueue_->close();
        cameraRateQueue_.reset();
    }
    tracker_.reset();
    vio_.reset();
}

void StereoVioNode::updateParams(const std::vector<rclcpp::Parameter>& params) {
    bool touched = false;
    for(const auto& parameter : params) {
        if(parameter.get_name().rfind("vio.", 0) == 0) {
            touched = true;
            break;
        }
    }
    if(!touched || !vio_) {
        return;
    }
    params_ = readParams();
    vio_->setParams(params_);
    if(tracker_) {
        tracker_->setParams(readTrackerParams());
    }
    RCLCPP_INFO(getLogger(), "Stereo VO parameters updated");
}

cv::Mat StereoVioNode::wrapGrayFrame(const std::shared_ptr<dai::ImgFrame>& frame) {
    const auto width = static_cast<int>(frame->getWidth());
    const auto height = static_cast<int>(frame->getHeight());
    if(width <= 0 || height <= 0) {
        return {};
    }

    const auto& data = frame->getData();
    const std::size_t needed = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if(data.size() < needed) {
        return {};
    }

    // Row stride is taken as the width, NOT derived from data.size(): that
    // returns the buffer CAPACITY, not the image size, so dividing it by the
    // height would give a bogus stride on any over-allocated buffer. Warn if the
    // capacity is not an exact fit, since that is the case where real padding
    // would shear the image.
    if(data.size() != needed) {
        RCLCPP_WARN_ONCE(getLogger(),
                         "rectifiedLeft buffer is %zu bytes for a %dx%d image (%zu expected). Assuming a tight "
                         "row stride; if tracking behaves as though the image were skewed, the frame is padded "
                         "and the stride needs to come from the frame metadata instead.",
                         data.size(),
                         width,
                         height,
                         needed);
    }

    return cv::Mat(height, width, CV_8UC1, const_cast<std::uint8_t*>(data.data()), static_cast<std::size_t>(width));
}

void StereoVioNode::onFrame(const std::shared_ptr<dai::MessageGroup>& group) {
    if(group == nullptr || vio_ == nullptr || tracker_ == nullptr) {
        return;
    }

    const auto rectified = group->get<dai::ImgFrame>(rectifiedKey_);
    const auto disparityFrame = group->get<dai::ImgFrame>(disparityKey_);
    if(rectified == nullptr || disparityFrame == nullptr) {
        return;
    }

    const cv::Mat gray = wrapGrayFrame(rectified);
    if(gray.empty()) {
        RCLCPP_WARN_THROTTLE(getLogger(), *getROSNode()->get_clock(), 3000, "rectifiedLeft frame unusable; skipping");
        return;
    }

    // ---- Corner detection and tracking, on the CPU -----------------------
    const auto trackStart = std::chrono::steady_clock::now();
    std::vector<Observation> observations = tracker_->track(gray);
    const double trackMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - trackStart).count();
    trackerMsEma_ = trackerMsEma_ <= 0.0 ? trackMs : (0.9 * trackerMsEma_ + 0.1 * trackMs);

    // ---- Sample disparity at each corner ---------------------------------
    const DisparityMapSource depthSource(reinterpret_cast<const std::uint16_t*>(disparityFrame->getData().data()),
                                         static_cast<int>(disparityFrame->getWidth()),
                                         static_cast<int>(disparityFrame->getHeight()),
                                         static_cast<int>(disparityFrame->getWidth()),
                                         subpixelFractionalBits_);

    const std::size_t tracked = observations.size();
    for(Observation& observation : observations) {
        observation.disparity = depthSource.disparityAt(observation.u, observation.v);
    }
    observations.erase(std::remove_if(observations.begin(),
                                      observations.end(),
                                      [](const Observation& observation) { return observation.disparity <= 0.0F; }),
                       observations.end());

    const auto deviceStamp = disparityFrame->getTimestamp();
    const double seconds = std::chrono::duration<double>(deviceStamp.time_since_epoch()).count();

    const VioFrameResult result = vio_->processFrame(observations, seconds);

    const rclcpp::Time stamp(std::chrono::duration_cast<std::chrono::nanoseconds>(deviceStamp.time_since_epoch()).count(), RCL_ROS_TIME);
    publish(result, stamp);

    if(observations.empty()) {
        RCLCPP_WARN_THROTTLE(getLogger(),
                             *getROSNode()->get_clock(),
                             3000,
                             "No usable observations: CPU tracker found %zu corners, none survived the disparity "
                             "lookup. Corners but no disparity means the stereo block has no support at those "
                             "pixels; no corners at all means the image lacks texture or is badly exposed.",
                             tracked);
    }

    // ---- Periodic health line --------------------------------------------
    const auto now = std::chrono::steady_clock::now();
    ++framesSinceLog_;
    if(lastRateLog_.time_since_epoch().count() == 0) {
        lastRateLog_ = now;
    } else if(now - lastRateLog_ >= std::chrono::seconds(5)) {
        const double elapsed = std::chrono::duration<double>(now - lastRateLog_).count();
        const std::uint64_t cameraFrames = cameraFrames_.exchange(0);
        RCLCPP_INFO(getLogger(),
                    "VO: %.1f Hz (camera %.1f, requested %.1f) | corners %zu tracked, %zu with disparity, %d inliers "
                    "| tracker %.1f ms, estimator %.2f ms | exposure %.1f ms ISO %d",
                    static_cast<double>(framesSinceLog_) / elapsed,
                    static_cast<double>(cameraFrames) / elapsed,
                    fps_,
                    tracked,
                    observations.size(),
                    result.numInliers,
                    trackerMsEma_,
                    result.solveMilliseconds,
                    static_cast<double>(lastExposureUs_.load()) / 1000.0,
                    lastSensitivityIso_.load());
        framesSinceLog_ = 0;
        lastRateLog_ = now;
    }
}

void StereoVioNode::publish(const VioFrameResult& result, const rclcpp::Time& stamp) {
    // Convert from the camera optical convention into the ROS body convention
    // so RViz and robot_localization see something sensible.
    Mat3 rotation = result.worldFromCamera.R;
    Vec3 translation = result.worldFromCamera.t;
    Vec3 linear = result.linearVelocity;
    Vec3 angular = result.angularVelocity;
    Mat6 covariance = result.poseCovariance;

    if(rosConvention_) {
        const Mat3 c = opticalToRos();
        rotation = c * rotation * c.transpose();
        translation = c * translation;
        linear = c * linear;
        angular = c * angular;
        Mat6 adjoint = Mat6::Zero();
        adjoint.topLeftCorner<3, 3>() = c;
        adjoint.bottomRightCorner<3, 3>() = c;
        covariance = adjoint * covariance * adjoint.transpose();
    }

    nav_msgs::msg::Odometry odometry;
    odometry.header.stamp = stamp;
    odometry.header.frame_id = odomFrame_;
    odometry.child_frame_id = baseFrame_;
    odometry.pose.pose.position.x = translation.x();
    odometry.pose.pose.position.y = translation.y();
    odometry.pose.pose.position.z = translation.z();
    odometry.pose.pose.orientation = toQuaternion(rotation);
    fillCovariance(covariance, odometry.pose.covariance);

    odometry.twist.twist.linear.x = linear.x();
    odometry.twist.twist.linear.y = linear.y();
    odometry.twist.twist.linear.z = linear.z();
    odometry.twist.twist.angular.x = angular.x();
    odometry.twist.twist.angular.y = angular.y();
    odometry.twist.twist.angular.z = angular.z();
    // Twist uncertainty is not separately estimated; reuse the relative pose
    // covariance as a stand-in and mark it clearly in the docs rather than
    // publishing zeros, which a filter would read as perfect knowledge.
    fillCovariance(covariance, odometry.twist.covariance);

    odometryPublisher_->publish(odometry);

    if(publishTf_) {
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = stamp;
        transform.header.frame_id = odomFrame_;
        transform.child_frame_id = baseFrame_;
        transform.transform.translation.x = translation.x();
        transform.transform.translation.y = translation.y();
        transform.transform.translation.z = translation.z();
        transform.transform.rotation = odometry.pose.pose.orientation;
        tfBroadcaster_->sendTransform(transform);
    }

    oak_vio_msgs::msg::VioStatus status;
    status.header.stamp = stamp;
    status.header.frame_id = baseFrame_;
    status.state = static_cast<std::uint8_t>(result.state);
    status.note = result.note;
    status.num_observations = static_cast<std::uint32_t>(result.numObservations);
    status.num_correspondences = static_cast<std::uint32_t>(result.numCorrespondences);
    status.num_selected = static_cast<std::uint32_t>(result.numSelected);
    status.num_inliers = static_cast<std::uint32_t>(result.numInliers);
    status.inlier_ratio = static_cast<float>(result.inlierRatio);
    status.keyframe_promoted = result.keyframePromoted;
    status.keyframe_index = result.frameIndex;
    status.median_parallax_px = static_cast<float>(result.medianParallaxPx);
    status.solve_ms = static_cast<float>(result.solveMilliseconds);
    status.frame_index = result.frameIndex;
    statusPublisher_->publish(status);

    if(result.state == TrackingState::Lost) {
        RCLCPP_WARN_THROTTLE(getLogger(), *getROSNode()->get_clock(), 2000, "VO tracking lost: %s (inliers %d/%d)", result.note, result.numInliers, result.numSelected);
    }
}

void StereoVioNode::handleReset(const std::shared_ptr<std_srvs::srv::Trigger::Request>& request, const std::shared_ptr<std_srvs::srv::Trigger::Response>& response) {
    (void)request;
    if(vio_ == nullptr) {
        response->success = false;
        response->message = "VO not initialised";
        return;
    }
    vio_->reset();
    response->success = true;
    response->message = "VO reset to origin";
    RCLCPP_INFO(getLogger(), "Stereo VO reset by service call");
}

}  // namespace oak_vio
