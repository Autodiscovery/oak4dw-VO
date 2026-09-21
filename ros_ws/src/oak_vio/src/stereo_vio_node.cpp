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

/// Map the model name in the parameter file onto the device model-zoo enum.
///
/// A string parameter rather than an int because the names are the only thing
/// documented, and a silent fallback rather than a throw because a typo here
/// should not take the whole VO down with it -- the VO does not need neural
/// depth to run.
dai::DeviceModelZoo neuralDepthModelFromName(const std::string& name) {
    if(name == "NEURAL_DEPTH_NANO") {
        return dai::DeviceModelZoo::NEURAL_DEPTH_NANO;
    }
    if(name == "NEURAL_DEPTH_MEDIUM") {
        return dai::DeviceModelZoo::NEURAL_DEPTH_MEDIUM;
    }
    if(name == "NEURAL_DEPTH_LARGE") {
        return dai::DeviceModelZoo::NEURAL_DEPTH_LARGE;
    }
    if(name == "NEURAL_DEPTH_EXTRA_LARGE") {
        return dai::DeviceModelZoo::NEURAL_DEPTH_EXTRA_LARGE;
    }
    // SMALL is the default and the sensible starting point on a device that is
    // also running VO: the larger models buy resolution the VO cannot use, at a
    // DSP cost that is the whole thing being measured.
    return dai::DeviceModelZoo::NEURAL_DEPTH_SMALL;
}

/// 4x4 extrinsic matrix from calibration to its rotation block.
Mat3 rotationFromExtrinsics(const std::vector<std::vector<float>>& matrix) {
    Mat3 rotation = Mat3::Identity();
    if(matrix.size() < 3) {
        return rotation;
    }
    for(int row = 0; row < 3; ++row) {
        if(matrix[static_cast<std::size_t>(row)].size() < 3) {
            return Mat3::Identity();
        }
        for(int col = 0; col < 3; ++col) {
            rotation(row, col) = static_cast<double>(matrix[static_cast<std::size_t>(row)][static_cast<std::size_t>(col)]);
        }
    }
    return rotation;
}

/// 3x3 matrix from calibration (the rectification rotations) to Mat3.
Mat3 matrixFromCalibration(const std::vector<std::vector<float>>& matrix) {
    return rotationFromExtrinsics(matrix);
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
    featuresKey_ = "features";
    neuralDepthKey_ = "neuralDepth";
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

    // Is a cable plugged into this camera's M8 "IN" port? A physical fact about
    // the rig, which decides who owns the frame rate. Defaults true (slave), the
    // safe assumption: a slave that does not try to set the rate simply runs at
    // the master's rate, whereas a master wrongly told it is slaved silently
    // ignores i_fps, and a slave wrongly told it is master aborts the pipeline.
    fsyncConnected_ = declareOnce<bool>(node, "vio.i_fsync_connected", true);

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

    // ---- Hardware feature tracker (optional) -----------------------------
    // Off by default. It was non-functional on this device across every
    // configuration tried, and it crashed the firmware rather than erroring --
    // see docs/luxonis-bug-featuretracker-rvc4.md. Luxonis have since explained
    // the empty output: the automatic corner-detector threshold yields nothing
    // on RVC4 and the detector has to be configured explicitly. These
    // parameters are that configuration. Confirm it on your device with
    // tools/probe_feature_tracker.py --luxonis-config before enabling.
    useHwTracker_ = declareOnce<bool>(node, "vio.i_use_hw_tracker", false);
    // 20000 is the value Luxonis gave. The scale is NOT normalised -- an
    // earlier probe passed 0.01, on the assumption that it was, and recorded the
    // resulting zero features as evidence that explicit thresholds did not help.
    // Six orders of magnitude out, and the wrong conclusion drawn from it.
    hwTrackerThreshold_ = static_cast<float>(declareOnce<double>(node, "vio.i_hw_tracker_threshold", 20000.0));
    // Both feature counts must be set. numMaxFeatures defaults to ZERO in the
    // bindings, so a CornerDetector that only has numTargetFeatures set caps the
    // detector at no features at all -- silently, and independently of the
    // threshold. That is the second half of why the earlier explicit-threshold
    // test failed.
    hwTrackerMaxFeatures_ = static_cast<int>(declareOnce<int>(node, "vio.i_hw_tracker_max_features", 256));
    hwTrackerHardwareResources_ = static_cast<int>(declareOnce<int>(node, "vio.i_hw_tracker_hardware_resources", -1));
    hwMotionEstimator_ = declareOnce<bool>(node, "vio.i_hw_motion_estimator", true);

    // ---- IMU (Phase 4) ----------------------------------------------------
    const ImuParams imuDefaults;
    declareOnce<bool>(node, "vio.i_imu_enabled", imuDefaults.enabled);
    declareOnce<double>(node, "vio.i_imu_report_rate_hz", imuDefaults.reportRateHz);
    declareOnce<int>(node, "vio.i_imu_batch_report_threshold", 1);
    declareOnce<int>(node, "vio.i_imu_max_batch_reports", 10);
    declareOnce<double>(node, "vio.i_imu_min_sample_coverage", imuDefaults.minSampleCoverage);
    declareOnce<double>(node, "vio.i_imu_max_sample_gap_factor", imuDefaults.maxSampleGapFactor);
    declareOnce<double>(node, "vio.i_imu_rotation_prior_weight", imuDefaults.rotationPriorWeight);
    declareOnce<bool>(node, "vio.i_imu_estimate_gyro_bias", imuDefaults.estimateGyroBias);
    declareOnce<double>(node, "vio.i_imu_static_gyro_threshold", imuDefaults.staticGyroThreshold);
    declareOnce<double>(node, "vio.i_imu_static_accel_tolerance", imuDefaults.staticAccelTolerance);
    declareOnce<int>(node, "vio.i_imu_bias_min_samples", imuDefaults.biasMinSamples);
    declareOnce<double>(node, "vio.i_imu_max_rotation_per_interval_rad", imuDefaults.maxRotationPerIntervalRad);
    declareOnce<bool>(node, "vio.i_imu_use_raw_sensors", true);

    // ---- IMU filter (Phase 4, steps 4-5; scaffold) ------------------------
    const ImuFilterParams filterDefaults;
    declareOnce<bool>(node, "vio.i_imu_filter_enabled", filterDefaults.enabled);
    declareOnce<bool>(node, "vio.i_imu_filter_use_gravity", filterDefaults.useGravityUpdate);
    declareOnce<double>(node, "vio.i_imu_filter_gravity_sigma", filterDefaults.gravitySigma);
    declareOnce<double>(node, "vio.i_imu_filter_gravity_gate", filterDefaults.gravityGateTolerance);
    declareOnce<double>(node, "vio.i_imu_filter_vo_covariance_scale", filterDefaults.voCovarianceScale);
    publishAtImuRate_ = declareOnce<bool>(node, "vio.i_publish_at_imu_rate", false);

    // ---- Neural depth (Phase 5) -------------------------------------------
    enableNeuralDepth_ = declareOnce<bool>(node, "vio.i_enable_neural_depth", false);
    neuralDepthModel_ = declareOnce<std::string>(node, "vio.i_neural_depth_model", "NEURAL_DEPTH_SMALL");
    // "block_matcher" or "neural". Separate from the switch above on purpose:
    // the co-run question (do the engines contend?) and the consumption question
    // (is smoothed neural depth good enough for pose?) have different answers.
    const std::string depthSource = declareOnce<std::string>(node, "vio.i_depth_source", "block_matcher");
    depthSourceNeural_ = depthSource == "neural";

    params_ = readParams();
    imuParams_ = readImuParams();
    imuFilterParams_ = readImuFilterParams();
}

ImuParams StereoVioNode::readImuParams() {
    auto node = getROSNode();
    ImuParams p;
    p.enabled = node->get_parameter("vio.i_imu_enabled").as_bool();
    p.reportRateHz = node->get_parameter("vio.i_imu_report_rate_hz").as_double();
    p.minSampleCoverage = node->get_parameter("vio.i_imu_min_sample_coverage").as_double();
    p.maxSampleGapFactor = node->get_parameter("vio.i_imu_max_sample_gap_factor").as_double();
    p.rotationPriorWeight = node->get_parameter("vio.i_imu_rotation_prior_weight").as_double();
    p.estimateGyroBias = node->get_parameter("vio.i_imu_estimate_gyro_bias").as_bool();
    p.staticGyroThreshold = node->get_parameter("vio.i_imu_static_gyro_threshold").as_double();
    p.staticAccelTolerance = node->get_parameter("vio.i_imu_static_accel_tolerance").as_double();
    p.biasMinSamples = static_cast<int>(node->get_parameter("vio.i_imu_bias_min_samples").as_int());
    p.maxRotationPerIntervalRad = node->get_parameter("vio.i_imu_max_rotation_per_interval_rad").as_double();
    return p;
}

ImuFilterParams StereoVioNode::readImuFilterParams() {
    auto node = getROSNode();
    ImuFilterParams p;
    p.enabled = node->get_parameter("vio.i_imu_filter_enabled").as_bool();
    p.useGravityUpdate = node->get_parameter("vio.i_imu_filter_use_gravity").as_bool();
    p.gravitySigma = node->get_parameter("vio.i_imu_filter_gravity_sigma").as_double();
    p.gravityGateTolerance = node->get_parameter("vio.i_imu_filter_gravity_gate").as_double();
    p.voCovarianceScale = node->get_parameter("vio.i_imu_filter_vo_covariance_scale").as_double();
    return p;
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
    // One parameter, two structs: the estimator needs the blend weight to build
    // the seed and the integrator carries it for reporting, so reading both from
    // the same place is the only way they cannot disagree.
    if(node->has_parameter("vio.i_imu_rotation_prior_weight")) {
        p.gyroPriorWeight = node->get_parameter("vio.i_imu_rotation_prior_weight").as_double();
    }
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
    // vio.i_fsync_connected: true to get back to a running app at whatever rate
    // the external source dictates.
    // FSYNC cable in the IN port means this camera is a slave and the master owns
    // the rate; nothing in IN means it generates FSYNC and owns its own timing, so
    // vio.i_fps applies and gets passed to the sensor.
    const std::optional<float> sensorFps = fsyncConnected_ ? std::nullopt : std::optional<float>(static_cast<float>(fps_));
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

    // ---- Neural depth, co-running (Phase 5) -------------------------------
    //
    // The engines are separate: VO uses the stereo block (and, optionally, the
    // feature-tracker block), while NeuralDepth runs on the DSP. So they should
    // coexist -- but "should" is the reason this is measurable rather than
    // assumed, and why the resource check in tools/phase5_resource_check.py
    // exists.
    //
    // Fed the ALREADY-RECTIFIED pair with its own rectification disabled. Two
    // reasons, one cheap and one that matters: it avoids paying for
    // rectification twice, and it puts the neural depth map in the same frame
    // as our disparity, which is what lets DepthMapSource substitute for
    // DisparityMapSource without touching the camera model.
    if(enableNeuralDepth_) {
        neuralDepth_ = pipeline->create<dai::node::NeuralDepth>();
        neuralDepth_->setRectification(false);
        neuralDepth_->build(stereo_->rectifiedLeft, stereo_->rectifiedRight, neuralDepthModelFromName(neuralDepthModel_));
        RCLCPP_INFO(getLogger(), "Phase 5: NeuralDepth co-running, model %s, rectification disabled (fed the rectified pair)", neuralDepthModel_.c_str());
    }

    // ---- Sync -------------------------------------------------------------
    //
    // Rectified left image, disparity, and -- when the hardware front-end is on
    // -- the tracked features, so the estimator never samples disparity from a
    // different frame than the corners came from. That mismatch produces a slow
    // scale drift which is very hard to attribute after the fact.
    sync_ = pipeline->create<dai::node::Sync>();
    const auto syncWindowMs = static_cast<int>(1500.0 / std::max(1.0, fps_));
    sync_->setSyncThreshold(std::chrono::milliseconds(syncWindowMs));
    stereo_->rectifiedLeft.link(sync_->inputs[rectifiedKey_]);
    stereo_->disparity.link(sync_->inputs[disparityKey_]);

    // ---- Hardware feature tracker (optional) ------------------------------
    if(useHwTracker_) {
        // The ImageManip stays even though rectifiedLeft is already 8-bit grey.
        // It is cheap insurance: feeding this node a format it does not support
        // crashes the firmware instead of raising a node error, and
        // Camera::requestOutput has been seen returning NV12 for a requested
        // GRAY8 on this device. Making the format explicit at the tracker's
        // input costs one manip and removes a class of crash.
        grayManip_ = pipeline->create<dai::node::ImageManip>();
        initialConfigRef(grayManip_->initialConfig).setFrameType(dai::ImgFrame::Type::GRAY8);
        grayManip_->setMaxOutputFrameSize(width_ * height_ * 3);
        stereo_->rectifiedLeft.link(grayManip_->inputImage);

        featureTracker_ = pipeline->create<dai::node::FeatureTracker>();

        // The corner detector is configured EXPLICITLY, and both feature counts
        // are set. This is the whole fix:
        //
        //   * The automatic threshold (initialValue 0) returns empty feature
        //     messages on RVC4 -- the node runs, finds nothing, and says
        //     nothing about why.
        //   * numMaxFeatures defaults to 0 in the bindings, so setting only
        //     numTargetFeatures caps the detector at no features regardless of
        //     the threshold.
        //
        // Either one alone produces exactly the symptom this project spent
        // weeks on, and the earlier "explicit threshold" test hit both at once.
        dai::FeatureTrackerConfig::CornerDetector corner;
        corner.type = dai::FeatureTrackerConfig::CornerDetector::Type::HARRIS;
        corner.numMaxFeatures = hwTrackerMaxFeatures_;
        corner.numTargetFeatures = hwTrackerMaxFeatures_;
        dai::FeatureTrackerConfig::CornerDetector::Thresholds thresholds;
        thresholds.initialValue = hwTrackerThreshold_;
        corner.thresholds = thresholds;
        initialConfigRef(featureTracker_->initialConfig).setCornerDetector(corner);
        initialConfigRef(featureTracker_->initialConfig).setMotionEstimator(hwMotionEstimator_);

        if(hwTrackerHardwareResources_ > 0) {
            featureTracker_->setHardwareResources(hwTrackerHardwareResources_, hwTrackerHardwareResources_);
        }

        grayManip_->out.link(featureTracker_->inputImage);
        featureTracker_->outputFeatures.link(sync_->inputs[featuresKey_]);

        // A second, unsynced tap on the same output. If the tracker emits
        // nothing, Sync never completes a group and onFrame never runs, so the
        // periodic health line that would have reported it never runs either --
        // the VO just goes quiet. This counter plus the startup watchdog is how
        // that becomes a message instead of a mystery.
        featureRateQueue_ = featureTracker_->outputFeatures.createOutputQueue(4, false);
        featureRateQueue_->addCallback([this](const std::shared_ptr<dai::ADatatype>&) { ++featureMessages_; });

        RCLCPP_WARN(getLogger(),
                    "VO feature front-end: RVC4 HARDWARE feature tracker (vio.i_use_hw_tracker=true), Harris, "
                    "threshold %.0f, up to %d features. This node returned zero features and crashed the firmware "
                    "on this device before the explicit corner-detector configuration; if no odometry appears "
                    "within a few seconds, set vio.i_use_hw_tracker=false to fall back to the CPU tracker.",
                    static_cast<double>(hwTrackerThreshold_),
                    hwTrackerMaxFeatures_);
    }

    if(enableNeuralDepth_ && depthSourceNeural_) {
        // Only sync the depth map in when the VO is actually going to read it.
        // Adding an input the estimator ignores would make every frame wait for
        // the DSP for nothing.
        neuralDepth_->depth.link(sync_->inputs[neuralDepthKey_]);
    }

    outputQueue_ = sync_->out.createOutputQueue(4, false);

    RCLCPP_INFO(getLogger(),
                "VO pipeline: %dx%d from %dx%d sensor, sync window %d ms, %s feature tracking, depth from %s",
                width_,
                height_,
                sensorWidth_,
                sensorHeight_,
                syncWindowMs,
                useHwTracker_ ? "RVC4 hardware" : "CPU",
                depthSourceNeural_ ? "NeuralDepth (LENS)" : "the stereo block matcher");

    if(depthSourceNeural_ && !enableNeuralDepth_) {
        RCLCPP_ERROR(getLogger(),
                     "vio.i_depth_source is 'neural' but vio.i_enable_neural_depth is false, so there is no depth "
                     "to read. Falling back to the block matcher.");
        depthSourceNeural_ = false;
    }
    if(depthSourceNeural_) {
        RCLCPP_WARN(getLogger(),
                    "VO depth comes from NeuralDepth. Neural depth is smoothed and, on low texture, partly "
                    "inferred -- good for dense perception, and a locally biased depth at a feature is a biased "
                    "pose. Compare closed-loop drift against the block matcher before trusting this.");
    }

    if(fsyncConnected_) {
        RCLCPP_WARN(getLogger(),
                    "FSYNC: cable IN (vio.i_fsync_connected=true) -- this camera is an FSYNC SLAVE, so the "
                    "master sets the frame rate and vio.i_fps=%.1f has NO EFFECT. To run faster, raise the "
                    "master's rate, or unplug this camera's M8 IN port, drive the others from its OUT, and set "
                    "vio.i_fsync_connected=false. Watch 'camera N Hz' below for the actual rate.",
                    fps_);
    } else {
        RCLCPP_INFO(getLogger(),
                    "FSYNC: cable OUT (vio.i_fsync_connected=false) -- this camera generates FSYNC and owns its "
                    "timing, so the sensor is being set to %.1f FPS and it defines the rate for any camera "
                    "slaved to it.",
                    fps_);
        // The OV9282 pair tops out at 60 FPS at full resolution. Asking for more
        // will fail at pipeline start, which is a confusing place to find out.
        if(fps_ > 60.0) {
            RCLCPP_WARN(getLogger(), "vio.i_fps=%.1f exceeds the 60 FPS the OV9282 pair supports at this resolution; expect pipeline start to fail.", fps_);
        }
        // If the cable is actually still plugged in, the failure lands inside the
        // driver's pipeline start where we cannot catch it, so say now what it will
        // look like.
        RCLCPP_INFO(getLogger(),
                    "If pipeline start now aborts with \"Cannot override fps while using external FSYNC slave "
                    "mode\", the cable is still in this camera's IN port -- set vio.i_fsync_connected=true.");
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

    // ---- IMU (Phase 4) ----------------------------------------------------
    //
    // Deliberately NOT routed through Sync, which is what the plan called for.
    //
    // Sync emits one message per input per completed group, so pairing a 200 Hz
    // IMU against a 10-30 Hz camera through it would deliver one IMU message
    // per frame and discard the rest -- and the rest is the entire point: the
    // prior is an integral over the frame interval, so it needs every sample in
    // that interval, not the one that happened to land nearest the shutter. A
    // batched IMUData message carries several packets, which softens that but
    // does not fix it, since the batching boundaries have nothing to do with the
    // frame boundaries.
    //
    // Its own queue instead, with a callback buffering into ImuIntegrator, and
    // the frame callback integrating over [previous frame, this frame]. That
    // also decouples the frame sync window from the IMU rate, so a late IMU
    // batch delays nothing.
    if(imuParams_.enabled) {
        imu_ = pipeline->create<dai::node::IMU>();

        const auto reportRate = static_cast<int>(std::lround(imuParams_.reportRateHz));
        const bool useRaw = getROSNode()->get_parameter("vio.i_imu_use_raw_sensors").as_bool();
        // RAW by default. The CALIBRATED variants apply the device's own bias
        // correction, which is fine, but then this node's bias estimate and the
        // firmware's are both correcting the same thing and the result is
        // double-counted. Better to take the sensor as it is and own the
        // correction in one place.
        imu_->enableIMUSensor(useRaw ? dai::IMUSensor::GYROSCOPE_RAW : dai::IMUSensor::GYROSCOPE_CALIBRATED, reportRate);
        imu_->enableIMUSensor(useRaw ? dai::IMUSensor::ACCELEROMETER_RAW : dai::IMUSensor::ACCELEROMETER, reportRate);

        // A low batch threshold keeps latency down, which matters because the
        // prior is only useful if the samples for this frame's interval have
        // actually arrived by the time the frame callback runs.
        imu_->setBatchReportThreshold(static_cast<int>(getROSNode()->get_parameter("vio.i_imu_batch_report_threshold").as_int()));
        imu_->setMaxBatchReports(static_cast<int>(getROSNode()->get_parameter("vio.i_imu_max_batch_reports").as_int()));

        // Deep enough to absorb a transport hiccup without dropping samples,
        // since a gap in the stream costs a whole frame's prior.
        imuQueue_ = imu_->out.createOutputQueue(30, false);

        RCLCPP_INFO(getLogger(), "Phase 4: IMU enabled, %s gyro + accelerometer at %d Hz", useRaw ? "raw" : "calibrated", reportRate);
    }
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

bool StereoVioNode::setupImu(const std::shared_ptr<dai::Device>& device) {
    if(!imuParams_.enabled || imu_ == nullptr || imuQueue_ == nullptr) {
        return false;
    }

    imuIntegrator_ = std::make_unique<ImuIntegrator>(imuParams_);

    // ---- IMU to rectified-left rotation -----------------------------------
    //
    // Two rotations, not one, and the second is the one that is easy to forget:
    //
    //   1. getImuToCameraExtrinsics(CAM_B) gives IMU -> RAW left camera.
    //   2. The estimator's pixels live in the RECTIFIED left frame, which
    //      differs from the raw one by the stereo rectification rotation.
    //
    // Skipping step 2 leaves the prior wrong by the rectification angle --
    // typically a couple of degrees, so the prior still looks broadly right and
    // simply underperforms. That is the sort of error that survives for months,
    // which is why both are logged below.
    Mat3 imuToRawLeft = Mat3::Identity();
    Mat3 rectification = Mat3::Identity();
    bool haveExtrinsics = false;

    try {
        auto calibration = device->readCalibration();
        imuToRawLeft = rotationFromExtrinsics(calibration.getImuToCameraExtrinsics(dai::CameraBoardSocket::CAM_B));
        rectification = matrixFromCalibration(calibration.getStereoLeftRectificationRotation());
        haveExtrinsics = true;

        // Device noise parameters where the device has them. They are stored
        // per axis; the filter is isotropic, so take the worst axis -- being
        // over-cautious about noise is the safe direction for a filter, and the
        // axes on a consumer part differ by very little anyway.
        const auto noise = calibration.getImuNoiseParameters();
        const double gyroDensity = std::max({noise.gyroscope.x.noiseDensity, noise.gyroscope.y.noiseDensity, noise.gyroscope.z.noiseDensity});
        const double accelDensity = std::max({noise.accelerometer.x.noiseDensity, noise.accelerometer.y.noiseDensity, noise.accelerometer.z.noiseDensity});
        const double gyroWalk = std::max({noise.gyroscope.x.randomWalk, noise.gyroscope.y.randomWalk, noise.gyroscope.z.randomWalk});
        const double accelWalk = std::max({noise.accelerometer.x.randomWalk, noise.accelerometer.y.randomWalk, noise.accelerometer.z.randomWalk});

        // Zero means the device simply has no stored figures, not a noiseless
        // IMU. Taking them at face value would make the filter infinitely
        // confident in its own propagation.
        if(gyroDensity > 0.0 && accelDensity > 0.0) {
            imuFilterParams_.gyroNoiseDensity = gyroDensity;
            imuFilterParams_.accelNoiseDensity = accelDensity;
            if(gyroWalk > 0.0) {
                imuFilterParams_.gyroBiasRandomWalk = gyroWalk;
            }
            if(accelWalk > 0.0) {
                imuFilterParams_.accelBiasRandomWalk = accelWalk;
            }
            RCLCPP_INFO(getLogger(),
                        "IMU noise from the device: gyro %.3e rad/s/sqrt(Hz), accel %.3e m/s^2/sqrt(Hz)",
                        imuFilterParams_.gyroNoiseDensity,
                        imuFilterParams_.accelNoiseDensity);
        } else {
            RCLCPP_INFO(getLogger(), "The device reports no IMU noise parameters; using the values in params/vio.yaml.");
        }
    } catch(const std::exception& error) {
        // No IMU, or no IMU-to-camera link in the calibration. Either way the
        // VO runs exactly as it does with the IMU switched off, which is the
        // behaviour that was shipping until now -- so this is a warning, not a
        // failure.
        RCLCPP_WARN(getLogger(),
                    "IMU extrinsics unavailable (%s). The gyro prior needs the IMU-to-camera rotation, so it is "
                    "disabled for this run; the VO continues on vision alone.",
                    error.what());
        imuIntegrator_.reset();
        return false;
    }

    const Mat3 imuToRectified = rectification * imuToRawLeft;
    imuIntegrator_->setImuToCameraRotation(imuToRectified);

    const double rectificationAngle = logSO3(rectification).norm() * 180.0 / M_PI;
    RCLCPP_INFO(getLogger(),
                "IMU -> rectified left rotation resolved (extrinsics %s). Stereo rectification contributes %.2f deg; "
                "the raw-camera extrinsic alone would have left the prior wrong by that much.",
                haveExtrinsics ? "read from the device" : "defaulted to identity",
                rectificationAngle);

    if(imuFilterParams_.enabled) {
        imuFilter_ = std::make_unique<ImuFilter>(imuFilterParams_);
        RCLCPP_WARN(getLogger(),
                    "vio.i_imu_filter_enabled is true. The error-state filter is a SCAFFOLD: its propagation and "
                    "gravity update are unit tested, but it has never run on hardware and its VO update treats "
                    "correlated absolute poses as independent fixes, which makes it over-confident. Compare "
                    "closed-loop drift against the filter off before relying on this.");
    }

    imuQueue_->addCallback([this](const std::shared_ptr<dai::ADatatype>& message) { onImuData(message); });
    imuAvailable_ = true;
    return true;
}

void StereoVioNode::onImuData(const std::shared_ptr<dai::ADatatype>& message) {
    const auto data = std::dynamic_pointer_cast<dai::IMUData>(message);
    if(data == nullptr || imuIntegrator_ == nullptr) {
        return;
    }

    const std::lock_guard<std::mutex> lock(imuMutex_);
    for(const auto& packet : data->packets) {
        // Gyro and accelerometer arrive as separate reports inside one packet,
        // each with its own timestamp. The DEVICE clock, not the host one: it is
        // the same clock ImgFrame::getTimestamp() uses, and mixing the two would
        // give a prior with a constant offset -- a plausible-looking number,
        // integrated over the wrong window.
        ImuSample sample;
        sample.angularVelocity = Vec3(packet.gyroscope.x, packet.gyroscope.y, packet.gyroscope.z);
        sample.acceleration = Vec3(packet.acceleroMeter.x, packet.acceleroMeter.y, packet.acceleroMeter.z);
        sample.hasGyro = true;
        sample.hasAccel = true;
        sample.timestampSeconds = std::chrono::duration<double>(packet.gyroscope.getTimestamp().time_since_epoch()).count();

        imuIntegrator_->addSample(sample);
        ++imuSamples_;

        if(imuFilter_ != nullptr && imuFilter_->state().initialised) {
            // The filter works in the rectified camera frame, so both readings
            // are rotated in before they reach it. The gyro bias is removed
            // here from the integrator's estimate; the filter's own bias state
            // then only has to track what is left, rather than re-learning the
            // whole thing from scratch.
            const Vec3 gyroCamera = imuIntegrator_->imuToCameraRotation() * (sample.angularVelocity - imuIntegrator_->gyroBias());
            const Vec3 accelCamera = imuIntegrator_->imuToCameraRotation() * sample.acceleration;
            if(imuFilter_->propagate(gyroCamera, accelCamera, sample.timestampSeconds)) {
                imuFilter_->updateGravity(accelCamera);

                // Phase 4, step 5: odometry at IMU rate rather than frame rate.
                // Published from this thread, which is safe -- rclcpp
                // publishers are thread safe -- and the frame-rate path stands
                // down while this is active so the topic carries one pose per
                // instant.
                if(publishAtImuRate_) {
                    const ImuFilterState& state = imuFilter_->state();
                    const rclcpp::Time stamp(static_cast<std::int64_t>(state.timestampSeconds * 1e9), RCL_ROS_TIME);
                    // Angular velocity is the bias-corrected gyro reading
                    // itself, which is a better answer than the VO's
                    // frame-differenced estimate; linear velocity is a filter
                    // state in its own right.
                    publishOdometry(state.worldFromBody, imuFilter_->poseCovariance(), state.velocity, gyroCamera, stamp);
                }
            }
        }
    }
}

std::vector<Observation> StereoVioNode::observationsFromTrackedFeatures(const std::shared_ptr<dai::TrackedFeatures>& features) const {
    std::vector<Observation> observations;
    if(features == nullptr) {
        return observations;
    }
    observations.reserve(features->trackedFeatures.size());

    for(const auto& feature : features->trackedFeatures) {
        Observation observation;
        // The IDs are the reason this front-end is worth having: the hardware
        // block maintains them across frames, so temporal matching is already
        // solved and the estimator's keyframe logic works unchanged.
        observation.id = feature.id;
        observation.u = feature.position.x;
        observation.v = feature.position.y;
        observation.age = feature.age;
        observation.trackingError = feature.trackingError;
        observation.disparity = 0.0F;  // the caller samples it
        observations.push_back(observation);
    }
    return observations;
}

void StereoVioNode::setupQueues(std::shared_ptr<dai::Device> device) {
    auto node = getROSNode();

    params_ = readParams();
    const RectifiedCamera camera = readCameraModel(device);

    vio_ = std::make_unique<StereoVio>(camera, params_, width_, height_);
    vio_->setRetainDebugData(node->get_parameter("vio.i_publish_debug_features").as_bool());

    // Needed to turn LENS's metric depth back into the equivalent disparity the
    // estimator's noise model is written in. Cached here rather than
    // recomputed per frame, and taken from the same camera model the estimator
    // uses so the two cannot disagree.
    fxBaselineM_ = camera.fx() * camera.baseline();

    if(useHwTracker_) {
        // Nothing to construct: the hardware block maintains the tracks and
        // their IDs itself, and its output arrives through Sync.
        RCLCPP_INFO(getLogger(), "VO feature front-end: RVC4 hardware feature tracker; the CPU tracker is not running.");
    } else {
        // CPU corner detection and tracking, standing in for the hardware
        // block. This is the one part of the pipeline that is not free, so its
        // cost is reported in the periodic health line.
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
    }

    setupImu(device);

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

    // Startup watchdog. Wall-clock rather than frame-driven, because the case
    // it exists for is precisely "no frames are arriving" -- and with the
    // hardware tracker as a Sync input, a tracker that emits nothing stalls
    // Sync and the VO goes silent with no error anywhere. One shot, five
    // seconds, then it cancels itself.
    startupWatchdog_ = node->create_wall_timer(std::chrono::seconds(5), [this]() {
        startupWatchdog_->cancel();
        if(syncedGroups_.load() > 0) {
            return;
        }
        if(useHwTracker_) {
            RCLCPP_ERROR(getLogger(),
                         "No synced frames after 5 s, and the hardware feature tracker is enabled. It has produced "
                         "%lu feature messages and the camera %lu frames. Zero feature messages means the tracker "
                         "is not emitting, which is the failure this device showed before: set "
                         "vio.i_use_hw_tracker=false to fall back to the CPU tracker. Feature messages arriving but "
                         "no synced groups means a timestamp mismatch instead -- raise the sync window.",
                         static_cast<unsigned long>(featureMessages_.load()),
                         static_cast<unsigned long>(cameraFrames_.load()));
        } else {
            RCLCPP_WARN(getLogger(),
                        "No synced frames after 5 s (camera %lu frames). If the camera count is also zero the "
                        "sensors are not delivering; if it is not, the stereo node or Sync is the holdup.",
                        static_cast<unsigned long>(cameraFrames_.load()));
        }
    });

    RCLCPP_INFO(getLogger(),
                "Stereo VO running: %dx%d @ %.1f FPS, keyframe mode %s, %s front-end, gyro prior %s",
                width_,
                height_,
                fps_,
                params_.keyframeReferenced ? "on" : "off (frame-to-frame)",
                useHwTracker_ ? "hardware" : "CPU",
                imuAvailable_ ? "on" : "off");
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
    // The IMU queue goes first, and before the integrator it writes into: its
    // callback runs on a DepthAI thread, so destroying the integrator while a
    // callback is in flight would be a use-after-free.
    if(imuQueue_) {
        imuQueue_->close();
        imuQueue_.reset();
    }
    if(featureRateQueue_) {
        featureRateQueue_->close();
        featureRateQueue_.reset();
    }
    if(startupWatchdog_) {
        startupWatchdog_->cancel();
        startupWatchdog_.reset();
    }
    {
        const std::lock_guard<std::mutex> lock(imuMutex_);
        imuFilter_.reset();
        imuIntegrator_.reset();
        imuAvailable_ = false;
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

    // The IMU's runtime tunables follow too, so the gyro prior can be weighted
    // or switched off live during an A/B without restarting the app. What
    // cannot change at runtime is whether the IMU node EXISTS -- that is a
    // pipeline-construction decision, so vio.i_imu_enabled only takes effect on
    // restart, and saying so beats appearing to accept it.
    {
        const std::lock_guard<std::mutex> lock(imuMutex_);
        const ImuParams updated = readImuParams();
        if(imuIntegrator_ != nullptr) {
            const bool wasEnabled = imuParams_.enabled;
            imuParams_ = updated;
            imuIntegrator_->setParams(imuParams_);
            if(updated.enabled != wasEnabled) {
                RCLCPP_WARN(getLogger(), "vio.i_imu_enabled changed at runtime. The IMU node is created at pipeline build, so this takes effect on restart.");
            }
        } else {
            imuParams_ = updated;
        }
        imuFilterParams_ = readImuFilterParams();
        if(imuFilter_ != nullptr) {
            imuFilter_->setParams(imuFilterParams_);
        }
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
    if(group == nullptr || vio_ == nullptr) {
        return;
    }
    // The CPU tracker is only constructed when it is the selected front-end, so
    // this guard has to depend on which one is running rather than on the
    // pointer alone.
    if(!useHwTracker_ && tracker_ == nullptr) {
        return;
    }
    ++syncedGroups_;

    const auto rectified = group->get<dai::ImgFrame>(rectifiedKey_);
    const auto disparityFrame = group->get<dai::ImgFrame>(disparityKey_);
    if(rectified == nullptr || disparityFrame == nullptr) {
        return;
    }

    // ---- Corners: hardware block or CPU ----------------------------------
    std::vector<Observation> observations;
    const auto trackStart = std::chrono::steady_clock::now();

    if(useHwTracker_) {
        const auto features = group->get<dai::TrackedFeatures>(featuresKey_);
        if(features == nullptr) {
            RCLCPP_WARN_THROTTLE(getLogger(),
                                 *getROSNode()->get_clock(),
                                 3000,
                                 "No TrackedFeatures in the synced group. The hardware tracker is enabled but not "
                                 "delivering; set vio.i_use_hw_tracker=false to fall back to the CPU tracker.");
            return;
        }
        observations = observationsFromTrackedFeatures(features);
    } else {
        const cv::Mat gray = wrapGrayFrame(rectified);
        if(gray.empty()) {
            RCLCPP_WARN_THROTTLE(getLogger(), *getROSNode()->get_clock(), 3000, "rectifiedLeft frame unusable; skipping");
            return;
        }
        observations = tracker_->track(gray);
    }

    const double trackMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - trackStart).count();
    trackerMsEma_ = trackerMsEma_ <= 0.0 ? trackMs : (0.9 * trackerMsEma_ + 0.1 * trackMs);

    // ---- Sample depth at each corner -------------------------------------
    //
    // Both sources present the same interface, so the estimator cannot tell
    // which one answered. The difference that matters is not the plumbing but
    // the statistics: block-matched disparity is a measurement, neural depth is
    // smoothed and in places inferred, and a locally biased depth at a feature
    // is a biased pose.
    std::unique_ptr<DepthSource> depthSource;
    if(depthSourceNeural_) {
        const auto depthFrame = group->get<dai::ImgFrame>(neuralDepthKey_);
        if(depthFrame == nullptr) {
            RCLCPP_WARN_THROTTLE(getLogger(), *getROSNode()->get_clock(), 3000, "vio.i_depth_source is 'neural' but no depth frame arrived in the synced group; skipping.");
            return;
        }
        depthSource = std::make_unique<DepthMapSource>(reinterpret_cast<const std::uint16_t*>(depthFrame->getData().data()),
                                                       static_cast<int>(depthFrame->getWidth()),
                                                       static_cast<int>(depthFrame->getHeight()),
                                                       static_cast<int>(depthFrame->getWidth()),
                                                       fxBaselineM_);
    } else {
        depthSource = std::make_unique<DisparityMapSource>(reinterpret_cast<const std::uint16_t*>(disparityFrame->getData().data()),
                                                           static_cast<int>(disparityFrame->getWidth()),
                                                           static_cast<int>(disparityFrame->getHeight()),
                                                           static_cast<int>(disparityFrame->getWidth()),
                                                           subpixelFractionalBits_);
    }

    const std::size_t tracked = observations.size();
    for(Observation& observation : observations) {
        observation.disparity = depthSource->disparityAt(observation.u, observation.v);
    }
    observations.erase(std::remove_if(observations.begin(),
                                      observations.end(),
                                      [](const Observation& observation) { return observation.disparity <= 0.0F; }),
                       observations.end());

    const auto deviceStamp = disparityFrame->getTimestamp();
    const double seconds = std::chrono::duration<double>(deviceStamp.time_since_epoch()).count();

    // ---- Gyro rotation prior (Phase 4) -----------------------------------
    //
    // Integrated over exactly the interval between this frame and the last,
    // using device timestamps on both ends so the window matches the motion the
    // corners actually saw.
    RotationPrior rotationPrior;
    if(imuAvailable_ && haveLastFrameStamp_) {
        const std::lock_guard<std::mutex> lock(imuMutex_);
        if(imuIntegrator_ != nullptr) {
            rotationPrior = imuIntegrator_->deltaRotation(lastFrameStamp_, seconds);
        }
    }
    if(rotationPrior.valid) {
        ++gyroPriorsUsed_;
    } else if(imuAvailable_ && haveLastFrameStamp_) {
        ++gyroPriorsRejected_;
        lastGyroRejection_ = rotationPrior.rejection;
    }
    lastFrameStamp_ = seconds;
    haveLastFrameStamp_ = true;

    const VioFrameResult result = vio_->processFrame(observations, seconds, rotationPrior);

    // ---- Filter (Phase 4, scaffold) --------------------------------------
    if(imuFilter_ != nullptr && result.valid) {
        const std::lock_guard<std::mutex> lock(imuMutex_);
        if(!imuFilter_->state().initialised) {
            // Anchor on the first good VO pose rather than on identity, so the
            // filter starts where the VO already is instead of converging to it.
            imuFilter_->initialise(result.worldFromCamera, seconds, imuIntegrator_ != nullptr ? imuIntegrator_->gyroBias() : Vec3::Zero());
        } else {
            imuFilter_->updatePose(result.worldFromCamera, result.poseCovariance);
        }
    }

    const rclcpp::Time stamp(std::chrono::duration_cast<std::chrono::nanoseconds>(deviceStamp.time_since_epoch()).count(), RCL_ROS_TIME);
    publish(result, stamp);

    if(observations.empty()) {
        RCLCPP_WARN_THROTTLE(getLogger(),
                             *getROSNode()->get_clock(),
                             3000,
                             "No usable observations: the %s tracker found %zu corners, none survived the depth "
                             "lookup. Corners but no depth means %s has no support at those pixels; no corners at "
                             "all means the image lacks texture or is badly exposed.",
                             useHwTracker_ ? "hardware" : "CPU",
                             tracked,
                             depthSourceNeural_ ? "the neural depth map" : "the stereo block");
    }

    // ---- Periodic health line --------------------------------------------
    const auto now = std::chrono::steady_clock::now();
    ++framesSinceLog_;
    if(lastRateLog_.time_since_epoch().count() == 0) {
        lastRateLog_ = now;
    } else if(now - lastRateLog_ >= std::chrono::seconds(5)) {
        const double elapsed = std::chrono::duration<double>(now - lastRateLog_).count();
        const std::uint64_t cameraFrames = cameraFrames_.exchange(0);
        const double cameraHz = static_cast<double>(cameraFrames) / elapsed;

        // When we claim to own the timing, check we actually got what we asked
        // for. Setting a rate and never verifying it is how the original 10 Hz
        // went unnoticed for so long -- requestOutput accepted 30 and delivered
        // 10 without a word.
        if(!fsyncConnected_ && cameraHz > 1.0 && std::abs(cameraHz - fps_) > 0.2 * fps_) {
            RCLCPP_WARN_THROTTLE(getLogger(),
                                 *getROSNode()->get_clock(),
                                 10000,
                                 "Requested %.1f FPS but the camera is delivering %.1f Hz. Either the sensor cannot "
                                 "sustain this rate at %dx%d, or vio.i_fsync_connected is wrong and something is "
                                 "driving this camera's FSYNC after all.",
                                 fps_,
                                 cameraHz,
                                 sensorWidth_,
                                 sensorHeight_);
        }

        RCLCPP_INFO(getLogger(),
                    "VO: %.1f Hz (camera %.1f, requested %.1f) | corners %zu tracked, %zu with depth, %d inliers "
                    "| %s tracker %.1f ms, estimator %.2f ms | exposure %.1f ms ISO %d",
                    static_cast<double>(framesSinceLog_) / elapsed,
                    cameraHz,
                    fps_,
                    tracked,
                    observations.size(),
                    result.numInliers,
                    useHwTracker_ ? "HW" : "CPU",
                    trackerMsEma_,
                    result.solveMilliseconds,
                    static_cast<double>(lastExposureUs_.load()) / 1000.0,
                    lastSensitivityIso_.load());

        // A separate line for the IMU, and it reports the REJECTIONS as well as
        // the uses. A prior that is being rejected every frame looks exactly
        // like a prior that does not help, so the distinction has to be visible
        // rather than inferred from drift numbers.
        if(imuAvailable_) {
            // Read and reset the counters into locals FIRST. Argument
            // evaluation order is unspecified in C++, so doing the exchange
            // inline alongside another read of the same atomic would print
            // whichever the compiler happened to sequence first.
            const std::uint64_t samples = imuSamples_.exchange(0);
            const std::uint64_t priorsUsed = gyroPriorsUsed_.exchange(0);
            const std::uint64_t priorsRejected = gyroPriorsRejected_.exchange(0);

            const std::lock_guard<std::mutex> lock(imuMutex_);
            const Vec3 bias = imuIntegrator_ != nullptr ? imuIntegrator_->gyroBias() : Vec3::Zero();
            const bool biasReady = imuIntegrator_ != nullptr && imuIntegrator_->biasReady();
            RCLCPP_INFO(getLogger(),
                        "IMU: %.1f Hz (%lu samples) | prior used %lu, rejected %lu%s%s | gyro bias %s [%.4f %.4f %.4f] rad/s",
                        static_cast<double>(samples) / elapsed,
                        static_cast<unsigned long>(samples),
                        static_cast<unsigned long>(priorsUsed),
                        static_cast<unsigned long>(priorsRejected),
                        priorsRejected > 0 && lastGyroRejection_[0] != '\0' ? " last reason: " : "",
                        priorsRejected > 0 ? lastGyroRejection_ : "",
                        biasReady ? "measured" : "still warming up",
                        bias.x(),
                        bias.y(),
                        bias.z());
        }

        framesSinceLog_ = 0;
        lastRateLog_ = now;
    }
}

void StereoVioNode::publish(const VioFrameResult& result, const rclcpp::Time& stamp) {
    // With the filter publishing at IMU rate, publishing here too would put two
    // poses for the same instant on one topic and leave a consumer to guess.
    // The status message still goes out every frame either way, since that is
    // the VO's own health and the filter does not produce it.
    if(!(publishAtImuRate_ && imuFilter_ != nullptr && imuFilter_->state().initialised)) {
        publishOdometry(result.worldFromCamera, result.poseCovariance, result.linearVelocity, result.angularVelocity, stamp);
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

    status.frontend = useHwTracker_ ? oak_vio_msgs::msg::VioStatus::FRONTEND_HARDWARE : oak_vio_msgs::msg::VioStatus::FRONTEND_CPU;
    status.tracker_ms = static_cast<float>(trackerMsEma_);
    status.depth_source = depthSourceNeural_ ? oak_vio_msgs::msg::VioStatus::DEPTH_NEURAL : oak_vio_msgs::msg::VioStatus::DEPTH_BLOCK_MATCHER;

    status.imu_available = imuAvailable_;
    status.gyro_prior_used = result.usedGyroPrior;
    status.gyro_prior_angle_rad = static_cast<float>(result.gyroPriorAngleRad);
    status.gyro_prior_rejection = result.gyroPriorRejection;
    if(imuAvailable_) {
        const std::lock_guard<std::mutex> lock(imuMutex_);
        if(imuIntegrator_ != nullptr) {
            status.gyro_bias_ready = imuIntegrator_->biasReady();
            const Vec3 bias = imuIntegrator_->gyroBias();
            status.gyro_bias[0] = static_cast<float>(bias.x());
            status.gyro_bias[1] = static_cast<float>(bias.y());
            status.gyro_bias[2] = static_cast<float>(bias.z());
        }
    }
    status.imu_filter_active = imuFilter_ != nullptr && imuFilter_->state().initialised && publishAtImuRate_;

    statusPublisher_->publish(status);

    if(result.state == TrackingState::Lost) {
        RCLCPP_WARN_THROTTLE(getLogger(), *getROSNode()->get_clock(), 2000, "VO tracking lost: %s (inliers %d/%d)", result.note, result.numInliers, result.numSelected);
    }
}

void StereoVioNode::publishOdometry(const Pose& worldFromCamera, const Mat6& poseCovariance, const Vec3& linearVelocity, const Vec3& angularVelocity, const rclcpp::Time& stamp) {
    // Convert from the camera optical convention into the ROS body convention
    // so RViz and robot_localization see something sensible.
    Mat3 rotation = worldFromCamera.R;
    Vec3 translation = worldFromCamera.t;
    Vec3 linear = linearVelocity;
    Vec3 angular = angularVelocity;
    Mat6 covariance = poseCovariance;

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
}

void StereoVioNode::handleReset(const std::shared_ptr<std_srvs::srv::Trigger::Request>& request, const std::shared_ptr<std_srvs::srv::Trigger::Response>& response) {
    (void)request;
    if(vio_ == nullptr) {
        response->success = false;
        response->message = "VO not initialised";
        return;
    }
    vio_->reset();
    if(tracker_) {
        tracker_->reset();
    }

    {
        const std::lock_guard<std::mutex> lock(imuMutex_);
        if(imuIntegrator_ != nullptr) {
            // Buffered samples go, the BIAS ESTIMATE stays. The bias is a
            // property of the sensor, not of the track, and throwing away a
            // converged estimate on every reset would mean re-earning it
            // through another second of stillness that the caller has no reason
            // to provide.
            imuIntegrator_->reset();
        }
        if(imuFilter_ != nullptr) {
            imuFilter_->reset();
        }
    }
    haveLastFrameStamp_ = false;

    response->success = true;
    response->message = "VO reset to origin";
    RCLCPP_INFO(getLogger(), "Stereo VO reset by service call");
}

}  // namespace oak_vio
