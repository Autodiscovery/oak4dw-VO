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
    featuresKey_ = "features";
}

void StereoVioNode::declareParams() {
    auto node = getROSNode();

    width_ = declareOnce<int>(node, "vio.i_width", 1280);
    height_ = declareOnce<int>(node, "vio.i_height", 800);
    fps_ = declareOnce<double>(node, "vio.i_fps", 30.0);
    enableIr_ = declareOnce<bool>(node, "vio.i_enable_ir", false);

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
    declareOnce<int>(node, "vio.i_num_target_features", 320);
    declareOnce<bool>(node, "vio.i_publish_debug_features", false);

    params_ = readParams();
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
    leftCamera_ = pipeline->create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_B);
    rightCamera_ = pipeline->create<dai::node::Camera>()->build(dai::CameraBoardSocket::CAM_C);

    // GRAY8, not NV12. The stereo pair is a mono OV9282; asking for a colour
    // format forces a needless conversion and gives the stereo block an input
    // it does not want. This was producing frames at the right rate with no
    // usable disparity in them -- everything looked alive while nothing worked.
    const auto requested = std::make_pair(width_, height_);
    auto* leftOutput = leftCamera_->requestOutput(requested, dai::ImgFrame::Type::GRAY8, dai::ImgResizeMode::CROP, static_cast<float>(fps_));
    auto* rightOutput = rightCamera_->requestOutput(requested, dai::ImgFrame::Type::GRAY8, dai::ImgResizeMode::CROP, static_cast<float>(fps_));

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

    // ---- Feature tracker -------------------------------------------------
    // Runs on the rectified left image, so its coordinates are already in the
    // same frame the estimator's camera model describes.
    featureTracker_ = pipeline->create<dai::node::FeatureTracker>();
    const auto targetFeatures = static_cast<int>(getROSNode()->get_parameter("vio.i_num_target_features").as_int());
    featureTracker_->initialConfig->setNumTargetFeatures(targetFeatures);
    featureTracker_->initialConfig->setMotionEstimator(true);
    stereo_->rectifiedLeft.link(featureTracker_->inputImage);

    // ---- Sync ------------------------------------------------------------
    // Pairs disparity with the feature list by timestamp. Without this the
    // estimator would eventually sample disparity from a different frame than
    // the features came from, which produces a slow, baffling scale drift.
    sync_ = pipeline->create<dai::node::Sync>();
    sync_->setSyncThreshold(std::chrono::milliseconds(static_cast<int>(500.0 / std::max(1.0, fps_))));
    stereo_->disparity.link(sync_->inputs[disparityKey_]);
    featureTracker_->outputFeatures.link(sync_->inputs[featuresKey_]);

    outputQueue_ = sync_->out.createOutputQueue(4, false);
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
    RCLCPP_INFO(getLogger(), "Stereo VO parameters updated");
}

void StereoVioNode::onFrame(const std::shared_ptr<dai::MessageGroup>& group) {
    if(group == nullptr || vio_ == nullptr) {
        return;
    }

    const auto disparityFrame = group->get<dai::ImgFrame>(disparityKey_);
    const auto features = group->get<dai::TrackedFeatures>(featuresKey_);
    if(disparityFrame == nullptr || features == nullptr) {
        return;
    }

    const DisparityMapSource depthSource(reinterpret_cast<const std::uint16_t*>(disparityFrame->getData().data()),
                                         disparityFrame->getWidth(),
                                         disparityFrame->getHeight(),
                                         disparityFrame->getWidth(),
                                         subpixelFractionalBits_);

    std::vector<Observation> observations;
    observations.reserve(features->trackedFeatures.size());
    for(const auto& feature : features->trackedFeatures) {
        Observation observation;
        observation.id = feature.id;
        observation.u = feature.position.x;
        observation.v = feature.position.y;
        observation.age = feature.age;
        observation.trackingError = feature.trackingError;
        observation.disparity = depthSource.disparityAt(observation.u, observation.v);
        if(observation.disparity <= 0.0F) {
            continue;  // no stereo support at this pixel
        }
        observations.push_back(observation);
    }

    // Empty observations are the one failure the status topic cannot explain on
    // its own: num_observations is counted *after* the disparity filter, so a
    // silent feature tracker and a dead stereo block look identical. Only pay
    // for this diagnostic when something is actually wrong.
    if(observations.empty()) {
        const auto& data = disparityFrame->getData();
        const auto* raw = reinterpret_cast<const std::uint16_t*>(data.data());
        const std::size_t pixels = data.size() / sizeof(std::uint16_t);
        std::size_t sampled = 0;
        std::size_t nonZero = 0;
        for(std::size_t i = 0; i < pixels; i += 16) {
            ++sampled;
            if(raw[i] != 0) {
                ++nonZero;
            }
        }
        const std::size_t expectedBytes = static_cast<std::size_t>(disparityFrame->getWidth()) * disparityFrame->getHeight() * sizeof(std::uint16_t);
        RCLCPP_WARN_THROTTLE(getLogger(),
                             *getROSNode()->get_clock(),
                             3000,
                             "VO produced no usable observations. tracker features=%zu | disparity %ux%u "
                             "type=%d bytes=%zu (expected %zu for RAW16) nonzero=%.1f%%. "
                             "features>0 with nonzero~0%% means the stereo block is not producing disparity; "
                             "features==0 means the feature tracker is not producing corners; "
                             "bytes!=expected means the frame is not RAW16 and the disparity decode is wrong.",
                             features->trackedFeatures.size(),
                             disparityFrame->getWidth(),
                             disparityFrame->getHeight(),
                             static_cast<int>(disparityFrame->getType()),
                             data.size(),
                             expectedBytes,
                             100.0 * static_cast<double>(nonZero) / static_cast<double>(std::max<std::size_t>(sampled, 1)));
    }

    const auto deviceStamp = disparityFrame->getTimestamp();
    const double seconds = std::chrono::duration<double>(deviceStamp.time_since_epoch()).count();

    const VioFrameResult result = vio_->processFrame(observations, seconds);

    // Use the host-synced timestamp for the ROS header so downstream nodes see
    // a clock they share.
    const auto hostStamp = disparityFrame->getTimestampDevice();
    (void)hostStamp;
    const rclcpp::Time stamp(std::chrono::duration_cast<std::chrono::nanoseconds>(deviceStamp.time_since_epoch()).count(), RCL_ROS_TIME);

    publish(result, stamp);
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
