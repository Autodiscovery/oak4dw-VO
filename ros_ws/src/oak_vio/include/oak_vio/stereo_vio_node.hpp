// depthai_ros_driver dai_node that owns the VO half of the pipeline and
// publishes its results.
//
// It builds its own Camera / StereoDepth / Sync subgraph inside the driver's
// pipeline rather than reusing the driver's sensor node wrappers. That keeps our
// dependency on driver internals down to two stable base classes (BaseNode,
// BasePipeline) instead of the whole dai_nodes tree. If you later want the
// driver's RGB or NN topics as well, add them in
// StereoVioPipeline::createPipeline alongside this node -- they share the
// pipeline and therefore the device.
//
// Feature detection and tracking run on the CPU via CpuFeatureTracker, NOT on
// the RVC4 hardware feature-tracker block. The hardware node is non-functional
// on this firmware; see docs/luxonis-bug-featuretracker-rvc4.md.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <depthai/depthai.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <oak_vio_msgs/msg/vio_status.hpp>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "oak_vio/cpu_feature_tracker.hpp"
#include "oak_vio/driver_compat.hpp"
#include "oak_vio/stereo_vio.hpp"

namespace oak_vio {

class StereoVioNode : public DriverBaseNode {
   public:
    StereoVioNode(const std::string& daiNodeName,
                  std::shared_ptr<rclcpp::Node> node,
                  std::shared_ptr<dai::Pipeline> pipeline,
                  const std::string& deviceName,
                  bool rsCompatibility);
    ~StereoVioNode() override;

    void setNames() override;
    void setInOut(std::shared_ptr<dai::Pipeline> pipeline) override;
    void setupQueues(std::shared_ptr<dai::Device> device) override;
    void closeQueues() override;
    void updateParams(const std::vector<rclcpp::Parameter>& params) override;

   private:
    void declareParams();
    // Neither of the next two can be const: BaseNode::getROSNode() and
    // BaseNode::getLogger() are non-const accessors, so calling them from a
    // const member discards qualifiers. Logically both are read-only.
    VioParams readParams();
    CpuTrackerParams readTrackerParams();

    /// Pull the rectified intrinsics and baseline off the device.
    RectifiedCamera readCameraModel(const std::shared_ptr<dai::Device>& device);

    /// Callback on the synced (rectifiedLeft, disparity) message group.
    void onFrame(const std::shared_ptr<dai::MessageGroup>& group);

    /// Wrap a GRAY8/RAW8 ImgFrame as a cv::Mat without copying. Returns an empty
    /// Mat if the buffer is too small for the reported dimensions.
    [[nodiscard]] cv::Mat wrapGrayFrame(const std::shared_ptr<dai::ImgFrame>& frame);

    void publish(const VioFrameResult& result, const rclcpp::Time& stamp);

    void handleReset(const std::shared_ptr<std_srvs::srv::Trigger::Request>& request,
                     const std::shared_ptr<std_srvs::srv::Trigger::Response>& response);

    // --- DepthAI graph -----------------------------------------------------
    std::shared_ptr<dai::node::Camera> leftCamera_;
    std::shared_ptr<dai::node::Camera> rightCamera_;
    std::shared_ptr<dai::node::StereoDepth> stereo_;
    std::shared_ptr<dai::node::Sync> sync_;
    std::shared_ptr<dai::MessageQueue> outputQueue_;

    std::string syncQueueName_;
    std::string disparityKey_;
    std::string rectifiedKey_;

    // --- Estimator ---------------------------------------------------------
    std::unique_ptr<CpuFeatureTracker> tracker_;
    std::unique_ptr<StereoVio> vio_;
    VioParams params_;

    // --- ROS ---------------------------------------------------------------
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometryPublisher_;
    rclcpp::Publisher<oak_vio_msgs::msg::VioStatus>::SharedPtr statusPublisher_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr resetService_;

    // --- Configuration -----------------------------------------------------
    int width_{640};
    int height_{400};
    /// Sensor mode, set explicitly so GRAY8 is honoured and the frame rate takes
    /// effect. Must be a resolution the OV9282 supports.
    int sensorWidth_{1280};
    int sensorHeight_{800};
    double fps_{30.0};
    int subpixelFractionalBits_{5};
    std::string odomFrame_{"odom"};
    std::string baseFrame_{"oak_vio_frame"};
    bool publishTf_{true};
    bool rosConvention_{true};
    float alphaScaling_{-1.0F};

    /// Is a cable plugged into this camera's M8 "IN" port?
    ///
    /// This is a physical fact about the rig, not a software preference, and it
    /// decides who owns the frame rate. Cable in IN means FSYNC slave: the master
    /// dictates the rate and we must not try to set it. Nothing in IN means this
    /// camera generates FSYNC, owns its timing, and vio.i_fps applies.
    ///
    /// There is no API to query this, so it has to be told to us.
    bool fsyncConnected_{true};

    // --- Timing / health ---------------------------------------------------
    /// Wall-clock cost of the CPU tracker, which is the one part of this
    /// pipeline that is no longer free. Reported so the LENS budget stays honest.
    double trackerMsEma_{0.0};
    std::chrono::steady_clock::time_point lastRateLog_{};
    std::uint64_t framesSinceLog_{0};
    std::atomic<std::uint64_t> cameraFrames_{0};
    std::shared_ptr<dai::MessageQueue> cameraRateQueue_;
    std::atomic<long> lastExposureUs_{0};
    std::atomic<int> lastSensitivityIso_{0};

    /// Rotation from the camera optical frame (x right, y down, z forward) to
    /// the ROS body convention (x forward, y left, z up). Applied to the pose,
    /// the twist and the covariance so RViz and robot_localization see
    /// something sane.
    static Mat3 opticalToRos();
};

}  // namespace oak_vio
