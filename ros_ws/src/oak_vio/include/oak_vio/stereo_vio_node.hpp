// depthai_ros_driver dai_node that owns the VO half of the pipeline and
// publishes its results.
//
// It builds its own Camera / StereoDepth / FeatureTracker / Sync subgraph
// inside the driver's pipeline rather than reusing the driver's sensor node
// wrappers. That keeps our dependency on driver internals down to two stable
// base classes (BaseNode, BasePipeline) instead of the whole dai_nodes tree,
// at the cost of not inheriting the driver's per-sensor parameter surface for
// these two cameras. If you later want the driver's RGB or NN topics as well,
// add them in StereoVioPipeline::createPipeline alongside this node — they
// share the pipeline and therefore the device.
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
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/transform_broadcaster.h>

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
    /// Re-read every `vio.*` parameter into params_. Safe to call at runtime.
    VioParams readParams();

    /// Pull the rectified intrinsics and baseline off the device.
    RectifiedCamera readCameraModel(const std::shared_ptr<dai::Device>& device);

    /// Callback on the synced (disparity, features) message group.
    void onFrame(const std::shared_ptr<dai::MessageGroup>& group);

    void publish(const VioFrameResult& result, const rclcpp::Time& stamp);

    void handleReset(const std::shared_ptr<std_srvs::srv::Trigger::Request>& request,
                     const std::shared_ptr<std_srvs::srv::Trigger::Response>& response);

    // --- DepthAI graph -----------------------------------------------------
    std::shared_ptr<dai::node::Camera> leftCamera_;
    std::shared_ptr<dai::node::Camera> rightCamera_;
    std::shared_ptr<dai::node::StereoDepth> stereo_;
    /// Converts rectifiedLeft from RAW8 to GRAY8. The feature tracker silently
    /// finds nothing in RAW8, so this is load-bearing, not cosmetic.
    std::shared_ptr<dai::node::ImageManip> imageManip_;
    std::shared_ptr<dai::node::FeatureTracker> featureTracker_;
    std::shared_ptr<dai::node::Sync> sync_;
    std::shared_ptr<dai::MessageQueue> outputQueue_;

    /// Bring-up instrumentation. Counts messages leaving each stage so the
    /// pipeline's rate bottleneck can be localised by measurement instead of
    /// hypothesis. Remove once bring-up is finished.
    struct StageRate {
        std::shared_ptr<dai::MessageQueue> queue;
        std::atomic<std::uint64_t> count{0};
        std::uint64_t previous{0};
    };
    StageRate cameraRate_;
    StageRate manipRate_;
    StageRate disparityRate_;
    StageRate featureRate_;
    std::chrono::steady_clock::time_point lastRateLog_{};
    /// Synced frames since the last rate report.
    std::uint64_t frameIndexDelta_{0};

    /// Last frame seen entering the tracker, and the last feature count leaving
    /// it. Recorded in the counting callbacks so no extra queue is needed.
    std::atomic<int> lastManipType_{-1};
    std::atomic<unsigned> lastManipWidth_{0};
    std::atomic<unsigned> lastManipHeight_{0};
    std::atomic<std::size_t> lastManipBytes_{0};
    /// Pixel statistics of the frame entering the tracker. A near-zero std
    /// means it is being fed a blank image and the fault is upstream of it.
    std::atomic<double> lastManipMean_{0.0};
    std::atomic<double> lastManipStd_{0.0};
    std::atomic<std::size_t> lastFeatureCount_{0};
    /// Auto-exposure state. An exposure near the frame period means AE is what
    /// is capping the frame rate, and a dim frame is why there are no corners.
    std::atomic<long> lastExposureUs_{0};
    std::atomic<int> lastSensitivityIso_{0};

    std::string syncQueueName_;
    std::string disparityKey_;
    std::string featuresKey_;

    // --- Estimator ---------------------------------------------------------
    std::unique_ptr<StereoVio> vio_;
    VioParams params_;

    // --- ROS ---------------------------------------------------------------
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometryPublisher_;
    rclcpp::Publisher<oak_vio_msgs::msg::VioStatus>::SharedPtr statusPublisher_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tfBroadcaster_;
    rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr resetService_;

    // --- Configuration -----------------------------------------------------
    int width_{1280};
    int height_{800};
    double fps_{30.0};
    int subpixelFractionalBits_{5};
    std::string odomFrame_{"odom"};
    std::string baseFrame_{"oak_vio_frame"};
    bool publishTf_{true};
    bool rosConvention_{true};
    bool enableIr_{false};
    float alphaScaling_{-1.0F};

    /// Rotation from the camera optical frame (x right, y down, z forward) to
    /// the ROS body convention (x forward, y left, z up). Applied to the pose,
    /// the twist and the covariance so RViz and robot_localization see
    /// something sane.
    static Mat3 opticalToRos();
};

}  // namespace oak_vio
