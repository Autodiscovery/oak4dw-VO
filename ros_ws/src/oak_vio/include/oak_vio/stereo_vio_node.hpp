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
    /// Re-read every `vio.*` parameter into params_. Safe to call at runtime.
    VioParams readParams() const;

    /// Pull the rectified intrinsics and baseline off the device.
    RectifiedCamera readCameraModel(const std::shared_ptr<dai::Device>& device) const;

    /// Callback on the synced (disparity, features) message group.
    void onFrame(const std::shared_ptr<dai::MessageGroup>& group);

    void publish(const VioFrameResult& result, const rclcpp::Time& stamp);

    void handleReset(const std::shared_ptr<std_srvs::srv::Trigger::Request>& request,
                     const std::shared_ptr<std_srvs::srv::Trigger::Response>& response);

    // --- DepthAI graph -----------------------------------------------------
    std::shared_ptr<dai::node::Camera> leftCamera_;
    std::shared_ptr<dai::node::Camera> rightCamera_;
    std::shared_ptr<dai::node::StereoDepth> stereo_;
    std::shared_ptr<dai::node::FeatureTracker> featureTracker_;
    std::shared_ptr<dai::node::Sync> sync_;
    std::shared_ptr<dai::MessageQueue> outputQueue_;

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
