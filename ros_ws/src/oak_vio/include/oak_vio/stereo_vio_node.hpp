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
// Feature detection and tracking run on the CPU via CpuFeatureTracker by
// default. The RVC4 hardware feature-tracker block is available as an option
// (vio.i_use_hw_tracker) now that Luxonis have identified what the node needs
// to produce anything: an explicit corner-detector configuration, since the
// automatic threshold returns empty feature messages. See
// docs/luxonis-bug-featuretracker-rvc4.md for the history and
// tools/probe_feature_tracker.py --luxonis-config to check it on your device
// before turning it on here.
#pragma once

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <mutex>
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
#include "oak_vio/imu_filter.hpp"
#include "oak_vio/imu_integrator.hpp"
#include "oak_vio/stereo_vio.hpp"

namespace oak_vio {

/// Unwraps a node's `initialConfig`, which is a std::shared_ptr<Config> in
/// DepthAI v3 and was a plain value in some v3 alphas.
///
/// The README flagged this as one of the two things to check on first compile.
/// Dispatching on the type costs nothing and removes the question, rather than
/// having every call site be a place the build can break on a different
/// DepthAI point release.
template <typename T>
T& initialConfigRef(T& config) {
    return config;
}
template <typename T>
T& initialConfigRef(std::shared_ptr<T>& config) {
    return *config;
}

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
    // None of the next few can be const: BaseNode::getROSNode() and
    // BaseNode::getLogger() are non-const accessors, so calling them from a
    // const member discards qualifiers. Logically all are read-only.
    VioParams readParams();
    CpuTrackerParams readTrackerParams();
    ImuParams readImuParams();
    ImuFilterParams readImuFilterParams();

    /// Build the IMU node and its queue, and read the IMU-to-rectified-camera
    /// rotation off the device. Returns false and logs when the device has no
    /// usable IMU or no IMU-to-camera extrinsic, in which case the VO runs
    /// exactly as it does with the IMU disabled.
    bool setupImu(const std::shared_ptr<dai::Device>& device);

    /// IMU queue callback. Converts dai::IMUPacket into ImuSample and feeds the
    /// integrator, and drives the filter when it is enabled.
    void onImuData(const std::shared_ptr<dai::ADatatype>& message);

    /// Convert the hardware tracker's output into the same Observation contract
    /// CpuFeatureTracker presents, so nothing downstream can tell which
    /// front-end produced it.
    [[nodiscard]] std::vector<Observation> observationsFromTrackedFeatures(const std::shared_ptr<dai::TrackedFeatures>& features) const;

    /// Pull the rectified intrinsics and baseline off the device.
    RectifiedCamera readCameraModel(const std::shared_ptr<dai::Device>& device);

    /// Callback on the synced (rectifiedLeft, disparity) message group.
    void onFrame(const std::shared_ptr<dai::MessageGroup>& group);

    /// Wrap a GRAY8/RAW8 ImgFrame as a cv::Mat without copying. Returns an empty
    /// Mat if the buffer is too small for the reported dimensions.
    [[nodiscard]] cv::Mat wrapGrayFrame(const std::shared_ptr<dai::ImgFrame>& frame);

    void publish(const VioFrameResult& result, const rclcpp::Time& stamp);

    /// Odometry and TF for one pose. Factored out of publish() because the
    /// Phase 4 filter publishes at IMU rate from a different thread, and both
    /// paths must apply the same optical-to-ROS conversion — including to the
    /// covariance, which is the part that is easy to forget in a second copy.
    void publishOdometry(const Pose& worldFromCamera, const Mat6& covariance, const Vec3& linearVelocity, const Vec3& angularVelocity, const rclcpp::Time& stamp);

    void handleReset(const std::shared_ptr<std_srvs::srv::Trigger::Request>& request,
                     const std::shared_ptr<std_srvs::srv::Trigger::Response>& response);

    // --- DepthAI graph -----------------------------------------------------
    std::shared_ptr<dai::node::Camera> leftCamera_;
    std::shared_ptr<dai::node::Camera> rightCamera_;
    std::shared_ptr<dai::node::StereoDepth> stereo_;
    std::shared_ptr<dai::node::Sync> sync_;
    std::shared_ptr<dai::MessageQueue> outputQueue_;

    /// Optional hardware front-end: rectified left -> ImageManip(GRAY8) ->
    /// FeatureTracker -> Sync. The ImageManip stays even though rectifiedLeft is
    /// already 8-bit grey, because feeding this node the wrong format crashes
    /// the firmware rather than erroring, and Camera::requestOutput has been
    /// observed substituting NV12 for a requested GRAY8.
    std::shared_ptr<dai::node::ImageManip> grayManip_;
    std::shared_ptr<dai::node::FeatureTracker> featureTracker_;

    /// Phase 5: LENS neural depth, co-running on the DSP. Created only when
    /// vio.i_enable_neural_depth is set. Fed the already-rectified pair with its
    /// own rectification disabled, so it does not redo work StereoDepth has
    /// already done and its depth lands in the same frame as our disparity.
    std::shared_ptr<dai::node::NeuralDepth> neuralDepth_;

    /// Phase 4: IMU. Deliberately NOT routed through Sync -- see setupImu.
    std::shared_ptr<dai::node::IMU> imu_;
    std::shared_ptr<dai::MessageQueue> imuQueue_;

    std::string syncQueueName_;
    std::string disparityKey_;
    std::string rectifiedKey_;
    std::string featuresKey_;
    std::string neuralDepthKey_;

    // --- Estimator ---------------------------------------------------------
    std::unique_ptr<CpuFeatureTracker> tracker_;
    std::unique_ptr<StereoVio> vio_;
    VioParams params_;

    // --- IMU ---------------------------------------------------------------
    /// Guards imuIntegrator_ and imuFilter_: the IMU queue callback and the
    /// frame callback run on different DepthAI threads.
    std::mutex imuMutex_;
    std::unique_ptr<ImuIntegrator> imuIntegrator_;
    std::unique_ptr<ImuFilter> imuFilter_;
    ImuParams imuParams_;
    ImuFilterParams imuFilterParams_;
    bool imuAvailable_{false};
    /// Phase 4, step 5: republish odometry at IMU rate from the filter rather
    /// than at frame rate from the VO. Only meaningful with the filter enabled,
    /// and at the 10 Hz this rig is FSYNC-limited to it is the difference
    /// between usable and not for a controller.
    bool publishAtImuRate_{false};
    /// Timestamp of the previous frame, which with the current one defines the
    /// gyro integration window.
    double lastFrameStamp_{0.0};
    bool haveLastFrameStamp_{false};
    std::atomic<std::uint64_t> imuSamples_{0};
    std::atomic<std::uint64_t> gyroPriorsUsed_{0};
    std::atomic<std::uint64_t> gyroPriorsRejected_{0};
    const char* lastGyroRejection_{""};

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

    // --- Front-end selection -----------------------------------------------
    /// Use the RVC4 hardware feature tracker instead of the CPU one.
    ///
    /// Off by default, and staying that way until it is confirmed on this
    /// device. The history is in docs/luxonis-bug-featuretracker-rvc4.md: the
    /// node returned zero features for every configuration tried and took the
    /// firmware down with it. Luxonis have since identified that the automatic
    /// corner-detector threshold yields empty messages and the detector must be
    /// configured explicitly, which the parameters below do.
    bool useHwTracker_{false};
    float hwTrackerThreshold_{20000.0F};
    int hwTrackerMaxFeatures_{256};
    int hwTrackerHardwareResources_{-1};  ///< negative leaves the node's default
    bool hwMotionEstimator_{true};

    /// Phase 5. Neural depth co-runs when this is on; whether the VO *consumes*
    /// it is a separate choice (depthSourceNeural_), because the two questions
    /// are separate: co-running is about engine contention, consuming is about
    /// whether smoothed neural depth is good enough for pose.
    bool enableNeuralDepth_{false};
    std::string neuralDepthModel_{"NEURAL_DEPTH_SMALL"};
    bool depthSourceNeural_{false};
    /// fx * baseline, in metre-pixels: converts metric depth back into the
    /// equivalent disparity DepthMapSource needs. Cached from the camera model.
    double fxBaselineM_{0.0};

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

    /// Startup watchdog.
    ///
    /// Adding the hardware tracker as a third Sync input creates a failure mode
    /// worth naming: if the tracker emits nothing — which is exactly what it did
    /// on this device before the explicit corner-detector configuration — Sync
    /// never completes a group, onFrame never runs, and the periodic health line
    /// that would have said so never runs either. The VO simply goes quiet. This
    /// timer is wall-clock rather than frame-driven for that reason, and it
    /// names the parameter to turn off.
    rclcpp::TimerBase::SharedPtr startupWatchdog_;
    std::atomic<std::uint64_t> syncedGroups_{0};
    std::atomic<std::uint64_t> featureMessages_{0};
    std::shared_ptr<dai::MessageQueue> featureRateQueue_;

    /// Rotation from the camera optical frame (x right, y down, z forward) to
    /// the ROS body convention (x forward, y left, z up). Applied to the pose,
    /// the twist and the covariance so RViz and robot_localization see
    /// something sane.
    static Mat3 opticalToRos();
};

}  // namespace oak_vio
