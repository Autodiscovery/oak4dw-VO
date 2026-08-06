// depthai_ros_driver pipeline plugin that installs the stereo VO subgraph.
//
// Selected at runtime with:
//
//     pipeline_gen:
//       i_pipeline_type: oak_vio::StereoVioPipeline
//
// Everything returned here shares one dai::Pipeline and therefore one device.
// That is the whole reason VO lives inside the driver rather than in its own
// process: a DepthAI device can only be opened once, so RGB, LENS depth and VO
// have to be nodes in a single graph.
#include <memory>
#include <string>
#include <vector>

#include <depthai/depthai.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>

#include "oak_vio/driver_compat.hpp"
#include "oak_vio/stereo_vio_node.hpp"

namespace oak_vio {

class StereoVioPipeline : public DriverBasePipeline {
   public:
    std::vector<std::unique_ptr<DriverBaseNode>> createPipeline(
        std::shared_ptr<rclcpp::Node> node,
        std::shared_ptr<dai::Device> device,
        std::shared_ptr<dai::Pipeline> pipeline,
        std::shared_ptr<depthai_ros_driver::param_handlers::PipelineGenParamHandler> ph,
        const std::string& deviceName,
        bool rsCompat,
        const std::string& nnType) override {
        (void)device;
        (void)ph;
        (void)nnType;

        std::vector<std::unique_ptr<DriverBaseNode>> nodes;
        nodes.emplace_back(std::make_unique<StereoVioNode>("vio", node, pipeline, deviceName, rsCompat));

        RCLCPP_INFO(node->get_logger(), "oak_vio: stereo VO pipeline created");

        // To add LENS neural depth or an RGB stream later, append them here.
        // They will share this pipeline and this device, which is exactly what
        // we want: the VO runs on the feature-tracker and stereo blocks while
        // NeuralDepth runs on the DSP, so the two do not contend for the same
        // engine.
        return nodes;
    }
};

}  // namespace oak_vio

// Deliberately spelled out rather than using oak_vio::DriverBasePipeline: this
// string must match `base_class_type` in plugins.xml exactly, and pluginlib
// resolves that by name at runtime. If the diagnostic shows the driver uses a
// different namespace, both this line and plugins.xml need the same edit.
PLUGINLIB_EXPORT_CLASS(oak_vio::StereoVioPipeline, depthai_ros_driver::pipeline_gen::BasePipeline)
