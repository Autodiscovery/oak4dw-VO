// Locates the depthai_ros_driver headers across packaging variants.
//
// The driver package is named `depthai_ros_driver` on Kilted and newer, and
// `depthai_ros_driver_v3` on Humble and Jazzy. Depending on how the renamed
// package was built, its headers may land under either:
//
//     include/depthai_ros_driver/...      (classic flat layout)
//     include/depthai_ros_driver_v3/...   (renamed include dir)
//
// Each puts a different string in the #include line, so this dispatches on
// __has_include rather than making the build depend on which distro packaged
// it. Include this instead of the driver headers directly.
//
// Confirmed on Luxonis OS with ros-jazzy-depthai-ros-v3: the headers are at
//     /opt/ros/jazzy/include/depthai_ros_driver_v3/dai_nodes/base_node.hpp
// while the C++ namespace stays `depthai_ros_driver` -- which is why the
// namespace alias below is not needed, and why plugins.xml and
// PLUGINLIB_EXPORT_CLASS keep the unsuffixed name.
#pragma once

#if __has_include(<depthai_ros_driver/dai_nodes/base_node.hpp>)
#include <depthai_ros_driver/dai_nodes/base_node.hpp>
#include <depthai_ros_driver/pipeline/base_pipeline.hpp>
#define OAK_VIO_DRIVER_INCLUDE_PREFIX "depthai_ros_driver"

#elif __has_include(<depthai_ros_driver_v3/dai_nodes/base_node.hpp>)
#include <depthai_ros_driver_v3/dai_nodes/base_node.hpp>
#include <depthai_ros_driver_v3/pipeline/base_pipeline.hpp>
#define OAK_VIO_DRIVER_INCLUDE_PREFIX "depthai_ros_driver_v3"

#else
#error                                                                                                    \
    "Cannot find the depthai_ros_driver headers (dai_nodes/base_node.hpp). "                              \
    "The package was found by CMake, so this is a layout mismatch rather than a missing dependency. "     \
    "Run the diagnostic build step in oakapp.toml (it prints the installed header tree) and add the "     \
    "correct prefix to include/oak_vio/driver_compat.hpp."
#endif

namespace oak_vio {

// The C++ namespace is expected to stay `depthai_ros_driver` even where the
// package was renamed, since renaming the namespace would break every existing
// plugin. If the diagnostic shows otherwise, add an alias here rather than
// editing every use site:
//
//     namespace depthai_ros_driver = depthai_ros_driver_v3;

/// Base class for a DepthAI node paired with its ROS publishers.
using DriverBaseNode = depthai_ros_driver::dai_nodes::BaseNode;

/// Base class for a pipeline plugin selected via `pipeline_gen.i_pipeline_type`.
using DriverBasePipeline = depthai_ros_driver::pipeline_gen::BasePipeline;

}  // namespace oak_vio
