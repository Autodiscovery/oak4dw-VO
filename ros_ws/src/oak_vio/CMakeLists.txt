cmake_minimum_required(VERSION 3.16)
project(oak_vio)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()

if(NOT CMAKE_BUILD_TYPE)
  # This runs on an ARM CPU alongside LENS. Never ship it unoptimised.
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)

# ---------------------------------------------------------------------------
# Core estimator: no ROS, no DepthAI. Kept separate so the maths can be built
# and tested on a workstation, and so the ROS layer stays thin.
# ---------------------------------------------------------------------------
add_library(oak_vio_core SHARED
  src/motion_estimator.cpp
  src/ransac.cpp
  src/bucketing.cpp
  src/keyframe_manager.cpp
  src/motion_model.cpp
  src/pose_integrator.cpp
  src/stereo_vio.cpp
)
target_include_directories(oak_vio_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(oak_vio_core PUBLIC Eigen3::Eigen)

# ---------------------------------------------------------------------------
# depthai_ros_driver pipeline plugin + the publishing node.
#
# On Humble/Jazzy the DepthAI v3 packages carry a `_v3` suffix; Kilted dropped
# it. Resolve either so this builds on both without edits.
# ---------------------------------------------------------------------------
set(DEPTHAI_PKGS "")
foreach(base depthai depthai_bridge depthai_ros_driver depthai_ros_msgs)
  if(NOT DEFINED ${base}_RESOLVED)
    find_package(${base}_v3 QUIET)
    if(${base}_v3_FOUND)
      list(APPEND DEPTHAI_PKGS ${base}_v3)
    else()
      find_package(${base} REQUIRED)
      list(APPEND DEPTHAI_PKGS ${base})
    endif()
  endif()
endforeach()
message(STATUS "oak_vio: using DepthAI packages: ${DEPTHAI_PKGS}")

find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(oak_vio_msgs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

add_library(oak_vio_pipeline_plugin SHARED
  src/stereo_vio_pipeline.cpp
  src/stereo_vio_node.cpp
)
target_include_directories(oak_vio_pipeline_plugin PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(oak_vio_pipeline_plugin PUBLIC oak_vio_core)
ament_target_dependencies(oak_vio_pipeline_plugin
  ${DEPTHAI_PKGS}
  geometry_msgs
  nav_msgs
  oak_vio_msgs
  pluginlib
  rclcpp
  sensor_msgs
  std_srvs
  tf2
  tf2_geometry_msgs
  tf2_ros
)
pluginlib_export_plugin_description_file(depthai_ros_driver plugins.xml)

# ---------------------------------------------------------------------------
# Tests. These exercise the core only -- no hardware, no ROS graph -- so they
# run in CI and on a laptop.
# ---------------------------------------------------------------------------
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_oak_vio
    test/test_motion_estimator.cpp
    test/test_ransac.cpp
    test/test_bucketing.cpp
    test/test_pose_integrator.cpp
    test/test_trajectory.cpp
  )
  if(TARGET test_oak_vio)
    target_include_directories(test_oak_vio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/test)
    target_link_libraries(test_oak_vio oak_vio_core)
  endif()
endif()

install(DIRECTORY include/ DESTINATION include)
install(TARGETS oak_vio_core oak_vio_pipeline_plugin
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
install(FILES plugins.xml DESTINATION share/${PROJECT_NAME})

ament_export_include_directories(include)
ament_export_libraries(oak_vio_core oak_vio_pipeline_plugin)
ament_export_dependencies(Eigen3 ${DEPTHAI_PKGS} rclcpp pluginlib)
ament_package()
