"""Launch the depthai_ros_driver with the stereo VO pipeline plugin selected.

We provide our own launch file rather than reusing the driver's so the app does
not depend on the driver's launch-file naming, which differs between the `_v3`
(Humble/Jazzy) and unsuffixed (Kilted+) package sets.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory, PackageNotFoundError


def _driver_package() -> str:
    """Resolve the driver package name across ROS distros."""
    for candidate in ("depthai_ros_driver_v3", "depthai_ros_driver"):
        try:
            get_package_share_directory(candidate)
            return candidate
        except PackageNotFoundError:
            continue
    raise RuntimeError(
        "Neither depthai_ros_driver_v3 nor depthai_ros_driver is installed. "
        "On Humble/Jazzy install ros-$ROS_DISTRO-depthai-ros-v3."
    )


def generate_launch_description() -> LaunchDescription:
    default_params = "/app/params/vio.yaml"

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Path to the driver parameter file selecting the VO pipeline.",
    )
    name_arg = DeclareLaunchArgument(
        "name", default_value="oak", description="Node name and topic namespace."
    )

    driver = Node(
        package=_driver_package(),
        executable="driver_node",
        name=LaunchConfiguration("name"),
        parameters=[LaunchConfiguration("params_file")],
        output="screen",
    )

    return LaunchDescription([params_file_arg, name_arg, driver])
