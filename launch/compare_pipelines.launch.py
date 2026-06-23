from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('lie_lane_detection')
    edge_config = os.path.join(pkg_share, 'config', 'lane_detector.yaml')
    line_config = os.path.join(pkg_share, 'config', 'lane_detector_line.yaml')

    return LaunchDescription([
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('camera_info_topic', default_value='/camera/camera_info'),
        Node(
            package='lie_lane_detection',
            executable='lane_detector_node',
            name='lane_detector_node',
            output='screen',
            parameters=[
                edge_config,
                {
                    'image_topic': LaunchConfiguration('image_topic'),
                    'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                },
            ],
        ),
        Node(
            package='lie_lane_detection',
            executable='line_lane_detector_node',
            name='line_lane_detector_node',
            output='screen',
            parameters=[
                line_config,
                {
                    'image_topic': LaunchConfiguration('image_topic'),
                    'camera_info_topic': LaunchConfiguration('camera_info_topic'),
                },
            ],
        ),
    ])
