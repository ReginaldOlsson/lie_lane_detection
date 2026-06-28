from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg_share = get_package_share_directory('lie_lane_detection')
    tracked_config = os.path.join(pkg_share, 'config', 'tracked_lane_detector.yaml')
    default_video = '/home/mosal/Downloads/truck_videos/truck_highway_7m40.webm'

    return LaunchDescription([
        DeclareLaunchArgument(
            'video_path',
            default_value=default_video,
            description='Path to truck frontal video (webm/mp4)'),
        DeclareLaunchArgument(
            'publish_rate_hz',
            default_value='10.0',
            description='Video publisher rate'),
        DeclareLaunchArgument(
            'target_width',
            default_value='640',
            description='Resize width (0 = native resolution)'),
        DeclareLaunchArgument(
            'track_main_interval',
            default_value='6',
            description='Run full Lie-Hough every N processed frames'),
        DeclareLaunchArgument(
            'publish_raw',
            default_value='true',
            description='Publish full detect (no tracking) on /lanes/raw/*'),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw'),
        # Node(
        #     package='lie_lane_detection',
        #     executable='video_image_publisher_node',
        #     name='video_image_publisher',
        #     output='screen',
        #     parameters=[{
        #         'video_path': LaunchConfiguration('video_path'),
        #         'image_topic': LaunchConfiguration('image_topic'),
        #         'publish_rate_hz': LaunchConfiguration('publish_rate_hz'),
        #         'target_width': LaunchConfiguration('target_width'),
        #         'loop': True,
        #     }],
        # ),
        Node(
            package='lie_lane_detection',
            executable='tracked_lane_detector_node',
            name='tracked_lane_detector_node',
            output='screen',
            parameters=[
                tracked_config,
                {
                    'image_topic': LaunchConfiguration('image_topic'),
                    'track_main_interval': LaunchConfiguration('track_main_interval'),
                    'publish_raw': LaunchConfiguration('publish_raw'),
                },
            ],
        ),
    ])
