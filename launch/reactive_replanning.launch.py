# reactive_replanning.launch.py
# Requires ur12e_hande_bringup to already be running with launch_moveit:=true.
# This launches the scene setup and the reactive replanning demo node.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('pick_x',  default_value='0.45'),
        DeclareLaunchArgument('pick_y',  default_value='0.0'),
        DeclareLaunchArgument('pick_z',  default_value='0.35'),
        DeclareLaunchArgument('place_x', default_value='0.0'),
        DeclareLaunchArgument('place_y', default_value='0.55'),
        DeclareLaunchArgument('place_z', default_value='0.35'),

        Node(
            package='reactive_replanning_ur12e',
            executable='scene_setup.py',
            name='scene_setup',
            output='screen',
        ),
        Node(
            package='reactive_replanning_ur12e',
            executable='reactive_replanning.py',
            name='reactive_replanner',
            output='screen',
            parameters=[{
                'pick_x':  LaunchConfiguration('pick_x'),
                'pick_y':  LaunchConfiguration('pick_y'),
                'pick_z':  LaunchConfiguration('pick_z'),
                'place_x': LaunchConfiguration('place_x'),
                'place_y': LaunchConfiguration('place_y'),
                'place_z': LaunchConfiguration('place_z'),
                'ik_attempts': 30,
                'vel_scale':   0.3,
                'acc_scale':   0.3,
            }],
        ),
    ])
