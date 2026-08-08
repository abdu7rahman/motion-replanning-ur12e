# reactive_replanning.launch.py
# Requires ur12e_hande_bringup to already be running with launch_moveit:=true.
# This launches the scene setup and the reactive replanning demo node.
#
# Tuning comes from config/tuning.yaml. Override anything in it on the command
# line without editing the file:
#
#   ros2 launch reactive_replanning_ur12e reactive_replanning.launch.py \
#     params_file:=/path/to/my_cell.yaml
#
# The pick and place poses are not coordinates -- they are offsets from the EEF
# pose that FK reports at the home joint configuration, so they follow the robot
# rather than assuming where it is bolted down.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    default_params = PathJoinSubstitution([
        FindPackageShare('reactive_replanning_ur12e'), 'config', 'tuning.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file', default_value=default_params,
            description='Every tuning parameter the node reads. See config/tuning.yaml.'),
        DeclareLaunchArgument(
            'pick_z_offset', default_value='-0.25',
            description='Metres down from the home EEF pose to the pick.'),
        DeclareLaunchArgument(
            'place_y_offset', default_value='0.75',
            description='Metres across from the home EEF pose to the place.'),

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
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'pick_z_offset':  LaunchConfiguration('pick_z_offset'),
                    'place_y_offset': LaunchConfiguration('place_y_offset'),
                },
            ],
        ),
    ])
