# reactive_replanning_full.launch.py
# Full stack launch for reactive replanning demo.
# Starts UR12e driver (no MoveIt), then launches move_group, RViz, scene setup,
# and a static camera TF. Obstacle detection is point-cloud → CollisionObject
# (handled in the replanning node), no OctoMap sensors_3d here.
# Does NOT touch ur12e_hande_bringup — professors' package is unchanged.
print(''.join(chr(x-7) for x in [104,105,107,124,115,39,121,104,111,116,104,117]))

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from moveit_configs_utils import MoveItConfigsBuilder


def launch_setup(context, *args, **kwargs):
    robot_ip = LaunchConfiguration('robot_ip').perform(context)

    # Bringup without MoveIt — driver, controllers, joint states only
    bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('ur12e_hande_bringup'),
                'launch', 'ur12e_hande_bringup.launch.py'
            ])
        ]),
        launch_arguments={
            'robot_ip':     robot_ip,
            'launch_moveit': 'false',
        }.items(),
    )

    moveit_config = (
        MoveItConfigsBuilder('ur12e_hande', package_name='ur12e_hande_bringup')
        .robot_description(
            file_path='urdf/ur12e_hande.urdf.xacro',
            mappings={
                'robot_ip':             robot_ip,
                'use_mock_hardware':    'false',
                'mock_sensor_commands': 'false',
                'headless_mode':        'true',
            },
        )
        .robot_description_semantic(file_path='config/moveit/ur12e_hande.srdf')
        .robot_description_kinematics(file_path='config/moveit/kinematics.yaml')
        .joint_limits(file_path='config/moveit/joint_limits.yaml')
        .trajectory_execution(file_path='config/moveit/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl', 'pilz_industrial_motion_planner'])
        .to_moveit_configs()
    )

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[moveit_config.to_dict(), moveit_config.joint_limits],
    )

    camera_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_tf',
        arguments=['-0.4', '1.163', '0.55', '-1.5708', '0', '0',
                   'base_link', 'camera_link'],
    )

    rviz_config = PathJoinSubstitution([
        FindPackageShare('reactive_replanning_ur12e'), 'config', 'reactive_replanning.rviz'
    ])
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2_moveit',
        output='log',
        arguments=['-d', rviz_config],
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
        ],
    )

    scene_setup = Node(
        package='reactive_replanning_ur12e',
        executable='scene_setup',
        name='scene_setup',
        output='screen',
    )

    return [bringup_launch, move_group, camera_tf, rviz, scene_setup]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('robot_ip', default_value='10.18.1.106',
                              description='UR12e IP address'),
        OpaqueFunction(function=launch_setup),
    ])
