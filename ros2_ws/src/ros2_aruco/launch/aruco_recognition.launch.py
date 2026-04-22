import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    aruco_params = os.path.join(
        get_package_share_directory('ros2_aruco'),
        'config',
        'aruco_parameters.yaml'
        )
    pkg_share = get_package_share_directory('ros2_aruco')
    camera_config_path = os.path.join(pkg_share, 'config/ost.yaml')

    return LaunchDescription([
        Node(
            package='ros2_aruco',
            executable='aruco_node',
            parameters=[aruco_params],
        ),
        # Node(
        #     package='ros2_aruco',
        #     executable='aruco_sub',
        #     name='aruco_sub',
        #     output='screen',
        # ),
        # Node(
        #     package='ros2_aruco',          
        #     executable='camera_node',
        #     name='pub_cam_node',
        #     output='log',
        #     emulate_tty=True,
        # ),

        # # Node 2: Camera Info Publisher
        # Node(
        #     package='ros2_aruco',          
        #     executable='cam_info',
        #     name='camera_info_publisher',
        #     output='screen',
        #     parameters=[{
        #         'yaml_file_path': camera_config_path
        #     }]
        # ),

    ])
