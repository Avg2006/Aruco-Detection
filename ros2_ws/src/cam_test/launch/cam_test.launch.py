import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Get the directory where the package is installed
    pkg_share_dir = get_package_share_directory('cam_test')
    
    # Define the path to your camera info YAML file
    yaml_config_path = os.path.join(pkg_share_dir, 'config', 'lapcare_ost.yaml')

    return LaunchDescription([
        # Launch the camera testing node
        Node(
            package='cam_test',
            executable='camera_tester',
            name='camera_tester_node',
            output='screen'
        ),
        
        # Launch the camera info publisher node
        Node(
            package='cam_test',
            executable='camera_info_pub',
            name='camera_info_publisher',
            parameters=[{'yaml_file_path': yaml_config_path}],
            output='screen'
        )
    ])
