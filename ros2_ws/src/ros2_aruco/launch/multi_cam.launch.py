"""
multi_cam_launch.py

Launches N usb_cam nodes + N aruco_node instances, each isolated under its
own namespace, plus a static_transform_publisher per camera that encodes its
extrinsic pose in the shared "world" frame.

USAGE
-----
    ros2 launch <your_pkg> multi_cam_launch.py

HOW TO GET THE STATIC TRANSFORM NUMBERS
----------------------------------------
This is where your "A + B" calibration result goes. Whatever method you use
(known-anchor solvePnP, or ChArUco/ArUco pairwise + optimization), you end up
with, for each camera, a 4x4 matrix T_world_cam. Decompose that into:

    x y z  (meters, translation)
    qx qy qz qw (quaternion, rotation)

and drop the numbers into the CAMERAS list below. If you only have a
rotation matrix / rvec, convert with:

    import numpy as np, cv2, tf_transformations
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4); T[:3,:3] = R; T[:3,3] = tvec.flatten()
    T_world_cam = np.linalg.inv(T)   # if you computed cam->world, invert as needed
    quat = tf_transformations.quaternion_from_matrix(T_world_cam)
    xyz = T_world_cam[:3, 3]

NOTE ON static_transform_publisher ARGUMENT ORDER
--------------------------------------------------
In current ROS2 distros (Humble+) the CLI/launch signature is:
    x y z qx qy qz qw parent_frame child_frame
(older distros used roll/pitch/yaw instead of quaternion -- adjust if needed).
"""

from launch import LaunchDescription
from launch_ros.actions import Node


# ---- Fill this in with your actual measured/calibrated extrinsics ----
CAMERAS = [
    {
        "name": "cam1",
        "video_device": "/dev/video2",
        "image_width": 1280,
        "image_height": 720,
        "pixel_format": "yuyv",
        "camera_info_url": "file:///home/zak/.ros/camera_info/default_cam.yaml",
        "frame_id": "cam1_optical_frame",
        # world -> cam1_optical_frame  (x y z qx qy qz qw)
        "world_xyz": [-0.17416, 0.92742, 2.68688],
        "world_quat": [0.83769, -0.52325, -0.12678, 0.09171],

    },
    {
        "name": "cam2",
        "video_device": "/dev/video4",
        "image_width": 1280,
        "image_height": 720,
        "pixel_format": "yuyv",
        "camera_info_url": "file:///home/zak/.ros/camera_info/default_cam2.yaml",
        "frame_id": "cam2_optical_frame",
        # world -> cam2_optical_frame
        "world_xyz": [0.02075, 0.86226, 2.84136],
        "world_quat": [0.79378, -0.55809, 0.21470, 0.11112],  
    },
    # add more cameras here...
]

MARKER_SIZE = 0.08664
ARUCO_DICT = "DICT_5X5_250"


def generate_launch_description():
    actions = []

    for cam in CAMERAS:
        ns = cam["name"]

        # --- usb_cam driver ---
        actions.append(Node(
            package="usb_cam",
            executable="usb_cam_node_exe",
            name=f"{ns}_usb_cam",
            namespace=ns,
            parameters=[{
                "video_device": cam["video_device"],
                "camera_name": ns,
                "camera_info_url": cam["camera_info_url"],
                "frame_id": cam["frame_id"],
            }],
            remappings=[
                ("image_raw", f"/{ns}/image_raw"),
                ("camera_info", f"/{ns}/camera_info"),
            ],
        ))

        # --- aruco detector for this camera ---
        actions.append(Node(
            package="ros2_aruco",       # <-- change to your actual package name
            executable="aruco_node",
            name=f"{ns}_aruco_node",
            namespace=ns,
            parameters=[{
                "marker_size": MARKER_SIZE,
                "aruco_dictionary_id": ARUCO_DICT,
                "image_topic": f"/{ns}/image_raw",
                "camera_info_topic": f"/{ns}/camera_info",
                "camera_frame": cam["frame_id"],
                "output_pose_topic": "aruco_poses",
                "output_markers_topic": "aruco_markers",
            }],
        ))

        # --- static extrinsic transform: world -> camera optical frame ---
        x, y, z = cam["world_xyz"]
        qx, qy, qz, qw = cam["world_quat"]
        actions.append(Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name=f"{ns}_static_tf",
            arguments=[
                str(x), str(y), str(z),
                str(qx), str(qy), str(qz), str(qw),
                "world", cam["frame_id"],
            ],
        ))

    return LaunchDescription(actions)