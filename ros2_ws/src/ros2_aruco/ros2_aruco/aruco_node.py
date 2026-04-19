import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class ArucoNode(rclpy.node.Node):

    def __init__(self):
        super().__init__("aruco_node")

        # ---------------- PARAMETERS ---------------- #

        self.declare_parameter(
            "marker_size",
            0.15,  # recommended realistic default (15 cm)
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Size of marker in meters"
            )
        )

        self.declare_parameter(
            "aruco_dictionary_id",
            "DICT_4X4_250",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Aruco dictionary"
            )
        )

        self.declare_parameter(
            "image_topic",
            "/camera/image_raw",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Image topic"
            )
        )

        self.declare_parameter(
            "camera_info_topic",
            "/camera/camera_info",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera info topic"
            )
        )

        self.declare_parameter(
            "camera_frame",
            "",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Camera optical frame"
            )
        )

        # ---------------- READ PARAMETERS ---------------- #

        self.marker_size = 0.5
        dictionary_name = "DICT_4X4_250"
        image_topic = "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/image"
        info_topic = "/world/aruco/model/x500_mono_cam_down_0/link/camera_link/sensor/camera/camera_info"
        self.camera_frame = self.get_parameter("camera_frame").value

        self.get_logger().info(f"Marker size: {self.marker_size}")
        self.get_logger().info(f"Dictionary: {dictionary_name}")

        # ---------------- ARUCO SETUP ---------------- #

        try:
            dictionary_id = getattr(cv2.aruco, dictionary_name)
        except AttributeError:
            self.get_logger().error(f"Invalid dictionary: {dictionary_name}")
            valid = [x for x in dir(cv2.aruco) if x.startswith("DICT")]
            self.get_logger().error(f"Valid options:\n{valid}")
            raise

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()

        self.detector = cv2.aruco.ArucoDetector(
            self.aruco_dictionary,
            self.aruco_parameters
        )

        # ---------------- ROS SETUP ---------------- #

        self.bridge = CvBridge()

        self.info_sub = self.create_subscription(
            CameraInfo,
            info_topic,
            self.info_callback,
            qos_profile_sensor_data
        )

        self.image_sub = self.create_subscription(
            Image,
            image_topic,
            self.image_callback,
            qos_profile_sensor_data
        )

        self.poses_pub = self.create_publisher(PoseArray, "aruco_poses", 10)
        self.markers_pub = self.create_publisher(ArucoMarkers, "aruco_markers", 10)

        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

    # ---------------- CAMERA INFO ---------------- #

    def info_callback(self, msg):
        self.info_msg = msg
        self.intrinsic_mat = np.array(msg.k).reshape((3, 3))
        self.distortion = np.array(msg.d)

        self.get_logger().info("Camera info received")
        self.destroy_subscription(self.info_sub)

    # ---------------- IMAGE CALLBACK ---------------- #

    def image_callback(self, img_msg):

        if self.info_msg is None:
            self.get_logger().warn("Waiting for camera info...")
            return

        cv_image = self.bridge.imgmsg_to_cv2(
            img_msg,
            desired_encoding="mono8"
        )

        corners, marker_ids, _ = self.detector.detectMarkers(cv_image)

        if marker_ids is None:
            return

        markers_msg = ArucoMarkers()
        pose_array = PoseArray()

        frame_id = (
            self.camera_frame
            if self.camera_frame != ""
            else self.info_msg.header.frame_id
        )

        markers_msg.header.frame_id = frame_id
        pose_array.header.frame_id = frame_id
        markers_msg.header.stamp = img_msg.header.stamp
        pose_array.header.stamp = img_msg.header.stamp

        # ---------------- POSE ESTIMATION ---------------- #

        half_size = self.marker_size / 2.0

        object_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        for i, marker_id in enumerate(marker_ids):

            corner = corners[i]

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                corner,
                self.intrinsic_mat,
                self.distortion
            )

            if not success:
                continue

            pose = Pose()

            # Translation
            pose.position.x = float(tvec[0][0])
            pose.position.y = float(tvec[1][0])
            pose.position.z = float(tvec[2][0])

            # Rotation
            rot_matrix = np.eye(4)
            rot_matrix[0:3, 0:3] = cv2.Rodrigues(rvec)[0]

            quat = tf_transformations.quaternion_from_matrix(rot_matrix)

            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            pose_array.poses.append(pose)
            markers_msg.poses.append(pose)
            markers_msg.marker_ids.append(int(marker_id[0]))

        self.poses_pub.publish(pose_array)
        self.markers_pub.publish(markers_msg)


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()