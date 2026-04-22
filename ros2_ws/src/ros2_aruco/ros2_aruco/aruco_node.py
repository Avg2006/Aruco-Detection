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
            0.5,
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
            "camera/image_raw",
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

        # Read parameters
        self.marker_size = self.get_parameter("marker_size").value
        dictionary_name = self.get_parameter("aruco_dictionary_id").value
        image_topic = self.get_parameter("image_topic").value
        info_topic = self.get_parameter("camera_info_topic").value
        self.camera_frame = self.get_parameter("camera_frame").value

        self.get_logger().info(f"Marker size: {self.marker_size}")
        self.get_logger().info(f"Dictionary: {dictionary_name}")

        # ---------------- ARUCO SETUP (NEW API) ---------------- #

        try:
            dictionary_id = getattr(cv2.aruco, dictionary_name)
        except AttributeError:
            self.get_logger().error(f"Invalid dictionary: {dictionary_name}")
            valid = [x for x in dir(cv2.aruco) if x.startswith("DICT")]
            self.get_logger().error(f"Valid options:\n{valid}")
            raise

        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()

        # NEW detector object (OpenCV 4.7+)
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

        self.create_subscription(
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

        # Camera parameters assumed static
        self.destroy_subscription(self.info_sub)

    # ---------------- IMAGE CALLBACK ---------------- #

    def image_callback(self, img_msg):

        if self.info_msg is None:
            self.get_logger().warn("Camera info not received yet")
            return

        cv_image = self.bridge.imgmsg_to_cv2(
            img_msg,
            desired_encoding="mono8"
        )

        # Create a BGR color copy so the drawn axes and boxes show up in color
        display_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)

        # NEW detection method
        corners, marker_ids, rejected = self.detector.detectMarkers(cv_image)

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

        # If markers are detected, compute poses and draw them
        if marker_ids is not None:
            
            # Draw the green bounding box and ID for each marker
            cv2.aruco.drawDetectedMarkers(display_image, corners, marker_ids)

            # Pose estimation (unchanged API)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners,
                self.marker_size,
                self.intrinsic_mat,
                self.distortion
            )

            for i, marker_id in enumerate(marker_ids):

                # Draw the 3D XYZ axes on the marker
                cv2.drawFrameAxes(
                    display_image,
                    self.intrinsic_mat,
                    self.distortion,
                    rvecs[i][0],
                    tvecs[i][0],
                    self.marker_size * 0.5  # Length of the axis lines
                )

                pose = Pose()

                # Translation
                pose.position.x = float(tvecs[i][0][0])
                pose.position.y = float(tvecs[i][0][1])
                pose.position.z = float(tvecs[i][0][2])

                # Rotation
                rot_matrix = np.eye(4)
                rot_matrix[0:3, 0:3] = cv2.Rodrigues(rvecs[i][0])[0]

                quat = tf_transformations.quaternion_from_matrix(rot_matrix)

                pose.orientation.x = float(quat[0])
                pose.orientation.y = float(quat[1])
                pose.orientation.z = float(quat[2])
                pose.orientation.w = float(quat[3])

                pose_array.poses.append(pose)
                markers_msg.poses.append(pose)
                markers_msg.marker_ids.append(int(marker_id[0]))

            # Publish the data
            self.poses_pub.publish(pose_array)
            self.markers_pub.publish(markers_msg)

        # Show the video feed (this happens whether a marker is visible or not)
        cv2.imshow("Aruco Visualization", display_image)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)
    
    # Cleanup OpenCV windows on shutdown
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()