import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations
from collections import deque


from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseArray, Pose
from nav_msgs.msg import Odometry 
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
            "DICT_5X5_250",
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
            "/" \
            "camera_info",
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
        self.c = 0
        self.prev_time = 0

        self.window_size = 5
        self.position_history = {}
        self.vel_history = {}
        # ---------------- READ PARAMETERS ---------------- #

        self.marker_size = 0.08664
        dictionary_name = "DICT_5X5_250"
        image_topic = "/camera/image_raw"
        info_topic = "/camera_info"
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
        self.vel_pub = self.create_publisher(Odometry, 'aruco_vel', 10)
        

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
        vel_msg = Odometry()

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
            if not success:
                continue
                
            m_id = int(marker_id[0])

            # 1. Extract the raw position
            raw_x = float(tvec[0][0])
            raw_y = float(tvec[1][0])
            raw_z = float(tvec[2][0])

            # 2. Initialize the history queue if this is a new marker
            if m_id not in self.position_history:
                self.position_history[m_id] = deque(maxlen=self.window_size)

            # 3. Add the new raw position to the sliding window
            self.position_history[m_id].append((raw_x, raw_y, raw_z))

            # 4. Extract lists of individual axes
            history = self.position_history[m_id]
            xs = [p[0] for p in history]
            ys = [p[1] for p in history]
            zs = [p[2] for p in history]

            # 5. Calculate the MEDIAN using numpy
            # (Note: we cast back to float because np.median returns a numpy type)
            pose.position.x = float(np.median(xs))
            pose.position.y = float(np.median(ys))
            pose.position.z = float(np.median(zs))


            vel_msg.pose.pose.position.x = float(np.median(xs)) 
            vel_msg.pose.pose.position.y = float(np.median(ys))
            vel_msg.pose.pose.position.z = float(np.median(zs))


            pose_array.poses.append(pose)
            markers_msg.poses.append(pose)
            markers_msg.marker_ids.append(int(marker_id[0]))
             # ---------------- Velocity ---------------- #
            seconds = img_msg.header.stamp.sec
            nanoseconds = img_msg.header.stamp.nanosec
            time_in_seconds = img_msg.header.stamp.sec + (img_msg.header.stamp.nanosec * 1e-9)
            dt = time_in_seconds - self.prev_time
            if dt > 0 and self.c == 1:
                raw_vx = (pose.position.x - self.prev_x)/dt
                raw_vy = (pose.position.y - self.prev_y)/dt
                raw_vz = (pose.position.z - self.prev_z)/dt
                if m_id not in self.vel_history:
                    self.vel_history[m_id] = deque(maxlen=self.window_size)
                self.vel_history[m_id].append((raw_vx, raw_vy, raw_vz))
                history2 = self.vel_history[m_id]
                vx = [p[0] for p in history2]
                vy = [p[1] for p in history2]
                vz = [p[2] for p in history2]
                vel_msg.twist.twist.linear.x = float(np.median(vx))
                vel_msg.twist.twist.linear.y = float(np.median(vy))
                vel_msg.twist.twist.linear.z = float(np.median(vz))
            self.prev_x = pose.position.x
            self.prev_y = pose.position.y
            self.prev_z = pose.position.z
            self.prev_time = time_in_seconds
            self.c = 1

        self.poses_pub.publish(pose_array)
        self.markers_pub.publish(markers_msg)
        self.vel_pub.publish(vel_msg)

       


def main(args=None):
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()