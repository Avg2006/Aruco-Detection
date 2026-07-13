import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations
from collections import deque

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseWithCovarianceStamped

class ArucoFusionNode(Node):
    def __init__(self):
        super().__init__("aruco_ekf")

        # --- PARAMETERS ---
        self.marker_size = 0.08664
        dictionary_name = "DICT_5X5_250"
        image_topic = "/camera/image_raw"
        info_topic = "/camera_info"
        self.camera_frame = "camera_link" # Make sure this matches your camera!

        # --- FILTER SETUP ---
        self.window_size = 5 # Odd number for a true median
        self.position_history = {} # Format: { marker_id: deque() }

        # --- ARUCO SETUP ---
        dictionary_id = getattr(cv2.aruco, dictionary_name)
        self.aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self.aruco_parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dictionary, self.aruco_parameters)

        # --- ROS SETUP ---
        self.bridge = CvBridge()
        self.info_msg = None
        self.intrinsic_mat = None
        self.distortion = None

        self.info_sub = self.create_subscription(CameraInfo, info_topic, self.info_callback, qos_profile_sensor_data)
        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, qos_profile_sensor_data)
        
        # Publisher for the EKF
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, "/aruco/pose_with_covariance", 10)

    def info_callback(self, msg):
        self.info_msg = msg
        self.intrinsic_mat = np.array(msg.k).reshape((3, 3))
        self.distortion = np.array(msg.d)
        self.get_logger().info("Camera info received. Ready to track.")
        self.destroy_subscription(self.info_sub)

    def image_callback(self, img_msg):
        if self.info_msg is None:
            return

        cv_image = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="mono8")
        corners, marker_ids, _ = self.detector.detectMarkers(cv_image)

        if marker_ids is None:
            return

        half_size = self.marker_size / 2.0
        object_points = np.array([
            [-half_size,  half_size, 0],
            [ half_size,  half_size, 0],
            [ half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        for i, marker_id in enumerate(marker_ids):
            corner = corners[i]
            success, rvec, tvec = cv2.solvePnP(object_points, corner, self.intrinsic_mat, self.distortion)

            if not success:
                continue

            m_id = int(marker_id[0])

            # --- MEDIAN FILTERING ---
            raw_x, raw_y, raw_z = float(tvec[0][0]), float(tvec[1][0]), float(tvec[2][0])

            if m_id not in self.position_history:
                self.position_history[m_id] = deque(maxlen=self.window_size)

            self.position_history[m_id].append((raw_x, raw_y, raw_z))
            
            history = self.position_history[m_id]
            avg_x = float(np.median([p[0] for p in history]))
            avg_y = float(np.median([p[1] for p in history]))
            avg_z = float(np.median([p[2] for p in history]))

            # --- BUILD MESSAGE FOR EKF ---
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.stamp = img_msg.header.stamp
            pose_msg.header.frame_id = self.camera_frame

            # Filtered Translation
            pose_msg.pose.pose.position.x = avg_x
            pose_msg.pose.pose.position.y = avg_y
            pose_msg.pose.pose.position.z = avg_z

            # Raw Rotation
            rot_matrix = np.eye(4)
            rot_matrix[0:3, 0:3] = cv2.Rodrigues(rvec)[0]
            quat = tf_transformations.quaternion_from_matrix(rot_matrix)
            
            pose_msg.pose.pose.orientation.x = float(quat[0])
            pose_msg.pose.pose.orientation.y = float(quat[1])
            pose_msg.pose.pose.orientation.z = float(quat[2])
            pose_msg.pose.pose.orientation.w = float(quat[3])

            # --- COVARIANCE MATRIX ---
            # 6x6 diagonal matrix [X, Y, Z, Roll, Pitch, Yaw]
            covariance = [0.0] * 36
            covariance[0]  = 0.05  # X variance 
            covariance[7]  = 0.05  # Y variance
            covariance[14] = 0.05  # Z variance
            covariance[21] = 0.1   # Roll variance
            covariance[28] = 0.1   # Pitch variance
            covariance[35] = 0.1   # Yaw variance
            
            pose_msg.pose.covariance = covariance

            # Publish the data!
            self.pose_pub.publish(pose_msg)

def main():
    rclpy.init()
    node = ArucoFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()