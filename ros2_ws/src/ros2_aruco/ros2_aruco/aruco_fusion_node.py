"""
aruco_fusion_node.py

Subscribes to each camera's /<cam_ns>/aruco_markers topic, transforms every
detected marker pose into the shared "world" frame using the static (or
dynamic) TF transforms published for each camera, and republishes a single
fused ArucoMarkers / PoseArray in world coordinates.

If the same marker is seen by more than one camera in the same update
window, poses are combined with a running median per axis (same style as
your existing smoothing) -- cheap, and robust to occasional bad detections
from one camera.
"""

import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from collections import deque
import numpy as np

import tf2_ros
from tf2_ros import TransformException
import tf2_geometry_msgs  # noqa: F401  (registers PoseStamped transform support)

from geometry_msgs.msg import PoseArray, Pose, PoseStamped
from ros2_aruco_interfaces.msg import ArucoMarkers
from rcl_interfaces.msg import ParameterDescriptor, ParameterType


class ArucoFusionNode(rclpy.node.Node):

    def __init__(self):
        super().__init__("aruco_fusion_node")

        self.declare_parameter(
            "camera_marker_topics",
            ["/cam1/aruco_markers", "/cam2/aruco_markers"],
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING_ARRAY,
                description="List of per-camera ArucoMarkers topics to fuse"
            )
        )

        self.declare_parameter(
            "world_frame",
            "world",
            ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Target frame to fuse all detections into"
            )
        )

        self.declare_parameter(
            "staleness_sec",
            0.5,
            ParameterDescriptor(
                type=ParameterType.PARAMETER_DOUBLE,
                description="Drop markers not seen by any camera within this window"
            )
        )

        topics = self.get_parameter("camera_marker_topics").value
        self.world_frame = self.get_parameter("world_frame").value
        self.staleness_sec = self.get_parameter("staleness_sec").value

        # TF2 buffer/listener to look up world <- camera_frame transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # marker_id -> deque of recent world-frame (x, y, z, stamp)
        self.window_size = 5
        self.fused_history = {}
        self.last_seen = {}  # marker_id -> last stamp (float seconds)

        self.subs = []
        for topic in topics:
            sub = self.create_subscription(
                ArucoMarkers,
                topic,
                self.markers_callback,
                qos_profile_sensor_data
            )
            self.subs.append(sub)
            self.get_logger().info(f"Subscribed to {topic}")

        self.fused_pub = self.create_publisher(ArucoMarkers, "/fused/aruco_markers", 10)
        self.fused_pose_pub = self.create_publisher(PoseArray, "/fused/aruco_poses", 10)

        # Republish the current fused snapshot at a fixed rate rather than
        # only on receipt -- keeps consumers fed even if only one camera
        # is currently seeing anything.
        self.create_timer(0.05, self.publish_fused)  # 20 Hz

    def markers_callback(self, msg: ArucoMarkers):
        source_frame = msg.header.frame_id
        if source_frame == "":
            self.get_logger().warn("Received ArucoMarkers with empty frame_id, skipping")
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                self.world_frame,
                source_frame,
                rclpy.time.Time()  # latest available transform
            )
        except TransformException as ex:
            self.get_logger().warn(
                f"Could not transform {source_frame} -> {self.world_frame}: {ex}",
                throttle_duration_sec=2.0
            )
            return

        stamp_sec = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        for marker_id, pose in zip(msg.marker_ids, msg.poses):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = pose

            try:
                world_ps = tf2_geometry_msgs.do_transform_pose(ps.pose, transform)
            except Exception as ex:
                self.get_logger().warn(f"Transform failed for marker {marker_id}: {ex}")
                continue

            if marker_id not in self.fused_history:
                self.fused_history[marker_id] = deque(maxlen=self.window_size)

            self.fused_history[marker_id].append(
                (world_ps.position.x, world_ps.position.y, world_ps.position.z)
            )
            self.last_seen[marker_id] = stamp_sec

    def publish_fused(self):
        now = self.get_clock().now().nanoseconds * 1e-9

        markers_msg = ArucoMarkers()
        pose_array = PoseArray()
        markers_msg.header.frame_id = self.world_frame
        pose_array.header.frame_id = self.world_frame
        stamp = self.get_clock().now().to_msg()
        markers_msg.header.stamp = stamp
        pose_array.header.stamp = stamp

        stale_ids = []
        for marker_id, history in self.fused_history.items():
            last = self.last_seen.get(marker_id, 0.0)
            if now - last > self.staleness_sec:
                stale_ids.append(marker_id)
                continue

            xs = [p[0] for p in history]
            ys = [p[1] for p in history]
            zs = [p[2] for p in history]

            pose = Pose()
            pose.position.x = float(np.median(xs))
            pose.position.y = float(np.median(ys))
            pose.position.z = float(np.median(zs))
            pose.orientation.w = 1.0  # orientation fusion omitted -- see note below

            pose_array.poses.append(pose)
            markers_msg.poses.append(pose)
            markers_msg.marker_ids.append(marker_id)

        # clean out markers no camera has seen recently
        for marker_id in stale_ids:
            del self.fused_history[marker_id]
            del self.last_seen[marker_id]

        self.fused_pub.publish(markers_msg)
        self.fused_pose_pub.publish(pose_array)


def main(args=None):
    rclpy.init()
    node = ArucoFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()