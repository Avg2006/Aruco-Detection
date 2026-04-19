import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg import State
from mavros_msgs.srv import CommandBool, SetMode
from rclpy.qos import QoSProfile, ReliabilityPolicy

import time


class Drone(Node):

    def _init_(self):
        super()._init_('offboard_control')

        self.state = State()
        self.current_pose = PoseStamped()

        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )

        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self.state_cb,
            10
        )

        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/mavros/local_position/pose',
            self.pose_cb,
            qos
        )

        self.pos_pub = self.create_publisher(
            PoseStamped,
            '/mavros/setpoint_position/local',
            10
        )


        self.arm_client = self.create_client(
            CommandBool,
            '/mavros/cmd/arming'
        )

        self.mode_client = self.create_client(
            SetMode,
            '/mavros/set_mode'
        )


        self.pose = PoseStamped()
        self.pose.pose.position.x = 0.0
        self.pose.pose.position.y = 0.0
        self.pose.pose.position.z = 2.5

    def state_cb(self, msg):
        self.state = msg

    def pose_cb(self, msg):
        self.current_pose = msg


def main(args=None):
    rclpy.init(args=args)
    node = Drone()

    node.get_logger().info("Waiting for FCU connection...")

    node.arm_client.wait_for_service()
    node.mode_client.wait_for_service()


    while rclpy.ok() and not node.state.connected:
        rclpy.spin_once(node)

    node.get_logger().info("Connecting...")


    for _ in range(100):
        node.pos_pub.publish(node.pose)
        rclpy.spin_once(node)
        time.sleep(0.05)

    offboard_req = SetMode.Request()
    offboard_req.custom_mode = "OFFBOARD"

    arm_req = CommandBool.Request()
    arm_req.value = True

    last_req = time.time()

    node.get_logger().info("offboarding and arming...")


    while rclpy.ok():
        node.pos_pub.publish(node.pose)
        rclpy.spin_once(node)

        if node.state.mode != "OFFBOARD" and (time.time() - last_req > 1.0):
            node.get_logger().info("offboardin...")
            node.mode_client.call_async(offboard_req)
            last_req = time.time()

        elif not node.state.armed and (time.time() - last_req > 1.0):
            node.get_logger().info("Armin....")
            node.arm_client.call_async(arm_req)
            last_req = time.time()

        if node.state.mode == "OFFBOARD" and node.state.armed:
            node.get_logger().info("offboarded and armed")
            break

        time.sleep(0.05)


    node.get_logger().info("reaching 5.5m")

    while rclpy.ok():
        current_z = node.current_pose.pose.position.z

        node.pos_pub.publish(node.pose)
        rclpy.spin_once(node)


        time.sleep(0.05)




if __name__ == '_main_':
    main()

