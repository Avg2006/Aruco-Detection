import rclpy
from rclpy.node import Node
from ros2_aruco_interfaces.msg import ArucoMarkers

class Aruco_Subscriber(Node):
    def __init__(self):
        super().__init__('aruco_subscriber')
        
        # Create the subscription
        self.subscription = self.create_subscription(
            ArucoMarkers,
            '/aruco_markers', 
            self.listener_callback,
            10
        )
        self.subscription  # prevent unused variable warning

    def listener_callback(self, msg):
        # self.get_logger().info('--- New Arucomarker Reading ---')
        # self.get_logger().info(f'Frame ID:          "{msg.header.frame_id}"')
        # self.get_logger().info(f'Timestamp:         {msg.header.stamp.sec}.{msg.header.stamp.nanosec}')
        # self.get_logger().info(f'Position x: {msg.poses[0].position.x} m')
        # self.get_logger().info(f'Position y: {msg.poses[0].position.y} m')
        # self.get_logger().info(f'Position z: {msg.poses[0].position.z} m')
        # self.get_logger().info(f'oritentation x: {msg.poses[0].orientation.x} m')
        # self.get_logger().info(f'oritentation y: {msg.poses[0].orientation.y} m')
        # self.get_logger().info(f'oritentation z: {msg.poses[0].orientation.z} m')
        # self.get_logger().info(f'oritentation w: {msg.poses[0].orientation.w} m')

        pose_x = msg.poses[0].position.x
        pose_y = msg.poses[0].position.y

        if ((pose_x**2)+(pose_y**2))**0.5 <= 0.04:
            self.get_logger().info("decrease z")
        else:
            if pose_x > 0.0282:
                self.get_logger().info("Go right")
            elif pose_x < -0.0282:
                self.get_logger().info("Go left")
            if pose_y > 0.0282:
                self.get_logger().info("Go down")
            elif pose_y < -0.0282:
                self.get_logger().info("Go up")
            elif -0.0282 < pose_y<0.0282:
                self.get_logger().info("Y_")

        

def main(args=None):
    rclpy.init(args=args)
    aruco_subscriber = Aruco_Subscriber()
    
    try:
        rclpy.spin(aruco_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up and shut down
        aruco_subscriber.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()