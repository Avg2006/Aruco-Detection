#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
import yaml
import os

class CameraInfoPublisher(Node):
    def __init__(self):
        super().__init__('camera_info_publisher')
        
        # 1. Declare the parameter 'yaml_file_path'
        # We give it a default empty string or a safe default
        self.declare_parameter('yaml_file_path', 'default_ost.yaml')
        
        # 2. Retrieve the parameter value
        yaml_file = self.get_parameter('yaml_file_path').get_parameter_value().string_value         
        
        self.publisher_ = self.create_publisher(CameraInfo, '/camera/camera_info', 10)
        
        self.get_logger().info(f"Attempting to load CameraInfo from: {yaml_file}")
        
        # 3. Load the data using the parameter path
        self.camera_info_msg = self.load_camera_info(yaml_file)
        
        # 4. Create Timer
        timer_period = 1.0 / 30.0
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def load_camera_info(self, yaml_file):
        """Parses the specific YAML format provided and returns a CameraInfo msg"""
        msg = CameraInfo()
        
        try:
            with open(yaml_file, "r") as file:
                calib_data = yaml.safe_load(file)

            msg.header.frame_id = "camera_link"
            msg.width = calib_data['image_width']
            msg.height = calib_data['image_height']
            msg.distortion_model = calib_data['distortion_model']
            msg.d = calib_data['distortion_coefficients']['data']
            msg.k = calib_data['camera_matrix']['data']
            msg.r = calib_data['rectification_matrix']['data']
            msg.p = calib_data['projection_matrix']['data']
            msg.binning_x = 0
            msg.binning_y = 0
            msg.roi.x_offset = 0
            msg.roi.y_offset = 0
            msg.roi.height = 0
            msg.roi.width = 0
            msg.roi.do_rectify = False
            
            self.get_logger().info("Successfully loaded camera info.")
            return msg
            
        except FileNotFoundError:
            self.get_logger().error(f"Could not find file: {yaml_file}")
            return CameraInfo()
        except KeyError as e:
            self.get_logger().error(f"Missing key in YAML file: {e}")
            return CameraInfo()
        except Exception as e:
            self.get_logger().error(f"Error loading YAML: {e}")
            return CameraInfo()

    def timer_callback(self):
        self.camera_info_msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher_.publish(self.camera_info_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CameraInfoPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()