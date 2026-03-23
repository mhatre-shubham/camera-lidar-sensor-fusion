#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from builtin_interfaces.msg import Time
from std_msgs.msg import Header

import os
import glob
import time
from datetime import datetime
from cv_bridge import CvBridge
import cv2
import numpy as np

class KittiCameraPublisher(Node):

    def __init__(self):
        super().__init__('kitti_camera_publisher')

        # KITTI camera folder
        self.dataset_path = "/home/mhatre/ros2_humble/src/camera_lidar_perception/kitti_dataset/2011_09_26_drive_0061_sync/2011_09_26/2011_09_26_drive_0061_sync/image_02/data"
        self.timestamp_file = os.path.join(os.path.dirname(self.dataset_path), "timestamps.txt")

        # Load image files
        self.files = sorted(glob.glob(os.path.join(self.dataset_path, "*.png")))
        if not self.files:
            self.get_logger().error(f"No image files found in {self.dataset_path}")
            return

        # Load and parse timestamps
        self.timestamps = self.load_kitti_timestamps(self.timestamp_file)
        if len(self.files) != len(self.timestamps):
            self.get_logger().error("Number of timestamps and images do not match!")
            return

        self.index = 0
        self.start_time = time.time()  # wall-clock start time

        self.publisher = self.create_publisher(Image, "/kitti/image/color/left", 10)
        self.bridge = CvBridge()

        self.timer = self.create_timer(0.01, self.publish_image)
        self.get_logger().info("KITTI Camera publisher started")

    def load_kitti_timestamps(self, timestamp_file):
        """Parse KITTI timestamps, returns seconds relative to first frame"""
        timestamps = []
        lines = open(timestamp_file).read().splitlines()
        start_time = None
        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Split date and time if needed
            if ' ' in line:
                date_str, time_str = line.split(' ')
                hms = time_str.split(':')
                sec_parts = hms[2].split('.')
                seconds = int(sec_parts[0])
                frac = float('0.' + sec_parts[1]) if len(sec_parts) > 1 else 0.0
                dt = datetime(
                    year=int(date_str[0:4]),
                    month=int(date_str[5:7]),
                    day=int(date_str[8:10]),
                    hour=int(hms[0]),
                    minute=int(hms[1]),
                    second=seconds,
                    microsecond=int(frac * 1e6)
                )
            else:
                # If timestamp is already seconds
                dt = datetime.fromtimestamp(float(line))

            if start_time is None:
                start_time = dt
                timestamps.append(0.0)
            else:
                delta = (dt - start_time).total_seconds()
                timestamps.append(delta)

        return timestamps

    def publish_image(self):
        if self.index >= len(self.files):
            self.get_logger().info("Finished publishing camera dataset")
            return

        # Check if it is time to publish
        elapsed_playback = time.time() - self.start_time
        frame_time = self.timestamps[self.index]
        if elapsed_playback < frame_time:
            return

        # Read image
        file = self.files[self.index]
        img = cv2.imread(file)
        if img is None:
            self.get_logger().error(f"Failed to load image {file}")
            self.index += 1
            return

        # Convert to ROS Image
        ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")

        # Header timestamp
        sec = int(frame_time)
        nanosec = int((frame_time - sec) * 1e9)
        ros_img.header = Header()
        ros_img.header.stamp = Time(sec=sec, nanosec=nanosec)
        ros_img.header.frame_id = "camera"

        self.publisher.publish(ros_img)
        self.get_logger().info(f"Published {os.path.basename(file)} at {sec}.{nanosec}")

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = KittiCameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()