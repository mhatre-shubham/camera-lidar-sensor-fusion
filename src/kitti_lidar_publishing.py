#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

import numpy as np
import os
import glob

from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header


class KittiLidarPublisher(Node):

    def __init__(self):
        super().__init__('kitti_lidar_publisher')

        # Folder containing KITTI .bin files
        self.dataset_path = "/home/mhatre/ros2_humble/src/camera_lidar_perception/kitti_dataset/2011_09_26_drive_0061_sync/2011_09_26/2011_09_26_drive_0061_sync/velodyne_points/data"

        # Get all bin files
        self.files = sorted(glob.glob(os.path.join(self.dataset_path, "*.bin")))
        self.index = 0

        # Publisher
        self.publisher = self.create_publisher(PointCloud2, "/velodyne_points", 10)

        # Timer (10 Hz like KITTI LiDAR)
        self.timer = self.create_timer(0.1, self.publish_pointcloud)

        self.get_logger().info("KITTI LiDAR publisher started")

    def publish_pointcloud(self):

        if self.index >= len(self.files):
            self.get_logger().info("Finished publishing dataset")
            return

        file = self.files[self.index]

        # Load .bin file
        points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)

        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "velodyne"

        cloud = pc2.create_cloud_xyz32(header, points[:, :3])

        self.publisher.publish(cloud)

        self.get_logger().info(f"Published {file}")

        self.index += 1


def main(args=None):
    rclpy.init(args=args)

    node = KittiLidarPublisher()

    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
