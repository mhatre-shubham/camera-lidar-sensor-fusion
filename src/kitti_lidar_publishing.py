#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from builtin_interfaces.msg import Time

import numpy as np
import os
import glob
import time
from datetime import datetime


class KittiLidarPublisher(Node):

    def __init__(self):
        super().__init__('kitti_lidar_publisher')

        # Folder containing KITTI .bin files
        self.dataset_path = "/home/mhatre/ros2_humble/src/camera_lidar_perception/kitti_dataset/2011_09_26_drive_0061_sync/2011_09_26/2011_09_26_drive_0061_sync/velodyne_points/data"

        # Path to timestamp file (date-time format)
        self.timestamp_file = os.path.join(os.path.dirname(self.dataset_path), "timestamps.txt")

        # Load all .bin files
        self.files = sorted(glob.glob(os.path.join(self.dataset_path, "*.bin")))
        if not self.files:
            self.get_logger().error(f"No .bin files found in {self.dataset_path}")
            return

        # Load and parse timestamps
        self.timestamps = self.load_kitti_timestamps(self.timestamp_file)
        if len(self.files) != len(self.timestamps):
            self.get_logger().error("Number of timestamps and .bin files do not match!")
            return

        self.index = 0
        self.start_time = time.time()  # wall-clock time for playback

        # Publisher
        self.publisher = self.create_publisher(PointCloud2, "/kitti/point_cloud", 10)

        # Timer triggers frequently; publishing is throttled by actual KITTI frame timing
        self.timer = self.create_timer(0.01, self.publish_pointcloud)

        self.get_logger().info("KITTI LiDAR publisher started")

    def load_kitti_timestamps(self, timestamp_file):
        """
        Parse KITTI timestamps with flexible fractional seconds.
        Returns a list of seconds relative to the first frame.
        """
        timestamps = []
        lines = open(timestamp_file).read().splitlines()
        start_time = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Split date and time
            try:
                date_str, time_str = line.split(' ')
            except ValueError:
                self.get_logger().error(f"Invalid timestamp format: {line}")
                continue

            hms = time_str.split(':')
            sec_parts = hms[2].split('.')  # seconds and fractional part
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

            if start_time is None:
                start_time = dt
                timestamps.append(0.0)
            else:
                delta = (dt - start_time).total_seconds()
                timestamps.append(delta)

        return timestamps

    def publish_pointcloud(self):
        if self.index >= len(self.files):
            self.get_logger().info("Finished publishing dataset")
            return

        # Calculate elapsed playback time
        elapsed_playback = time.time() - self.start_time

        # Only publish if it's time for the next frame
        frame_time = self.timestamps[self.index]
        if elapsed_playback < frame_time:
            return

        file = self.files[self.index]
        points = np.fromfile(file, dtype=np.float32).reshape(-1, 4)

        header = Header()
        header.frame_id = "velodyne"

        # Convert relative seconds to ROS2 Time
        sec = int(frame_time)
        nanosec = int((frame_time - sec) * 1e9)
        header.stamp = Time(sec=sec, nanosec=nanosec)

        cloud = pc2.create_cloud_xyz32(header, points[:, :3])
        self.publisher.publish(cloud)

        self.get_logger().info(f"Published {os.path.basename(file)} at {sec}.{nanosec}")

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = KittiLidarPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()