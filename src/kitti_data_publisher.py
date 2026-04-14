#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Header
from builtin_interfaces.msg import Time

import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import cv2
from cv_bridge import CvBridge

import os
import glob
from datetime import datetime


class KittiSyncPublisher(Node):

    def __init__(self):
        super().__init__('kitti_sync_publisher')

        self.base_path = "/home/mhatre/ros2_humble/data/2011_09_26_drive_0005_sync/2011_09_26/2011_09_26_drive_0005_sync"

        self.lidar_path = os.path.join(self.base_path, "velodyne_points/data")
        self.image_path = os.path.join(self.base_path, "image_02/data")
        self.timestamp_file = os.path.join(self.base_path, "velodyne_points/timestamps.txt")

        self.playback_rate = 1.0 
        self.loop = True

        self.lidar_files = sorted(glob.glob(os.path.join(self.lidar_path, "*.bin")))
        self.image_files = sorted(glob.glob(os.path.join(self.image_path, "*.png")))

        if not self.lidar_files or not self.image_files:
            self.get_logger().error("Dataset files not found!")
            return

        if len(self.lidar_files) != len(self.image_files):
            self.get_logger().error("Mismatch between LiDAR and image files!")
            return

        self.timestamps = self.load_timestamps(self.timestamp_file)

        if len(self.timestamps) != len(self.lidar_files):
            self.get_logger().error("Mismatch between timestamps and data!")
            return

        self.total_frames = len(self.lidar_files)
        self.index = 0

        self.bridge = CvBridge()

        self.lidar_pub = self.create_publisher(PointCloud2, "/kitti/point_cloud", 10)
        self.image_pub = self.create_publisher(Image, "/kitti/image/color/left", 10)

        self.start_time = self.get_clock().now()

        self.timer = self.create_timer(0.001, self.timer_callback)

        self.get_logger().info(f"Loaded {self.total_frames} frames")
        self.get_logger().info("KITTI Sync Publisher Started")


    def load_timestamps(self, file):
        timestamps = []
        lines = open(file).read().splitlines()

        start_time = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

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

            if start_time is None:
                start_time = dt
                timestamps.append(0.0)
            else:
                timestamps.append((dt - start_time).total_seconds())

        return timestamps

    def timer_callback(self):

        if self.index >= self.total_frames:
            if self.loop:
                self.get_logger().info("Looping dataset")
                self.index = 0
                self.start_time = self.get_clock().now()
            else:
                self.get_logger().info("Playback finished")
                return

        now = self.get_clock().now()
        elapsed = (now - self.start_time).nanoseconds * 1e-9

        elapsed *= self.playback_rate

        frame_time = self.timestamps[self.index]

        if elapsed < frame_time:
            return

        lidar_file = self.lidar_files[self.index]
        points = np.fromfile(lidar_file, dtype=np.float32).reshape(-1, 4)

        img_file = self.image_files[self.index]
        img = cv2.imread(img_file)

        if img is None:
            self.get_logger().error(f"Failed to load image {img_file}")
            self.index += 1
            return

        sec = int(frame_time)
        nanosec = int((frame_time - sec) * 1e9)

        header = Header()
        header.stamp = Time(sec=sec, nanosec=nanosec)
        header.frame_id = "velodyne"

        cloud = pc2.create_cloud_xyz32(header, points[:, :3])

        ros_img = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        ros_img.header.stamp = header.stamp
        ros_img.header.frame_id = "camera_left"

        self.lidar_pub.publish(cloud)
        self.image_pub.publish(ros_img)

        self.get_logger().info(
            f"Frame {self.index} | time: {sec}.{nanosec}"
        )

        self.index += 1


def main(args=None):
    rclpy.init(args=args)
    node = KittiSyncPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()