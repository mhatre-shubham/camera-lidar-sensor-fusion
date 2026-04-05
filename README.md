# LiDAR Camera Sensor Fusion

This repository contains a multi-sensor fusion pipeline that combines camera data and LiDAR point cloud data for robust perception in autonomous driving. The system performs 3D object detection, object classification, object tracking and perception tasks using the KITTI Dataset.

The project is built using ROS2, which provides a modular and scalable framework for developing and integrating different components of the perception pipeline. For camera-based object detection, the system utilizes YOLO (YOLOv11), while LiDAR data is processed using point cloud techniques to extract spatial and geometric information about the environment.

## Pipeline Overview
The system is built using ROS2 and is structured as modular ROS2 nodes, where each node performs a specific task in the perception pipeline.

### 1. KITTI Data Publisher (kitti_data_publisher)
- Utilized the KITTI dataset to publish synchronized 3D LiDAR and camera data.

### 2. LiDAR Preprocessing Node (lidar_preprocessing_node)
- Subscribes to raw LiDAR point cloud data.
- Performs Voxel downsampling, ROI filtering for performance.
- Performs ground plane segmentation using RANSAC to separate ground points from objects points.

### 3. LiDAR 3D Clustering (lidar_cluster_detector_dbscan_node)
- Subscribes to non-ground LiDAR point clouds.
- Performs clustering using DBSCAN.
- Applies geometric filtering to extrac objects and remove small objects or noise.
- Publishes clustered objects ( 3D axis-aligned bonding boxes).

### 4. Object Tracking (lidar_tracking_node)
- Subscribes to clustered LiDAR objects.
- Tracks objects across frames and maintains object identities using a Kalman Filter.
- Utilizes Greedy Nearest-Neighbor for data association.

### 5. Camera Object Detection (camera_object_detection_node)
- Uses deep learning (YOLOv11) to detect objects in camera images.
- Filters relevant classes from the COCO dataset.
- Outputs 2D bounding boxes (vision_msgs).

### 6. Image Overlay (image_overlay_node)
- Subscribes to camera images and 3D tracked object data.
- Uses calibration matrices to project LiDAR points onto the camera image plane.
- Publishes annotated images with object ID, class and distance for visualization and debugging.

### 7. Sensor Fusion (lidar_camera_fusion_node)
- Fuses the lidar and camera sensor data.
- Associates LiDAR and camera detections using projection.
- Matches YOLO object semantic class to persistent 3D Kalman Filter tracks.
- Uses label memory to maintain semantic class information for tracked objects, even if the camera temporarily loses detection.


## Environment
- ROS2 Humble 
- Ubuntu 22.04 LTS

## Dataset
To run the pipeline, the KITTI dataset is required. 

- Download the synchronized and rectified camera and LiDAR data from the KITTI website, and place the raw data in the appropriate folder in the workspace.
[KITTI Dataset – Raw Data and Calibration](http://www.cvlibs.net/datasets/kitti/)
- Ensure the `kitti_data_publisher` node is configured with the file paths to the KITTI camera and LiDAR data directories.

## Setup

1. Clone the repository into your ROS2 workspace:
```bash
cd ~/ros2_humble/src
git clone https://github.com/mhatre-shubham/camera_lidar_perception.git
```

2. Build the package:
```bash
cd ~/ros2_humble
colcon build --packages-select camera_lidar_perception
source install/setup.bash
```
