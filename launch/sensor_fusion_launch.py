from launch import LaunchDescription
from launch_ros.actions import Node
 
def generate_launch_description():
    return LaunchDescription([
 
        # 1. Lidar Preprocessing
        Node(
            package='camera_lidar_perception',
            executable='lidar_preprocessing_node',
            name='lidar_preprocessing',
            output='screen'
        ),
 
        # 2. Lidar Cluster Detector
        Node(
            package='camera_lidar_perception',
            executable='lidar_cluster_detector_dbscan_node',
            name='lidar_cluster_detector',
            output='screen'
        ),
       
        # 3. Lidar Tracking
        Node(
            package='camera_lidar_perception',
            executable='lidar_tracking_node',
            name='lidar_tracking',
            output='screen'
        ),
 
        # 4. Image Overlay
        Node(
            package='camera_lidar_perception',
            executable='image_overlay_node',
            name='image_overlay',
            output='screen'
        ),
       
        # 5. Lidar-Camera Projection
        Node(
            package='camera_lidar_perception',
            executable='lidar_camera_projection_node',
            name='lidar_camera_projection',
            output='screen'
        ),
       
        # 6. Lidar-Camera Fusion
        Node(
            package='camera_lidar_perception',
            executable='lidar_camera_fusion_node',
            name='lidar_camera_fusion',
            output='screen'
        ),
       
        # Python nodes
        Node(
            package='camera_lidar_perception',
            executable='camera_object_detection_node.py',
            name='camera_object_detection',
            output='screen'
        ),
    ])
 