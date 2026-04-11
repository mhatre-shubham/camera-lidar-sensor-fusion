cmake_minimum_required(VERSION 3.8)
project(camera_lidar_perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(PCL REQUIRED)
find_package(rclpy REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(Eigen3 REQUIRED)

find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(
  include
  include/clustering
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# DBSCAN Library
add_library(dbscan_lib src/dbscan.cpp)

target_link_libraries(dbscan_lib ${PCL_LIBRARIES})


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

# 1. lidar_preprocessing_node
add_executable(lidar_preprocessing_node src/lidar_preprocessing_node.cpp)
ament_target_dependencies(lidar_preprocessing_node rclcpp sensor_msgs pcl_conversions)
target_link_libraries(lidar_preprocessing_node ${PCL_LIBRARIES})

# 2. lidar_cluster_detector_node
add_executable(lidar_cluster_detector_node src/lidar_cluster_detector_node.cpp)
ament_target_dependencies(lidar_cluster_detector_node
  rclcpp
  sensor_msgs
  visualization_msgs
  vision_msgs
  pcl_conversions
)
target_link_libraries(lidar_cluster_detector_node ${PCL_LIBRARIES})

# 3. lidar_tracking_node
add_executable(lidar_tracking_node src/lidar_tracking_node.cpp)
ament_target_dependencies(lidar_tracking_node
  rclcpp
  visualization_msgs
  vision_msgs
)
target_link_libraries(lidar_tracking_node Eigen3::Eigen)

# 4. image_overlay_node
add_executable(image_overlay_node src/image_overlay_node.cpp)
ament_target_dependencies(image_overlay_node
  rclcpp
  sensor_msgs
  vision_msgs
  visualization_msgs
  cv_bridge
)
target_link_libraries(image_overlay_node ${OpenCV_LIBS})

# 5. lidar_camera_Projection
add_executable(lidar_camera_projection_node src/lidar_camera_projection_node.cpp)
ament_target_dependencies(lidar_camera_projection_node
  rclcpp
  sensor_msgs
  cv_bridge
  message_filters
)
target_link_libraries(lidar_camera_projection_node ${OpenCV_LIBS})


# 6. lidar_camera_fusion_node
add_executable(lidar_camera_fusion_node src/lidar_camera_fusion_node.cpp)
target_include_directories(lidar_camera_fusion_node PUBLIC
  ${OpenCV_INCLUDE_DIRS}
)
ament_target_dependencies(lidar_camera_fusion_node
  rclcpp
  sensor_msgs
  vision_msgs
  visualization_msgs
  std_msgs
  cv_bridge
)
target_link_libraries(lidar_camera_fusion_node ${OpenCV_LIBRARIES})

# 7. lidar_cluster_detector_dbscan_node
add_executable(lidar_cluster_detector_dbscan_node src/lidar_cluster_detector_dbscan_node.cpp)
ament_target_dependencies(lidar_cluster_detector_dbscan_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  vision_msgs
)
target_link_libraries(lidar_cluster_detector_dbscan_node 
  ${PCL_LIBRARIES}
  dbscan_lib)


install(TARGETS 
  lidar_preprocessing_node 
  lidar_cluster_detector_node
  lidar_tracking_node
  image_overlay_node
  lidar_camera_projection_node
  lidar_camera_fusion_node
  lidar_cluster_detector_dbscan_node
  DESTINATION lib/${PROJECT_NAME}
)

# Python
install(PROGRAMS
  src/kitti_lidar_publisher.py
  src/camera_object_detection_node.py
  src/kitti_camera_publisher.py
  src/kitti_data_publisher.py
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
