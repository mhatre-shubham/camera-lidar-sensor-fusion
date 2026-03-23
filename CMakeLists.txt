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
  ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})


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
  pcl_conversions
)
target_link_libraries(lidar_cluster_detector_node ${PCL_LIBRARIES})

# 3. lidar_tracking_node
add_executable(lidar_tracking_node src/lidar_tracking_node.cpp)
ament_target_dependencies(lidar_tracking_node
  rclcpp
  visualization_msgs
)
target_link_libraries(lidar_tracking_node Eigen3::Eigen)

install(TARGETS 
  lidar_preprocessing_node 
  lidar_cluster_detector_node
  lidar_tracking_node
  DESTINATION lib/${PROJECT_NAME}
)

# Python
install(PROGRAMS
  src/kitti_lidar_publishing.py
  src/camera_object_detection_node.py
  src/kitti_camera_publishing.py
  DESTINATION lib/${PROJECT_NAME}
)
ament_package()
