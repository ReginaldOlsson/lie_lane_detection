cmake_minimum_required(VERSION 3.8)
project(lie_lane_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(Sophus REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(TBB REQUIRED)

include_directories(
  include
  ${EIGEN3_INCLUDE_DIRS}
)

# --- Core library (grouped by module) ---
set(CORE_SOURCES
  src/geometry/template_curve.cpp
  src/preprocessing/ipm_transformer.cpp
  src/preprocessing/auto_frontal_ipm.cpp
  src/preprocessing/edge_extractor.cpp
  src/voting/lie_hough_voter.cpp
  src/voting/line_hough_extractor.cpp
  src/voting/line_lie_hough_voter.cpp
  src/fitting/manifold_ransac.cpp
  src/extraction/multi_lane_extractor.cpp
  src/extraction/merge_topology.cpp
  src/pipeline/detection_common.cpp
  src/pipeline/lane_detection_runner.cpp
  src/pipeline/lane_detection_pipeline.cpp
  src/pipeline/line_lane_detection_runner.cpp
  src/pipeline/line_lane_detection_pipeline.cpp
  src/tracking/lane_tracker.cpp
  src/motion/ego_motion_estimator.cpp
  src/visualization/visualization.cpp
  src/testing/synthetic_bev_generator.cpp
)

add_library(lie_lane_detection_core SHARED ${CORE_SOURCES})
target_include_directories(lie_lane_detection_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(lie_lane_detection_core
  Sophus::Sophus
  TBB::tbb
  ${OpenCV_LIBS}
)
ament_target_dependencies(lie_lane_detection_core
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  cv_bridge
)

# --- ROS nodes ---
add_executable(lane_detector_node src/nodes/lane_detector_node.cpp)
target_link_libraries(lane_detector_node lie_lane_detection_core)
ament_target_dependencies(lane_detector_node
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  cv_bridge
  image_transport
)

add_executable(line_lane_detector_node src/nodes/line_lane_detector_node.cpp)
target_link_libraries(line_lane_detector_node lie_lane_detection_core)
ament_target_dependencies(line_lane_detector_node
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  cv_bridge
  image_transport
)

add_executable(video_image_publisher_node src/nodes/video_image_publisher_node.cpp)
target_link_libraries(video_image_publisher_node ${OpenCV_LIBS})
ament_target_dependencies(video_image_publisher_node
  rclcpp
  sensor_msgs
  cv_bridge
)

add_executable(tracked_lane_detector_node src/nodes/tracked_lane_detector_node.cpp)
target_link_libraries(tracked_lane_detector_node lie_lane_detection_core)
ament_target_dependencies(tracked_lane_detector_node
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  cv_bridge
  image_transport
)

# --- Offline tools ---
add_executable(lane_detect_offline src/tools/lane_detect_offline.cpp)
target_link_libraries(lane_detect_offline lie_lane_detection_core ${OpenCV_LIBS})

add_executable(compare_lane_detection src/tools/compare_lane_detection.cpp)
target_link_libraries(compare_lane_detection lie_lane_detection_core ${OpenCV_LIBS})

add_executable(generate_test_images src/tools/generate_test_images.cpp)
target_link_libraries(generate_test_images lie_lane_detection_core ${OpenCV_LIBS})

add_executable(lane_detect_video_offline src/tools/lane_detect_video_offline.cpp)
target_link_libraries(lane_detect_video_offline lie_lane_detection_core ${OpenCV_LIBS})

add_executable(estimate_frontal_homography src/tools/estimate_frontal_homography.cpp)
target_link_libraries(estimate_frontal_homography lie_lane_detection_core ${OpenCV_LIBS})

install(TARGETS
  lie_lane_detection_core
  lane_detector_node
  line_lane_detector_node
  video_image_publisher_node
  tracked_lane_detector_node
  lane_detect_offline
  lane_detect_video_offline
  estimate_frontal_homography
  compare_lane_detection
  generate_test_images
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY include/
  DESTINATION include
)

install(DIRECTORY config launch
  DESTINATION share/${PROJECT_NAME}
)

# --- Unit tests (mirror src/ layout) ---
if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_template_curve test/geometry/test_template_curve.cpp)
  target_link_libraries(test_template_curve lie_lane_detection_core)

  ament_add_gtest(test_lie_hough test/voting/test_lie_hough.cpp)
  target_link_libraries(test_lie_hough lie_lane_detection_core)

  ament_add_gtest(test_manifold_ransac test/fitting/test_manifold_ransac.cpp)
  target_link_libraries(test_manifold_ransac lie_lane_detection_core)

  ament_add_gtest(test_multi_lane test/extraction/test_multi_lane.cpp)
  target_link_libraries(test_multi_lane lie_lane_detection_core)

  ament_add_gtest(test_curved_lane test/pipeline/test_curved_lane.cpp)
  target_link_libraries(test_curved_lane lie_lane_detection_core)

  ament_add_gtest(test_merge_topology test/extraction/test_merge_topology.cpp)
  target_link_libraries(test_merge_topology lie_lane_detection_core)

  ament_add_gtest(test_lane_tracker test/tracking/test_lane_tracker.cpp)
  target_link_libraries(test_lane_tracker lie_lane_detection_core)

  ament_add_gtest(test_vanishing_point_tracker test/preprocessing/test_vanishing_point_tracker.cpp)
  target_link_libraries(test_vanishing_point_tracker lie_lane_detection_core)

  ament_add_gtest(test_ego_motion_estimator test/motion/test_ego_motion_estimator.cpp)
  target_link_libraries(test_ego_motion_estimator lie_lane_detection_core)
endif()

ament_package()
