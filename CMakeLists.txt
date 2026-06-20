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

set(CORE_SOURCES
  src/template_curve.cpp
  src/ipm_transformer.cpp
  src/edge_extractor.cpp
  src/lie_hough_voter.cpp
  src/manifold_ransac.cpp
  src/multi_lane_extractor.cpp
  src/merge_topology.cpp
  src/lane_detection_pipeline.cpp
  src/lane_detection_runner.cpp
  src/visualization.cpp
  src/synthetic_bev_generator.cpp
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

add_executable(lane_detector_node src/lane_detector_node.cpp)
target_link_libraries(lane_detector_node lie_lane_detection_core)

add_executable(lane_detect_offline src/lane_detect_offline.cpp)
target_link_libraries(lane_detect_offline lie_lane_detection_core ${OpenCV_LIBS})

add_executable(generate_test_images src/generate_test_images.cpp)
target_link_libraries(generate_test_images lie_lane_detection_core ${OpenCV_LIBS})
ament_target_dependencies(lane_detector_node
  rclcpp
  sensor_msgs
  geometry_msgs
  visualization_msgs
  std_msgs
  cv_bridge
  image_transport
)

install(TARGETS lie_lane_detection_core lane_detector_node lane_detect_offline generate_test_images
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

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)

  ament_add_gtest(test_template_curve test/test_template_curve.cpp)
  target_link_libraries(test_template_curve lie_lane_detection_core)

  ament_add_gtest(test_lie_hough test/test_lie_hough.cpp)
  target_link_libraries(test_lie_hough lie_lane_detection_core)

  ament_add_gtest(test_manifold_ransac test/test_manifold_ransac.cpp)
  target_link_libraries(test_manifold_ransac lie_lane_detection_core)

  ament_add_gtest(test_multi_lane test/test_multi_lane.cpp)
  target_link_libraries(test_multi_lane lie_lane_detection_core)

  ament_add_gtest(test_curved_lane test/test_curved_lane.cpp)
  target_link_libraries(test_curved_lane lie_lane_detection_core)

  ament_add_gtest(test_merge_topology test/test_merge_topology.cpp)
  target_link_libraries(test_merge_topology lie_lane_detection_core)
endif()

ament_package()
