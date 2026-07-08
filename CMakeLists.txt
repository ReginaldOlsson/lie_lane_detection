cmake_minimum_required(VERSION 3.8)
project(lie_lane_detection)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # Scope strict warnings to C++ only so nvcc is not handed -Wpedantic etc.
  add_compile_options(
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Optional CUDA (custom Lie-Hough voting kernels) ---
# Enabled when the CUDA toolkit is found. Falls back to a CPU-only build
# otherwise, so the package still compiles on machines without CUDA.
option(LIE_USE_CUDA "Build custom CUDA voting kernels when a toolkit is available" ON)
set(LIE_CUDA_ENABLED FALSE)
if(LIE_USE_CUDA)
  include(CheckLanguage)
  find_package(CUDAToolkit QUIET)
  if(CUDAToolkit_FOUND)
    # CUDA 12.0's host_config caps at gcc-12; pin it if the default is newer.
    if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER AND EXISTS /usr/bin/g++-12)
      set(CMAKE_CUDA_HOST_COMPILER /usr/bin/g++-12)
    endif()
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      enable_language(CUDA)
      if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 89)  # Ada (RTX 40-series); override if needed
      endif()
      set(LIE_CUDA_ENABLED TRUE)
      message(STATUS "lie_lane_detection: CUDA enabled (arch ${CMAKE_CUDA_ARCHITECTURES}, host ${CMAKE_CUDA_HOST_COMPILER})")
    endif()
  endif()
  if(NOT LIE_CUDA_ENABLED)
    message(STATUS "lie_lane_detection: CUDA toolkit not found - building CPU-only")
  endif()
endif()

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
find_package(Ceres REQUIRED)

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
  src/voting/sparse_accumulator.cpp
  src/voting/lie_exp_cache.cpp
  src/voting/peak_refinement.cpp
  src/voting/line_kdtree.cpp
  src/voting/dual_space_pruner.cpp
  src/voting/longitudinal_line_filter.cpp
  src/voting/coarse_se2_voter.cpp
  src/fitting/manifold_ransac.cpp
  src/fitting/observation_association.cpp
  src/fitting/line_segment_ransac_gate.cpp
  src/fitting/ceres_lane_fitter.cpp
  src/fitting/road_manifold_fitter.cpp
  src/extraction/multi_lane_extractor.cpp
  src/extraction/merge_topology.cpp
  src/pipeline/detection_common.cpp
  src/pipeline/lane_detection_runner.cpp
  src/pipeline/intensity_strip_lane_detector.cpp
  src/pipeline/lane_detection_pipeline.cpp
  src/pipeline/line_lane_detection_runner.cpp
  src/pipeline/line_lane_detection_pipeline.cpp
  src/tracking/lane_tracker.cpp
  src/motion/ego_motion_estimator.cpp
  src/mosaic/bev_mosaic_accumulator.cpp
  src/mosaic/bev_registration.cpp
  src/mosaic/bev_orb_matcher.cpp
  src/nodes/node_params.cpp
  src/visualization/visualization.cpp
  src/testing/synthetic_bev_generator.cpp
)

if(LIE_CUDA_ENABLED)
  list(APPEND CORE_SOURCES src/voting/cuda_voting.cu)
endif()

add_library(lie_lane_detection_core SHARED ${CORE_SOURCES})

if(LIE_CUDA_ENABLED)
  target_compile_definitions(lie_lane_detection_core PUBLIC LIE_HAS_CUDA)
  # Resolve device symbols into the shared library so pure-C++ consumers
  # (nodes, tools, tests) link without needing a CUDA device-link step.
  set_target_properties(lie_lane_detection_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  target_link_libraries(lie_lane_detection_core CUDA::cudart)
endif()
target_include_directories(lie_lane_detection_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(lie_lane_detection_core
  Sophus::Sophus
  TBB::tbb
  Ceres::ceres
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

add_executable(frontal_ipm_node src/nodes/frontal_ipm_node.cpp)
target_link_libraries(frontal_ipm_node lie_lane_detection_core)
ament_target_dependencies(frontal_ipm_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

add_executable(bev_lane_detector_node src/nodes/bev_lane_detector_node.cpp)
target_link_libraries(bev_lane_detector_node lie_lane_detection_core)
ament_target_dependencies(bev_lane_detector_node
  rclcpp
  sensor_msgs
  visualization_msgs
  std_msgs
  cv_bridge
  image_transport
)

add_executable(bev_mosaic_node src/nodes/bev_mosaic_node.cpp)
target_link_libraries(bev_mosaic_node lie_lane_detection_core)
ament_target_dependencies(bev_mosaic_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
)

add_executable(intensity_strip_lane_detector_node src/nodes/intensity_strip_lane_detector_node.cpp)
target_link_libraries(intensity_strip_lane_detector_node lie_lane_detection_core)
ament_target_dependencies(intensity_strip_lane_detector_node
  rclcpp
  sensor_msgs
  visualization_msgs
  std_msgs
  cv_bridge
  image_transport
)

# --- Offline tools ---
add_executable(lane_detect_offline src/tools/lane_detect_offline.cpp)
target_link_libraries(lane_detect_offline lie_lane_detection_core ${OpenCV_LIBS})

add_executable(benchmark_bev_detection src/tools/benchmark_bev_detection.cpp)
target_link_libraries(benchmark_bev_detection lie_lane_detection_core ${OpenCV_LIBS})

add_executable(compare_lane_detection src/tools/compare_lane_detection.cpp)
target_link_libraries(compare_lane_detection lie_lane_detection_core ${OpenCV_LIBS})

add_executable(generate_test_images src/tools/generate_test_images.cpp)
target_link_libraries(generate_test_images lie_lane_detection_core ${OpenCV_LIBS})

add_executable(lane_detect_video_offline src/tools/lane_detect_video_offline.cpp)
target_link_libraries(lane_detect_video_offline lie_lane_detection_core ${OpenCV_LIBS})

add_executable(estimate_frontal_homography src/tools/estimate_frontal_homography.cpp)
target_link_libraries(estimate_frontal_homography lie_lane_detection_core ${OpenCV_LIBS})

add_executable(test_bev_mosaic_align src/tools/test_bev_mosaic_align.cpp)
target_link_libraries(test_bev_mosaic_align lie_lane_detection_core ${OpenCV_LIBS})

install(TARGETS
  lie_lane_detection_core
  lane_detector_node
  line_lane_detector_node
  video_image_publisher_node
  tracked_lane_detector_node
  frontal_ipm_node
  bev_lane_detector_node
  bev_mosaic_node
  intensity_strip_lane_detector_node
  lane_detect_offline
  benchmark_bev_detection
  lane_detect_video_offline
  estimate_frontal_homography
  test_bev_mosaic_align
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

  ament_add_gtest(test_ipm_roi test/preprocessing/test_ipm_roi.cpp)
  target_link_libraries(test_ipm_roi lie_lane_detection_core)

  ament_add_gtest(test_bev_mosaic test/mosaic/test_bev_mosaic.cpp)
  target_link_libraries(test_bev_mosaic lie_lane_detection_core)

  ament_add_gtest(test_ego_motion_estimator test/motion/test_ego_motion_estimator.cpp)
  target_link_libraries(test_ego_motion_estimator lie_lane_detection_core)

  ament_add_gtest(test_ceres_lane_fitter test/fitting/test_ceres_lane_fitter.cpp)
  target_link_libraries(test_ceres_lane_fitter lie_lane_detection_core)

  ament_add_gtest(test_coarse_se2_voter test/voting/test_coarse_se2_voter.cpp)
  target_link_libraries(test_coarse_se2_voter lie_lane_detection_core)

  ament_add_gtest(test_longitudinal_line_filter test/voting/test_longitudinal_line_filter.cpp)
  target_link_libraries(test_longitudinal_line_filter lie_lane_detection_core)

  ament_add_gtest(test_road_manifold_fitter test/fitting/test_road_manifold_fitter.cpp)
  target_link_libraries(test_road_manifold_fitter lie_lane_detection_core)

  if(LIE_CUDA_ENABLED)
    ament_add_gtest(test_cuda_voting test/voting/test_cuda_voting.cpp)
    target_link_libraries(test_cuda_voting lie_lane_detection_core)
  endif()
endif()

ament_package()
