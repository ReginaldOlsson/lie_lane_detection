#pragma once

#include <rclcpp/rclcpp.hpp>

#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

/// IPM / homography parameters for frontal_ipm_node.
PipelineParams loadIpmParams(rclcpp::Node & node);

/// Detection-only parameters for bev_lane_detector_node.
PipelineParams loadDetectionParams(rclcpp::Node & node);

}  // namespace lie_lane_detection
