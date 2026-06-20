#pragma once

#include <string>

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lie_lane_detection/types.hpp"

namespace lie_lane_detection
{

visualization_msgs::msg::MarkerArray lanesToMarkers(
  const std::vector<LaneHypothesis> & lanes,
  const std::string & frame_id,
  const rclcpp::Time & stamp);

visualization_msgs::msg::MarkerArray mergesToMarkers(
  const std::vector<MergeEvent> & merges,
  const std::string & frame_id,
  const rclcpp::Time & stamp);

cv::Mat drawOverlay(
  const cv::Mat & bev_bgr,
  const std::vector<LaneHypothesis> & lanes,
  const std::vector<MergeEvent> & merges);

}  // namespace lie_lane_detection
