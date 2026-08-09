#pragma once

#include "lie_lane_detection/core/types.hpp"

#include <rclcpp/rclcpp.hpp>

#include <visualization_msgs/msg/marker_array.hpp>

#include <string>

namespace lie_lane_detection
{

visualization_msgs::msg::MarkerArray lanesToMarkers(
  const std::vector<LaneHypothesis> & lanes, const std::string & frame_id,
  const rclcpp::Time & stamp);

visualization_msgs::msg::MarkerArray mergesToMarkers(
  const std::vector<MergeEvent> & merges, const std::string & frame_id, const rclcpp::Time & stamp);

cv::Mat drawOverlay(
  const cv::Mat & bev_bgr, const std::vector<LaneHypothesis> & lanes,
  const std::vector<MergeEvent> & merges);

/// Draw lane overlays on the original frontal image by inverse-warping BEV polylines.
cv::Mat drawFrontalOverlay(
  const cv::Mat & frontal_bgr, const std::vector<LaneHypothesis> & lanes,
  const std::vector<MergeEvent> & merges, const cv::Mat & H_img2bev);

}  // namespace lie_lane_detection
