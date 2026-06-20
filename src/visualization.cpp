#include "lie_lane_detection/visualization.hpp"

#include <array>

#include <opencv2/imgproc.hpp>

namespace lie_lane_detection
{

namespace
{

std::array<float, 4> roleColor(LaneRole role)
{
  switch (role) {
    case LaneRole::LEFT_ADJACENT:
      return {0.2f, 0.6f, 1.0f, 1.0f};
    case LaneRole::LEFT_EGO:
      return {0.0f, 1.0f, 0.0f, 1.0f};
    case LaneRole::CENTER:
      return {1.0f, 1.0f, 0.0f, 1.0f};
    case LaneRole::RIGHT_EGO:
      return {1.0f, 0.5f, 0.0f, 1.0f};
    case LaneRole::RIGHT_ADJACENT:
      return {1.0f, 0.2f, 0.6f, 1.0f};
    default:
      return {1.0f, 1.0f, 1.0f, 1.0f};
  }
}

}  // namespace

visualization_msgs::msg::MarkerArray lanesToMarkers(
  const std::vector<LaneHypothesis> & lanes,
  const std::string & frame_id,
  const rclcpp::Time & stamp)
{
  visualization_msgs::msg::MarkerArray array;
  for (const auto & lane : lanes) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = "lanes";
    marker.id = static_cast<int>(lane.lane_id);
    marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = 0.08;
    const auto rgba = roleColor(lane.role);
    marker.color.r = rgba[0];
    marker.color.g = rgba[1];
    marker.color.b = rgba[2];
    marker.color.a = rgba[3];
    marker.pose.orientation.w = 1.0;

    for (const auto & pt : lane.polyline) {
      geometry_msgs::msg::Point p;
      p.x = pt.x() * 0.05;
      p.y = pt.y() * 0.05;
      p.z = 0.0;
      marker.points.push_back(p);
    }
    array.markers.push_back(marker);
  }
  return array;
}

visualization_msgs::msg::MarkerArray mergesToMarkers(
  const std::vector<MergeEvent> & merges,
  const std::string & frame_id,
  const rclcpp::Time & stamp)
{
  visualization_msgs::msg::MarkerArray array;
  int id = 0;
  for (const auto & merge : merges) {
    visualization_msgs::msg::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = "merge_points";
    marker.id = id++;
    marker.type = visualization_msgs::msg::Marker::SPHERE;
    marker.action = visualization_msgs::msg::Marker::ADD;
    marker.scale.x = marker.scale.y = marker.scale.z = 0.4;
    marker.color.r = merge.type == MergeTopologyType::MERGE ? 1.0f : 0.2f;
    marker.color.g = merge.type == MergeTopologyType::DIVERGE ? 1.0f : 0.2f;
    marker.color.b = 0.2f;
    marker.color.a = 1.0f;
    marker.pose.position.x = merge.merge_point.x() * 0.05;
    marker.pose.position.y = merge.merge_point.y() * 0.05;
    marker.pose.position.z = 0.0;
    marker.pose.orientation.w = 1.0;
    array.markers.push_back(marker);
  }
  return array;
}

cv::Mat drawOverlay(
  const cv::Mat & bev_bgr,
  const std::vector<LaneHypothesis> & lanes,
  const std::vector<MergeEvent> & merges)
{
  cv::Mat overlay = bev_bgr.clone();
  for (const auto & lane : lanes) {
    const auto rgba = roleColor(lane.role);
    const cv::Scalar color(
      rgba[2] * 255, rgba[1] * 255, rgba[0] * 255);
    for (size_t i = 1; i < lane.polyline.size(); ++i) {
      const cv::Point p0(
        static_cast<int>(lane.polyline[i - 1].x()),
        static_cast<int>(lane.polyline[i - 1].y()));
      const cv::Point p1(
        static_cast<int>(lane.polyline[i].x()),
        static_cast<int>(lane.polyline[i].y()));
      cv::line(overlay, p0, p1, color, 2);
    }
  }
  for (const auto & merge : merges) {
    const cv::Point c(
      static_cast<int>(merge.merge_point.x()),
      static_cast<int>(merge.merge_point.y()));
    cv::circle(overlay, c, 8, cv::Scalar(0, 0, 255), -1);
  }
  return overlay;
}

}  // namespace lie_lane_detection
