#pragma once

#include <string>

#include <sensor_msgs/msg/camera_info.hpp>
#include <opencv2/core.hpp>

#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

struct LineLaneDetectionResult
{
  std::vector<LaneHypothesis> lanes;
  std::vector<MergeEvent> merges;
  DebugImages debug;
  cv::Mat H_img2bev;
  size_t line_segment_count{0};
  size_t edge_point_count{0};
  double elapsed_ms{0.0};
  double line_hough_ms{0.0};
  double lie_vote_ms{0.0};
};

class LineLaneDetectionPipeline
{
public:
  explicit LineLaneDetectionPipeline(const PipelineParams & params);

  void updateParams(const PipelineParams & params);

  LineLaneDetectionResult detect(
    const cv::Mat & image_bgr,
    const sensor_msgs::msg::CameraInfo * camera_info = nullptr);

private:
  PipelineParams params_;
};

}  // namespace lie_lane_detection
