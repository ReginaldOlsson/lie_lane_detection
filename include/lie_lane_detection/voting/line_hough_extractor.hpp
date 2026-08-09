#pragma once

#include "lie_lane_detection/core/types.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace lie_lane_detection
{

class LineHoughExtractor
{
public:
  explicit LineHoughExtractor(const PipelineParams & params);

  std::vector<LineSegment> extract(const cv::Mat & edge_image) const;

  /// Draw detected segments on a BGR image for debug.
  static cv::Mat drawSegments(const cv::Mat & bev_bgr, const std::vector<LineSegment> & lines);

private:
  PipelineParams params_;
};

}  // namespace lie_lane_detection
