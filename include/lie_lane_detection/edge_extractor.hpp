#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "lie_lane_detection/types.hpp"

namespace lie_lane_detection
{

class EdgeExtractor
{
public:
  explicit EdgeExtractor(const PipelineParams & params);

  std::vector<EdgePoint> extract(const cv::Mat & bev_bgr, cv::Mat * debug_edges = nullptr);

private:
  PipelineParams params_;
  cv::Mat applySteerableBank(const cv::Mat & gray, cv::Mat * orientation = nullptr) const;
  void thinBinaryEdges(cv::Mat & edges) const;
  void hysteresisThreshold(cv::Mat & magnitude, cv::Mat & orientation) const;
};

}  // namespace lie_lane_detection
