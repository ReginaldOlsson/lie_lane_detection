#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

class EdgeExtractor
{
public:
  explicit EdgeExtractor(const PipelineParams & params);

  std::vector<EdgePoint> extract(const cv::Mat & bev_bgr, cv::Mat * debug_edges = nullptr);

  /// Otsu binarization on road pixels only; non-road stays 0. Returns Otsu T or -1 on failure.
  static double otsuThresholdMasked(
    const cv::Mat & gray,
    const cv::Mat & road_mask,
    cv::Mat & binary_out);

private:
  PipelineParams params_;
  cv::Mat applySteerableBank(const cv::Mat & gray, cv::Mat * orientation = nullptr) const;
  void thinBinaryEdges(cv::Mat & edges) const;
  void hysteresisThreshold(cv::Mat & magnitude, cv::Mat & orientation) const;
};

}  // namespace lie_lane_detection
