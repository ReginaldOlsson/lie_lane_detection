#pragma once

#include "lie_lane_detection/core/types.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>

#include <string>

namespace lie_lane_detection
{

/// Lightweight ONNX segmenter for BEV road / lane / snow separation.
///
/// Classes (channel order):
///   0 = ignore  (sky, vehicles, curbs, off-road)
///   1 = road    (drivable asphalt/concrete)
///   2 = lane    (lane paint / markings on road)
///   3 = snow    (snow, slush, heavy glare clutter)
class RoadFeatureSegmenter
{
public:
  struct Result
  {
    cv::Mat ignore_prob;
    cv::Mat road_prob;
    cv::Mat lane_prob;
    cv::Mat snow_prob;
    /// 255 where edges/lanes are allowed (road ∪ lane, minus high snow).
    cv::Mat drivable_mask;
    /// BGR debug: green=road, cyan=lane, magenta=snow reject.
    cv::Mat debug_bgr;
    double infer_ms{0.0};
  };

  bool load(const std::string & onnx_path);
  bool isReady() const { return ready_; }

  void setInputSize(int width, int height);
  void updateParams(const PipelineParams & params);

  /// Run on a BEV image (any size). Prob maps are resized to input size.
  Result infer(const cv::Mat & bev_bgr);

private:
  cv::dnn::Net net_;
  bool ready_{false};
  int input_w_{256};
  int input_h_{128};
  double road_threshold_{0.45};
  double lane_threshold_{0.35};
  double snow_reject_threshold_{0.55};
};

/// Apply drivable mask to a BEV image used for edge extraction.
void applyRoadFeatureMask(cv::Mat & bev_bgr, const cv::Mat & drivable_mask);

/// Mask binary/thin edge image with the segmenter drivable mask.
cv::Mat maskEdgesWithRoadFeatures(const cv::Mat & edges_gray, const cv::Mat & drivable_mask);

}  // namespace lie_lane_detection
