#pragma once

#include "lie_lane_detection/core/types.hpp"

#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>

#include <string>
#include <vector>

namespace lie_lane_detection
{

struct IntensityStripParams
{
  int num_horizontal_strips{12};
  int max_peaks_per_strip{4};
  /// Column-projection peak must exceed this fraction of the strip maximum.
  double min_peak_height_ratio{0.32};
  int min_peak_separation_px{42};
  int vertical_strip_half_width_px{52};
  int projection_smooth_ksize{9};
  int min_road_gray{20};
  double bev_bottom_exclude_px{200.0};
  bool dedupe_vertical_rois{true};
  double min_lane_merge_px{38.0};
};

struct VerticalStripRoi
{
  int strip_index{0};
  int peak_index{0};
  cv::Rect roi;
  double peak_x{0.0};
  double peak_value{0.0};
};

struct IntensityStripDetectionResult
{
  std::vector<VerticalStripRoi> vertical_strips;
  std::vector<LaneHypothesis> lanes;
  std::vector<MergeEvent> merges;
  cv::Mat overlay;
  cv::Mat projection_debug;
  int horizontal_strip_count{0};
  double elapsed_ms{0.0};
};

IntensityStripParams loadIntensityStripParams(rclcpp::Node & node);

/// Split BEV into horizontal bands, find column-intensity peaks, detect lanes per vertical strip.
IntensityStripDetectionResult detectLanesIntensityStrips(
  const cv::Mat & bev_bgr, const PipelineParams & detect_params,
  const IntensityStripParams & strip_params);

/// Draw horizontal band cuts, vertical strip boxes, lane overlays, and strip counts.
cv::Mat drawIntensityStripOverlay(
  const cv::Mat & bev_bgr, const IntensityStripDetectionResult & result,
  const IntensityStripParams & strip_params);

/// Project BEV strip cuts + lane curves onto the original camera image.
cv::Mat drawFrontalIntensityStripOverlay(
  const cv::Mat & frontal_bgr, const IntensityStripDetectionResult & result,
  const IntensityStripParams & strip_params, const cv::Mat & H_img2bev);

}  // namespace lie_lane_detection
