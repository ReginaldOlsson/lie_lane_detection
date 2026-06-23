#pragma once

#include <opencv2/core.hpp>

#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

struct BevDetectionResult
{
  std::vector<LaneHypothesis> lanes;
  std::vector<MergeEvent> merges;
  cv::Mat edges;
  cv::Mat hough_slice;
  cv::Mat overlay;
  size_t edge_point_count{0};
  size_t line_segment_count{0};
  double elapsed_ms{0.0};
  double line_hough_ms{0.0};
  double lie_vote_ms{0.0};
};

/// Tune Hough/RANSAC bins for a BEV image in pixel coordinates.
void configureParamsForBev(PipelineParams & params, int cols, int rows);

/// Finer omega/kappa/sigma bins for curved and merge/diverge synthetic scenes.
void enhanceParamsForCurvature(PipelineParams & params);

/// Mask invalid (black) IPM corners and return working BEV copy.
cv::Mat prepareBevImage(const cv::Mat & bev_bgr);

/// Set a highway-style IPM trapezoid scaled to image size (TuSimple/CULane-like FOV).
void setDefaultHighwayIpmRoi(PipelineParams & params, int cols, int rows);

/// Relaxed thresholds for noisy IPM warps from forward-camera datasets.
void configureParamsForPerspectiveIpm(PipelineParams & params);

/// Tune for raw forward camera (no IPM): converging lanes, wider omega/sigma.
void configureParamsForFrontalImage(PipelineParams & params, int cols, int rows);

/// Mask sky band; keep lower road region for frontal-camera detection.
cv::Mat prepareFrontalImage(const cv::Mat & image_bgr);

/// Warp a forward camera image to BEV using params.ipm_src/dst_points.
cv::Mat warpPerspectiveToBev(const cv::Mat & image_bgr, const PipelineParams & params);

/// Full detection on a prepared BEV image (shared by offline tool, generator, ROS pipeline).
BevDetectionResult detectLanesInBev(const cv::Mat & bev_bgr, PipelineParams params);

}  // namespace lie_lane_detection
