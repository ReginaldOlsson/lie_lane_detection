#pragma once

#include <opencv2/core.hpp>

#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

struct VanishingPointEstimate
{
  double x{0.0};
  double y{0.0};
  double confidence{0.0};
  bool valid{false};
};

/// Result of automatic homography estimation from a frontal image.
struct FrontalHomographyResult
{
  bool valid{false};
  bool used_fallback_roi{false};
  VanishingPointEstimate vanishing_point;
  PipelineParams params;
  cv::Mat H_img2bev;
  cv::Mat bev;
  cv::Mat debug_roi;
};

/// Estimate horizon vanishing point from lane-like Hough segments.
VanishingPointEstimate estimateVanishingPoint(
  const cv::Mat & image_bgr,
  const PipelineParams & params = PipelineParams{});

/// Build ipm_src/dst trapezoid from VP (apex) and road bottom corners.
bool configureAutoIpmRoi(
  PipelineParams & params,
  int cols,
  int rows,
  double vp_x,
  double vp_y);

/// Estimate VP → ipm_src_points → 3×3 homography → BEV warp.
FrontalHomographyResult estimateFrontalHomography(
  const cv::Mat & image_bgr,
  PipelineParams params = PipelineParams{});

/// Warp forward camera → pseudo-BEV using auto-estimated VP. Falls back to highway ROI.
cv::Mat warpFrontalAutoIpm(
  const cv::Mat & image_bgr,
  PipelineParams & params,
  VanishingPointEstimate * vp_out = nullptr,
  cv::Mat * debug_viz = nullptr);

}  // namespace lie_lane_detection
