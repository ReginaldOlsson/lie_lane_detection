#pragma once

#include <opencv2/core.hpp>

#include "lie_lane_detection/core/types.hpp"
#include "lie_lane_detection/motion/ego_motion_estimator.hpp"

namespace lie_lane_detection
{

struct VanishingPointEstimate
{
  double x{0.0};
  double y{0.0};
  double confidence{0.0};
  bool valid{false};
  bool rejected_as_outlier{false};
  bool used_temporal_prior{false};
};

/// Temporal filter for a fixed-mount camera: VP should move slowly.
struct VanishingPointTrackerParams
{
  double process_noise_px{1.5};
  double base_measurement_noise_px{18.0};
  double max_jump_px{30.0};
  double max_jump_y_px{15.0};
  int min_init_frames{2};
  double min_confidence_ratio{0.35};
};

class VanishingPointTracker
{
public:
  explicit VanishingPointTracker(VanishingPointTrackerParams params = VanishingPointTrackerParams{});

  void reset();
  void setParams(const VanishingPointTrackerParams & params) {params_ = params;}

  /// Fuse a per-frame VP estimate; returns the VP to use for IPM.
  VanishingPointEstimate update(const VanishingPointEstimate & measurement);

  const VanishingPointEstimate & filtered() const {return filtered_;}
  bool initialized() const {return initialized_;}

private:
  bool isOutlier(const VanishingPointEstimate & measurement) const;

  VanishingPointTrackerParams params_;
  VanishingPointEstimate filtered_;
  double p_x_{100.0};
  double p_y_{100.0};
  int init_count_{0};
  bool initialized_{false};
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
/// Pass \p vp_tracker to apply fixed-camera temporal smoothing across frames.
FrontalHomographyResult estimateFrontalHomography(
  const cv::Mat & image_bgr,
  PipelineParams params = PipelineParams{},
  VanishingPointTracker * vp_tracker = nullptr,
  EgoMotionEstimator * ego_motion = nullptr);

/// Warp forward camera → pseudo-BEV using auto-estimated VP. Falls back to highway ROI.
cv::Mat warpFrontalAutoIpm(
  const cv::Mat & image_bgr,
  PipelineParams & params,
  VanishingPointEstimate * vp_out = nullptr,
  cv::Mat * debug_viz = nullptr,
  VanishingPointTracker * vp_tracker = nullptr);

}  // namespace lie_lane_detection
