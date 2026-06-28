#pragma once

#include <opencv2/core.hpp>

#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

/// Per-frame ego-motion estimate from road-region optical flow (no CAN).
struct EgoMotionEstimate
{
  double delta_image_x{0.0};
  double delta_image_y{0.0};
  double delta_yaw_rad{0.0};
  double delta_bev_x{0.0};
  double confidence{0.0};
  bool valid{false};
};

struct EgoMotionEstimatorParams
{
  double road_y_min_ratio{0.30};
  int grid_step_px{24};
  double max_flow_px{40.0};
  double bev_lateral_scale{1.0};
  double integral_decay{0.92};
  double max_integral_image_x_ratio{0.12};
};

/// Estimates image-plane road motion between consecutive frontal frames.
class EgoMotionEstimator
{
public:
  explicit EgoMotionEstimator(EgoMotionEstimatorParams params = EgoMotionEstimatorParams{});

  void reset();
  void setParams(const EgoMotionEstimatorParams & params) {params_ = params;}

  /// Update with a new BGR frame; returns motion since the previous frame.
  EgoMotionEstimate update(const cv::Mat & image_bgr, double dt = 0.0);

  /// Integrated lateral image shift (decayed) for IPM ROI nudging.
  double integratedImageShiftX() const {return integrated_image_x_;}

  const EgoMotionEstimate & lastEstimate() const {return last_;}

  /// Shift ipm_src_points x-coordinates by integrated ego motion.
  void applyIntegratedShiftToIpmRoi(PipelineParams & params, int cols, int rows) const;

  /// Map per-frame image lateral motion to BEV lateral delta.
  static double imageDeltaToBevLateral(
    double delta_image_x,
    const PipelineParams & params,
    int bev_cols,
    double scale = 1.0);

private:
  static double robustMedian(std::vector<double> values);

  EgoMotionEstimatorParams params_;
  cv::Mat prev_gray_;
  EgoMotionEstimate last_;
  double integrated_image_x_{0.0};
};

}  // namespace lie_lane_detection
