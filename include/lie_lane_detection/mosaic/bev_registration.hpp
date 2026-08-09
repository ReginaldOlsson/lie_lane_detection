#pragma once

#include "lie_lane_detection/mosaic/bev_orb_matcher.hpp"

#include <opencv2/core.hpp>

#include <string>

namespace lie_lane_detection
{

enum class BevRegistrationMethod { ECC, ORB, ECC_THEN_ORB, ORB_THEN_ECC };

struct BevRegistrationParams
{
  BevRegistrationMethod method{BevRegistrationMethod::ORB};
  int ecc_max_iterations{80};
  double ecc_epsilon{1e-5};
  double min_ecc_correlation{0.25};
  /// Legacy alias for orb_uniform_cap.
  int orb_max_features{1500};
  double orb_match_ratio{0.75};
  int orb_min_inliers{10};
  double orb_ransac_threshold{3.0};
  /// Monocular-VO style ORB extraction / matching.
  int orb_extract_count{8000};
  int orb_uniform_cap{1500};
  int orb_grid_cell_px{16};
  int orb_max_per_cell{8};
  double orb_scale_factor{1.2};
  int orb_nlevels{4};
  int orb_fast_threshold{20};
  BevOrbMatchMethod orb_match_method{BevOrbMatchMethod::RADIUS_BF};
  double orb_xiang_gao_ratio{2.0};
  double orb_lowe_ratio{0.75};
  int orb_radius_match_px{100};
  uchar mask_gray_threshold{25};
  double mask_bottom_exclude_ratio{0.12};
  double max_step_translation_px{200.0};
  double max_step_rotation_rad{0.15};
  int ecc_gaussian_blur_size{3};
  bool use_coarse_translation_init{true};
  int coarse_max_dy_px{120};
  int coarse_dy_step_px{2};
};

struct BevRegistrationResult
{
  bool valid{false};
  cv::Mat relative_affine_2x3;  // maps current -> previous (findTransformECC convention)
  cv::Mat relative_transform_3x3;
  double correlation{0.0};
  double dx_px{0.0};
  double dy_px{0.0};
  double yaw_rad{0.0};
  int inlier_count{0};
  std::string method_used;
};

struct BevAlignmentDebug
{
  cv::Mat prev_bgr;
  cv::Mat curr_bgr;
  cv::Mat curr_aligned;
  cv::Mat blend;
  cv::Mat abs_diff;
  cv::Mat side_by_side;
  cv::Mat valid_mask;
};

cv::Mat affine2x3ToHomography3x3(const cv::Mat & affine_2x3);

cv::Mat homography3x3ToAffine2x3(const cv::Mat & transform_3x3);

void fillMotionFromAffine(BevRegistrationResult & result);

/// Binary mask of road pixels (excludes black borders and bottom hood band).
cv::Mat buildBevRoadMask(const cv::Mat & gray, const BevRegistrationParams & params);

/// Estimate rigid Euclidean motion between consecutive orthographic BEV frames.
BevRegistrationResult estimateBevFrameMotion(
  const cv::Mat & prev_bgr, const cv::Mat & curr_bgr,
  const BevRegistrationParams & params = BevRegistrationParams{});

/// Warp current frame into previous frame coordinates.
cv::Mat alignCurrentToPrevious(
  const cv::Mat & prev_bgr, const cv::Mat & curr_bgr, const cv::Mat & relative_affine_2x3);

/// Build blend / diff / side-by-side debug images for a pair.
BevAlignmentDebug makeBevAlignmentDebug(
  const cv::Mat & prev_bgr, const cv::Mat & curr_bgr, const BevRegistrationResult & reg,
  const BevRegistrationParams & params = BevRegistrationParams{});

}  // namespace lie_lane_detection
