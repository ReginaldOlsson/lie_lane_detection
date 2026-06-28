#pragma once

#include <string>

#include <opencv2/core.hpp>

namespace lie_lane_detection
{

enum class BevRegistrationMethod
{
  ECC,
  ORB,
  ECC_THEN_ORB
};

struct BevRegistrationParams
{
  BevRegistrationMethod method{BevRegistrationMethod::ECC_THEN_ORB};
  int ecc_max_iterations{50};
  double ecc_epsilon{1e-5};
  double min_ecc_correlation{0.35};
  int orb_max_features{1200};
  double orb_match_ratio{0.75};
  int orb_min_inliers{12};
  double orb_ransac_threshold{3.0};
  uchar mask_gray_threshold{25};
};

struct BevRegistrationResult
{
  bool valid{false};
  cv::Mat relative_transform_3x3;  // maps current frame coords -> previous frame coords
  double correlation{0.0};
  std::string method_used;
};

/// Estimate rigid Euclidean motion between consecutive orthographic BEV frames.
BevRegistrationResult estimateBevFrameMotion(
  const cv::Mat & prev_gray,
  const cv::Mat & curr_gray,
  const BevRegistrationParams & params = BevRegistrationParams{});

cv::Mat affine2x3ToHomography3x3(const cv::Mat & affine_2x3);

cv::Mat homography3x3ToAffine2x3(const cv::Mat & transform_3x3);

}  // namespace lie_lane_detection
