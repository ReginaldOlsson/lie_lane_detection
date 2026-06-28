#pragma once

#include <opencv2/core.hpp>

namespace lie_lane_detection
{

enum class BevRegistrationMethod
{
  ECC,
  FEATURES,
  AUTO
};

struct BevMosaicParams
{
  int canvas_margin_px{400};
  double min_ecc_correlation{0.35};
  int ecc_max_iterations{50};
  double ecc_epsilon{1e-5};
  int orb_features{1200};
  double feature_match_ratio{0.75};
  int feature_min_inliers{12};
  double max_step_translation_px{80.0};
  double max_step_rotation_rad{0.12};
  bool use_constant_velocity_fallback{true};
  BevRegistrationMethod registration_method{BevRegistrationMethod::AUTO};
};

struct BevRegistrationResult
{
  cv::Mat relative_affine_2x3;  // maps current -> previous (same convention as findTransformECC)
  cv::Mat global_affine_2x3;    // maps current frame -> mosaic canvas
  double correlation{0.0};
  bool valid{false};
  bool used_ecc{false};
  bool used_features{false};
  bool used_fallback{false};
};

/// Temporal BEV orthomosaic: 2D Euclidean registration + global pose accumulation.
class BevMosaicAccumulator
{
public:
  explicit BevMosaicAccumulator(BevMosaicParams params = BevMosaicParams{});

  void setParams(const BevMosaicParams & params) {params_ = params;}
  void reset();

  /// Register \p current_bgr against the previous frame and warp onto the growing canvas.
  BevRegistrationResult accumulate(const cv::Mat & current_bgr);

  const cv::Mat & canvas() const {return canvas_;}
  const cv::Mat & lastAlignedFrame() const {return last_aligned_;}
  bool initialized() const {return initialized_;}

private:
  static cv::Mat toGray(const cv::Mat & bgr);
  static cv::Mat affine2x3To3x3(const cv::Mat & affine_2x3);
  static cv::Mat affine3x3To2x3(const cv::Mat & affine_3x3);
  static bool withinStepLimits(const cv::Mat & relative_2x3, const BevMosaicParams & params);

  BevRegistrationResult registerEcc(const cv::Mat & prev_gray, const cv::Mat & curr_gray) const;
  BevRegistrationResult registerFeatures(const cv::Mat & prev_gray, const cv::Mat & curr_gray) const;
  BevRegistrationResult registerFrame(const cv::Mat & prev_gray, const cv::Mat & curr_gray);
  void expandCanvasIfNeeded(const cv::Mat & global_3x3);
  void warpOntoCanvas(const cv::Mat & frame_bgr, const cv::Mat & global_2x3);

  BevMosaicParams params_;
  cv::Mat canvas_;
  cv::Mat weight_map_;
  cv::Mat prev_frame_bgr_;
  cv::Mat prev_gray_;
  cv::Mat last_aligned_;
  cv::Mat global_pose_3x3_;
  cv::Mat last_relative_3x3_;
  bool initialized_{false};
  bool have_last_relative_{false};
};

}  // namespace lie_lane_detection
