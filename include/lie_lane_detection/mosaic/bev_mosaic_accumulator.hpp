#pragma once

#include <deque>
#include <opencv2/core.hpp>

#include "lie_lane_detection/mosaic/bev_registration.hpp"

namespace lie_lane_detection
{

struct BevMosaicParams
{
  BevRegistrationParams registration;
  int max_stored_frames{3};
  bool use_constant_velocity_fallback{true};
};

struct BevMosaicFrameResult
{
  BevRegistrationResult motion;
  cv::Mat global_affine_2x3;  // maps current frame -> fixed canvas
  cv::Mat blend_with_previous;
  cv::Mat abs_diff_with_previous;
  bool used_fallback{false};
  int stored_frames{0};
};

/// Fixed-size temporal BEV mosaic: register consecutive frames, keep last N only.
class BevMosaicAccumulator
{
public:
  explicit BevMosaicAccumulator(BevMosaicParams params = BevMosaicParams{});

  void setParams(const BevMosaicParams & params) {params_ = params;}
  void reset();

  BevMosaicFrameResult accumulate(const cv::Mat & current_bgr);

  const cv::Mat & canvas() const {return canvas_;}
  const cv::Mat & previousFrame() const;
  const cv::Mat & lastAlignedFrame() const {return last_aligned_;}
  bool initialized() const {return initialized_;}
  int storedFrameCount() const {return static_cast<int>(frames_.size());}

private:
  struct StoredFrame
  {
    cv::Mat image;
    cv::Mat pose_3x3;
  };

  void rebuildCanvas();

  BevMosaicParams params_;
  std::deque<StoredFrame> frames_;
  cv::Mat canvas_;
  cv::Mat weight_map_;
  cv::Mat last_aligned_;
  cv::Mat last_relative_3x3_;
  cv::Size canvas_size_;
  bool initialized_{false};
  bool have_last_relative_{false};
};

}  // namespace lie_lane_detection
