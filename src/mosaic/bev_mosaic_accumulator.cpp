#include "lie_lane_detection/mosaic/bev_mosaic_accumulator.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

namespace lie_lane_detection
{
namespace
{

cv::Mat identity3x3()
{
  return cv::Mat::eye(3, 3, CV_64F);
}

}  // namespace

BevMosaicAccumulator::BevMosaicAccumulator(BevMosaicParams params)
: params_(std::move(params))
{
  reset();
}

void BevMosaicAccumulator::reset()
{
  frames_.clear();
  canvas_.release();
  weight_map_.release();
  last_aligned_.release();
  last_relative_3x3_ = identity3x3();
  canvas_size_ = {};
  initialized_ = false;
  have_last_relative_ = false;
}

const cv::Mat & BevMosaicAccumulator::previousFrame() const
{
  static const cv::Mat empty;
  if (frames_.empty()) {
    return empty;
  }
  return frames_.back().image;
}

void BevMosaicAccumulator::rebuildCanvas()
{
  if (frames_.empty() || canvas_size_.width <= 0 || canvas_size_.height <= 0) {
    return;
  }

  canvas_ = cv::Mat(canvas_size_, CV_8UC3, cv::Scalar(0, 0, 0));
  weight_map_ = cv::Mat(canvas_size_, CV_32F, cv::Scalar(0.f));

  for (const auto & frame : frames_) {
    const cv::Mat pose_2x3 = homography3x3ToAffine2x3(frame.pose_3x3);
    cv::Mat warped_frame;
    cv::warpAffine(frame.image, warped_frame, pose_2x3, canvas_.size(), cv::INTER_LINEAR);

    cv::Mat mask(frame.image.rows, frame.image.cols, CV_8U, cv::Scalar(255));
    cv::Mat warped_mask;
    cv::warpAffine(mask, warped_mask, pose_2x3, canvas_.size(), cv::INTER_NEAREST);

    cv::Mat gray;
    if (frame.image.channels() == 1) {
      gray = frame.image;
    } else {
      cv::cvtColor(frame.image, gray, cv::COLOR_BGR2GRAY);
    }
    cv::Mat valid_pixels = gray > params_.registration.mask_gray_threshold;
    cv::Mat valid_mask;
    cv::warpAffine(valid_pixels, valid_mask, pose_2x3, canvas_.size(), cv::INTER_NEAREST);
    cv::bitwise_and(warped_mask, valid_mask, warped_mask);

    for (int y = 0; y < canvas_.rows; ++y) {
      for (int x = 0; x < canvas_.cols; ++x) {
        if (warped_mask.at<uchar>(y, x) == 0) {
          continue;
        }
        const float w_new = 1.f;
        const float w_old = weight_map_.at<float>(y, x);
        const float w_sum = w_old + w_new;
        cv::Vec3b & dst = canvas_.at<cv::Vec3b>(y, x);
        const cv::Vec3b src = warped_frame.at<cv::Vec3b>(y, x);
        dst[0] = static_cast<uchar>((dst[0] * w_old + src[0] * w_new) / w_sum);
        dst[1] = static_cast<uchar>((dst[1] * w_old + src[1] * w_new) / w_sum);
        dst[2] = static_cast<uchar>((dst[2] * w_old + src[2] * w_new) / w_sum);
        weight_map_.at<float>(y, x) = w_sum;
      }
    }
  }
}

BevMosaicFrameResult BevMosaicAccumulator::accumulate(const cv::Mat & current_bgr)
{
  BevMosaicFrameResult output;
  if (current_bgr.empty()) {
    return output;
  }

  const int max_frames = std::max(1, params_.max_stored_frames);

  if (!initialized_) {
    canvas_size_ = current_bgr.size();
    frames_.push_back(StoredFrame{current_bgr.clone(), identity3x3()});
    rebuildCanvas();
    last_aligned_ = current_bgr.clone();
    output.motion.valid = true;
    output.motion.method_used = "init";
    output.global_affine_2x3 = homography3x3ToAffine2x3(identity3x3());
    output.stored_frames = static_cast<int>(frames_.size());
    initialized_ = true;
    return output;
  }

  const cv::Mat & prev_bgr = frames_.back().image;
  BevRegistrationResult motion = estimateBevFrameMotion(prev_bgr, current_bgr, params_.registration);

  cv::Mat relative_3x3 = identity3x3();
  if (motion.valid) {
    relative_3x3 = motion.relative_transform_3x3.clone();
    have_last_relative_ = true;
    last_relative_3x3_ = relative_3x3.clone();
  } else if (params_.use_constant_velocity_fallback && have_last_relative_) {
    relative_3x3 = last_relative_3x3_.clone();
    motion.relative_affine_2x3 = homography3x3ToAffine2x3(relative_3x3);
    fillMotionFromAffine(motion);
    motion.valid = true;
    motion.method_used = "fallback_cv";
    output.used_fallback = true;
  } else {
    motion.relative_affine_2x3 = cv::Mat::eye(2, 3, CV_64F);
    fillMotionFromAffine(motion);
    motion.valid = false;
    motion.method_used = "identity_fail";
    output.used_fallback = true;
  }

  last_aligned_ = alignCurrentToPrevious(prev_bgr, current_bgr, motion.relative_affine_2x3);
  const BevAlignmentDebug pair_debug =
    makeBevAlignmentDebug(prev_bgr, current_bgr, motion, params_.registration);
  output.blend_with_previous = pair_debug.blend;
  output.abs_diff_with_previous = pair_debug.abs_diff;

  for (auto & stored : frames_) {
    stored.pose_3x3 = relative_3x3 * stored.pose_3x3;
  }
  frames_.push_back(StoredFrame{current_bgr.clone(), identity3x3()});
  while (static_cast<int>(frames_.size()) > max_frames) {
    frames_.pop_front();
  }

  rebuildCanvas();

  output.motion = motion;
  output.global_affine_2x3 = homography3x3ToAffine2x3(identity3x3());
  output.stored_frames = static_cast<int>(frames_.size());
  return output;
}

}  // namespace lie_lane_detection
