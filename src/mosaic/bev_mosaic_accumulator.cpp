#include "lie_lane_detection/mosaic/bev_mosaic_accumulator.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace lie_lane_detection
{
namespace
{

cv::Mat identity2x3()
{
  cv::Mat m = cv::Mat::eye(2, 3, CV_64F);
  return m;
}

cv::Mat identity3x3()
{
  return cv::Mat::eye(3, 3, CV_64F);
}

double rotationFromAffine2x3(const cv::Mat & m)
{
  if (m.empty() || m.rows < 2 || m.cols < 2) {
    return 0.0;
  }
  const double r00 = m.at<double>(0, 0);
  const double r10 = m.at<double>(1, 0);
  return std::atan2(r10, r00);
}

cv::Point2d translationFromAffine2x3(const cv::Mat & m)
{
  if (m.empty() || m.rows < 2 || m.cols < 3) {
    return {0.0, 0.0};
  }
  return {m.at<double>(0, 2), m.at<double>(1, 2)};
}

}  // namespace

BevMosaicAccumulator::BevMosaicAccumulator(BevMosaicParams params)
: params_(std::move(params))
{
  reset();
}

void BevMosaicAccumulator::reset()
{
  canvas_.release();
  weight_map_.release();
  prev_frame_bgr_.release();
  prev_gray_.release();
  last_aligned_.release();
  global_pose_3x3_ = identity3x3();
  last_relative_3x3_ = identity3x3();
  initialized_ = false;
  have_last_relative_ = false;
}

cv::Mat BevMosaicAccumulator::toGray(const cv::Mat & bgr)
{
  if (bgr.empty()) {
    return {};
  }
  if (bgr.channels() == 1) {
    return bgr.clone();
  }
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  return gray;
}

cv::Mat BevMosaicAccumulator::affine2x3To3x3(const cv::Mat & affine_2x3)
{
  cv::Mat out = identity3x3();
  if (affine_2x3.empty()) {
    return out;
  }
  cv::Mat converted;
  affine_2x3.convertTo(converted, CV_64F);
  converted.copyTo(out(cv::Rect(0, 0, 3, 2)));
  return out;
}

cv::Mat BevMosaicAccumulator::affine3x3To2x3(const cv::Mat & affine_3x3)
{
  return affine_3x3(cv::Rect(0, 0, 3, 2)).clone();
}

bool BevMosaicAccumulator::withinStepLimits(
  const cv::Mat & relative_2x3,
  const BevMosaicParams & params)
{
  const cv::Point2d t = translationFromAffine2x3(relative_2x3);
  const double rot = std::abs(rotationFromAffine2x3(relative_2x3));
  return std::hypot(t.x, t.y) <= params.max_step_translation_px &&
         rot <= params.max_step_rotation_rad;
}

BevRegistrationResult BevMosaicAccumulator::registerEcc(
  const cv::Mat & prev_gray,
  const cv::Mat & curr_gray) const
{
  BevRegistrationResult result;
  result.relative_affine_2x3 = identity2x3();
  result.used_ecc = true;

  if (prev_gray.empty() || curr_gray.empty() ||
    prev_gray.size() != curr_gray.size())
  {
    return result;
  }

  cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);
  const cv::TermCriteria criteria(
    cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
    params_.ecc_max_iterations,
    params_.ecc_epsilon);

  try {
    result.correlation = cv::findTransformECC(
      prev_gray, curr_gray, warp_matrix, cv::MOTION_EUCLIDEAN, criteria);
    warp_matrix.convertTo(result.relative_affine_2x3, CV_64F);
    result.valid = result.correlation >= params_.min_ecc_correlation &&
      withinStepLimits(result.relative_affine_2x3, params_);
  } catch (const cv::Exception &) {
    result.valid = false;
  }
  return result;
}

BevRegistrationResult BevMosaicAccumulator::registerFeatures(
  const cv::Mat & prev_gray,
  const cv::Mat & curr_gray) const
{
  BevRegistrationResult result;
  result.relative_affine_2x3 = identity2x3();
  result.used_features = true;

  if (prev_gray.empty() || curr_gray.empty()) {
    return result;
  }

  cv::Ptr<cv::ORB> orb = cv::ORB::create(params_.orb_features);
  std::vector<cv::KeyPoint> kp1;
  std::vector<cv::KeyPoint> kp2;
  cv::Mat desc1;
  cv::Mat desc2;
  orb->detectAndCompute(prev_gray, cv::noArray(), kp1, desc1);
  orb->detectAndCompute(curr_gray, cv::noArray(), kp2, desc2);

  if (desc1.empty() || desc2.empty() || kp1.size() < 8 || kp2.size() < 8) {
    return result;
  }

  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> matches;
  matcher.match(desc1, desc2, matches);
  if (matches.size() < static_cast<size_t>(params_.feature_min_inliers)) {
    return result;
  }

  std::sort(matches.begin(), matches.end(), [](const cv::DMatch & a, const cv::DMatch & b) {
      return a.distance < b.distance;
    });
  const size_t keep = std::max<size_t>(
    static_cast<size_t>(params_.feature_min_inliers),
    static_cast<size_t>(matches.size() * params_.feature_match_ratio));
  matches.resize(std::min(keep, matches.size()));

  std::vector<cv::Point2f> pts1;
  std::vector<cv::Point2f> pts2;
  pts1.reserve(matches.size());
  pts2.reserve(matches.size());
  for (const auto & m : matches) {
    pts1.push_back(kp1[static_cast<size_t>(m.queryIdx)].pt);
    pts2.push_back(kp2[static_cast<size_t>(m.trainIdx)].pt);
  }

  cv::Mat inliers;
  cv::Mat affine = cv::estimateAffinePartial2D(
    pts2, pts1, inliers, cv::RANSAC, 3.0, 2000, 0.99);
  if (affine.empty()) {
    return result;
  }
  affine.convertTo(result.relative_affine_2x3, CV_64F);

  const int inlier_count = inliers.empty() ? 0 : cv::countNonZero(inliers);
  result.correlation = static_cast<double>(inlier_count) /
    static_cast<double>(std::max<size_t>(1, matches.size()));
  result.valid = inlier_count >= params_.feature_min_inliers &&
    withinStepLimits(result.relative_affine_2x3, params_);
  return result;
}

BevRegistrationResult BevMosaicAccumulator::registerFrame(
  const cv::Mat & prev_gray,
  const cv::Mat & curr_gray)
{
  BevRegistrationResult ecc_result;
  BevRegistrationResult feature_result;

  const bool try_ecc = params_.registration_method == BevRegistrationMethod::ECC ||
    params_.registration_method == BevRegistrationMethod::AUTO;
  const bool try_features = params_.registration_method == BevRegistrationMethod::FEATURES ||
    params_.registration_method == BevRegistrationMethod::AUTO;

  if (try_ecc) {
    ecc_result = registerEcc(prev_gray, curr_gray);
    if (ecc_result.valid) {
      return ecc_result;
    }
  }

  if (try_features) {
    feature_result = registerFeatures(prev_gray, curr_gray);
    if (feature_result.valid) {
      return feature_result;
    }
  }

  if (try_ecc && ecc_result.correlation > 0.0) {
    return ecc_result;
  }
  if (try_features) {
    return feature_result;
  }
  return ecc_result;
}

void BevMosaicAccumulator::expandCanvasIfNeeded(const cv::Mat & global_3x3)
{
  if (canvas_.empty() || prev_frame_bgr_.empty()) {
    return;
  }

  std::vector<cv::Point2f> corners = {
    cv::Point2f(0.f, 0.f),
    cv::Point2f(static_cast<float>(prev_frame_bgr_.cols), 0.f),
    cv::Point2f(static_cast<float>(prev_frame_bgr_.cols), static_cast<float>(prev_frame_bgr_.rows)),
    cv::Point2f(0.f, static_cast<float>(prev_frame_bgr_.rows)),
  };
  std::vector<cv::Point2f> warped_corners;
  cv::transform(corners, warped_corners, global_3x3(cv::Rect(0, 0, 3, 2)));

  float min_x = 0.f;
  float min_y = 0.f;
  float max_x = static_cast<float>(canvas_.cols);
  float max_y = static_cast<float>(canvas_.rows);
  for (const auto & p : warped_corners) {
    min_x = std::min(min_x, p.x);
    min_y = std::min(min_y, p.y);
    max_x = std::max(max_x, p.x);
    max_y = std::max(max_y, p.y);
  }

  const int margin = params_.canvas_margin_px;
  const int new_width = static_cast<int>(std::ceil(max_x - min_x)) + 2 * margin;
  const int new_height = static_cast<int>(std::ceil(max_y - min_y)) + 2 * margin;
  if (new_width <= canvas_.cols && new_height <= canvas_.rows && min_x >= 0.f && min_y >= 0.f) {
    return;
  }

  const cv::Mat shift = (cv::Mat_<double>(2, 3) <<
    1.0, 0.0, static_cast<double>(margin - min_x),
    0.0, 1.0, static_cast<double>(margin - min_y));
  const cv::Mat shift_3x3 = affine2x3To3x3(shift);

  cv::Mat new_canvas(new_height, new_width, canvas_.type(), cv::Scalar(0, 0, 0));
  cv::Mat new_weights(new_height, new_width, CV_32F, cv::Scalar(0.f));
  cv::warpAffine(canvas_, new_canvas, shift, new_canvas.size());
  cv::warpAffine(weight_map_, new_weights, shift, new_weights.size());

  canvas_ = new_canvas;
  weight_map_ = new_weights;
  global_pose_3x3_ = shift_3x3 * global_pose_3x3_;
}

void BevMosaicAccumulator::warpOntoCanvas(
  const cv::Mat & frame_bgr,
  const cv::Mat & global_2x3)
{
  cv::Mat warped_frame;
  cv::warpAffine(frame_bgr, warped_frame, global_2x3, canvas_.size(), cv::INTER_LINEAR);

  cv::Mat mask(frame_bgr.rows, frame_bgr.cols, CV_8U, cv::Scalar(255));
  cv::Mat warped_mask;
  cv::warpAffine(mask, warped_mask, global_2x3, canvas_.size(), cv::INTER_NEAREST);

  cv::Mat gray;
  cv::cvtColor(frame_bgr, gray, cv::COLOR_BGR2GRAY);
  cv::Mat valid_pixels = gray > 20;
  cv::Mat valid_mask;
  cv::warpAffine(valid_pixels, valid_mask, global_2x3, canvas_.size(), cv::INTER_NEAREST);
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

BevRegistrationResult BevMosaicAccumulator::accumulate(const cv::Mat & current_bgr)
{
  BevRegistrationResult output;
  if (current_bgr.empty()) {
    return output;
  }

  if (!initialized_) {
    canvas_ = current_bgr.clone();
    weight_map_ = cv::Mat(current_bgr.rows, current_bgr.cols, CV_32F, cv::Scalar(1.f));
    prev_frame_bgr_ = current_bgr.clone();
    prev_gray_ = toGray(current_bgr);
    last_aligned_ = current_bgr.clone();
    global_pose_3x3_ = identity3x3();
    output.relative_affine_2x3 = identity2x3();
    output.global_affine_2x3 = affine3x3To2x3(global_pose_3x3_);
    output.valid = true;
    initialized_ = true;
    return output;
  }

  const cv::Mat curr_gray = toGray(current_bgr);
  BevRegistrationResult reg = registerFrame(prev_gray_, curr_gray);

  cv::Mat relative_3x3 = identity3x3();
  if (reg.valid) {
    relative_3x3 = affine2x3To3x3(reg.relative_affine_2x3);
    have_last_relative_ = true;
    last_relative_3x3_ = relative_3x3.clone();
  } else if (params_.use_constant_velocity_fallback && have_last_relative_) {
    relative_3x3 = last_relative_3x3_.clone();
    reg.used_fallback = true;
    reg.valid = true;
    reg.relative_affine_2x3 = affine3x3To2x3(relative_3x3);
  } else {
    relative_3x3 = identity3x3();
    reg.relative_affine_2x3 = identity2x3();
    reg.valid = true;
    reg.used_fallback = true;
  }

  global_pose_3x3_ = global_pose_3x3_ * relative_3x3;
  expandCanvasIfNeeded(global_pose_3x3_);

  const cv::Mat global_2x3 = affine3x3To2x3(global_pose_3x3_);
  cv::warpAffine(
    current_bgr, last_aligned_, reg.relative_affine_2x3, prev_frame_bgr_.size(),
    cv::INTER_LINEAR + cv::WARP_INVERSE_MAP);
  warpOntoCanvas(current_bgr, global_2x3);

  output = reg;
  output.global_affine_2x3 = global_2x3;
  prev_frame_bgr_ = current_bgr.clone();
  prev_gray_ = curr_gray;
  return output;
}

}  // namespace lie_lane_detection
