#include "lie_lane_detection/mosaic/bev_registration.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace lie_lane_detection
{
namespace
{

cv::Mat toGray(const cv::Mat & bgr)
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

bool withinStepLimits(const cv::Mat & affine_2x3, const BevRegistrationParams & params)
{
  if (affine_2x3.empty()) {
    return false;
  }
  cv::Mat m;
  affine_2x3.convertTo(m, CV_64F);
  const double dx = m.at<double>(0, 2);
  const double dy = m.at<double>(1, 2);
  const double yaw = std::atan2(m.at<double>(1, 0), m.at<double>(0, 0));
  return std::hypot(dx, dy) <= params.max_step_translation_px &&
         std::abs(yaw) <= params.max_step_rotation_rad;
}

cv::Point2d estimateCoarseTranslation(
  const cv::Mat & prev_gray,
  const cv::Mat & curr_gray,
  const cv::Mat & mask,
  const BevRegistrationParams & params)
{
  auto maskedError = [&](double dx, double dy) {
      const cv::Mat shift = (cv::Mat_<double>(2, 3) <<
        1.0, 0.0, dx,
        0.0, 1.0, dy);
      cv::Mat warped;
      cv::warpAffine(curr_gray, warped, shift, prev_gray.size(), cv::INTER_LINEAR);
      double err = 0.0;
      int count = 0;
      for (int y = 0; y < prev_gray.rows; ++y) {
        for (int x = 0; x < prev_gray.cols; ++x) {
          if (mask.at<uchar>(y, x) == 0 || warped.at<uchar>(y, x) <= params.mask_gray_threshold) {
            continue;
          }
          err += std::abs(
            static_cast<int>(prev_gray.at<uchar>(y, x)) -
            static_cast<int>(warped.at<uchar>(y, x)));
          ++count;
        }
      }
      if (count < 500) {
        return std::numeric_limits<double>::max();
      }
      return err / static_cast<double>(count);
    };

  double best_err = maskedError(0.0, 0.0);
  cv::Point2d best(0.0, 0.0);

  for (int dy = -params.coarse_max_dy_px; dy <= params.coarse_max_dy_px;
    dy += params.coarse_dy_step_px)
  {
    for (int dx = -30; dx <= 30; dx += params.coarse_dy_step_px) {
      if (dx == 0 && dy == 0) {
        continue;
      }
      const double err = maskedError(static_cast<double>(dx), static_cast<double>(dy));
      if (err < best_err) {
        best_err = err;
        best = cv::Point2d(dx, dy);
      }
    }
  }
  return best;
}

}  // namespace

cv::Mat affine2x3ToHomography3x3(const cv::Mat & affine_2x3)
{
  cv::Mat out = cv::Mat::eye(3, 3, CV_64F);
  if (affine_2x3.empty()) {
    return out;
  }
  cv::Mat converted;
  affine_2x3.convertTo(converted, CV_64F);
  converted.copyTo(out(cv::Rect(0, 0, 3, 2)));
  return out;
}

cv::Mat homography3x3ToAffine2x3(const cv::Mat & transform_3x3)
{
  return transform_3x3(cv::Rect(0, 0, 3, 2)).clone();
}

void fillMotionFromAffine(BevRegistrationResult & result)
{
  if (result.relative_affine_2x3.empty()) {
    return;
  }
  cv::Mat m;
  result.relative_affine_2x3.convertTo(m, CV_64F);
  result.dx_px = m.at<double>(0, 2);
  result.dy_px = m.at<double>(1, 2);
  result.yaw_rad = std::atan2(m.at<double>(1, 0), m.at<double>(0, 0));
  result.relative_transform_3x3 = affine2x3ToHomography3x3(m);
}

cv::Mat buildBevRoadMask(const cv::Mat & gray, const BevRegistrationParams & params)
{
  cv::Mat mask = gray > params.mask_gray_threshold;
  const int exclude_rows = static_cast<int>(gray.rows * params.mask_bottom_exclude_ratio);
  if (exclude_rows > 0 && exclude_rows < gray.rows) {
    mask.rowRange(gray.rows - exclude_rows, gray.rows).setTo(0);
  }
  return mask;
}

namespace
{

BevRegistrationResult registerEcc(
  const cv::Mat & prev_gray,
  const cv::Mat & curr_gray,
  const cv::Mat & prev_mask,
  const BevRegistrationParams & params)
{
  BevRegistrationResult result;
  result.method_used = "ecc";

  if (prev_gray.empty() || curr_gray.empty() || prev_gray.size() != curr_gray.size()) {
    return result;
  }

  cv::Mat prev_proc = prev_gray.clone();
  cv::Mat curr_proc = curr_gray.clone();
  if (params.ecc_gaussian_blur_size >= 3) {
    const int k = params.ecc_gaussian_blur_size | 1;
    cv::GaussianBlur(prev_proc, prev_proc, cv::Size(k, k), 0.0);
    cv::GaussianBlur(curr_proc, curr_proc, cv::Size(k, k), 0.0);
  }

  cv::Mat warp_matrix = cv::Mat::eye(2, 3, CV_32F);
  if (params.use_coarse_translation_init) {
    const cv::Point2d coarse = estimateCoarseTranslation(prev_gray, curr_gray, prev_mask, params);
    if (std::hypot(coarse.x, coarse.y) > 0.5) {
      warp_matrix.at<float>(0, 2) = static_cast<float>(coarse.x);
      warp_matrix.at<float>(1, 2) = static_cast<float>(coarse.y);
    }
  }
  const cv::TermCriteria criteria(
    cv::TermCriteria::COUNT | cv::TermCriteria::EPS,
    params.ecc_max_iterations,
    params.ecc_epsilon);

  try {
    result.correlation = cv::findTransformECC(
      prev_proc, curr_proc, warp_matrix, cv::MOTION_EUCLIDEAN, criteria, prev_mask);
    warp_matrix.convertTo(result.relative_affine_2x3, CV_64F);
    fillMotionFromAffine(result);
    result.valid = result.correlation >= params.min_ecc_correlation &&
      withinStepLimits(result.relative_affine_2x3, params);
  } catch (const cv::Exception &) {
    result.valid = false;
  }
  return result;
}

BevRegistrationResult registerOrb(
  const cv::Mat & prev_gray,
  const cv::Mat & curr_gray,
  const cv::Mat & prev_mask,
  const BevRegistrationParams & params)
{
  BevRegistrationResult result;
  result.method_used = "orb";

  cv::Ptr<cv::ORB> orb = cv::ORB::create(params.orb_max_features);
  std::vector<cv::KeyPoint> kp_prev;
  std::vector<cv::KeyPoint> kp_curr;
  cv::Mat desc_prev;
  cv::Mat desc_curr;
  orb->detectAndCompute(prev_gray, prev_mask, kp_prev, desc_prev);
  orb->detectAndCompute(curr_gray, prev_mask, kp_curr, desc_curr);

  if (desc_prev.empty() || desc_curr.empty()) {
    return result;
  }

  cv::BFMatcher matcher(cv::NORM_HAMMING, true);
  std::vector<cv::DMatch> matches;
  matcher.match(desc_prev, desc_curr, matches);
  if (matches.size() < static_cast<size_t>(params.orb_min_inliers)) {
    return result;
  }

  std::sort(matches.begin(), matches.end(), [](const cv::DMatch & a, const cv::DMatch & b) {
      return a.distance < b.distance;
    });
  const size_t keep = std::max<size_t>(
    static_cast<size_t>(params.orb_min_inliers),
    static_cast<size_t>(matches.size() * params.orb_match_ratio));
  matches.resize(std::min(keep, matches.size()));

  std::vector<cv::Point2f> pts_prev;
  std::vector<cv::Point2f> pts_curr;
  pts_prev.reserve(matches.size());
  pts_curr.reserve(matches.size());
  for (const auto & m : matches) {
    pts_prev.push_back(kp_prev[static_cast<size_t>(m.queryIdx)].pt);
    pts_curr.push_back(kp_curr[static_cast<size_t>(m.trainIdx)].pt);
  }

  cv::Mat inliers;
  cv::Mat affine = cv::estimateAffinePartial2D(
    pts_curr, pts_prev, inliers, cv::RANSAC, params.orb_ransac_threshold, 2000, 0.99);
  if (affine.empty()) {
    return result;
  }
  affine.convertTo(result.relative_affine_2x3, CV_64F);
  fillMotionFromAffine(result);
  result.inlier_count = inliers.empty() ? 0 : cv::countNonZero(inliers);
  result.correlation = static_cast<double>(result.inlier_count) /
    static_cast<double>(std::max<size_t>(1, matches.size()));
  result.valid = result.inlier_count >= params.orb_min_inliers &&
    withinStepLimits(result.relative_affine_2x3, params);
  return result;
}

}  // namespace

BevRegistrationResult estimateBevFrameMotion(
  const cv::Mat & prev_bgr,
  const cv::Mat & curr_bgr,
  const BevRegistrationParams & params)
{
  BevRegistrationResult out;
  const cv::Mat prev_gray = toGray(prev_bgr);
  const cv::Mat curr_gray = toGray(curr_bgr);
  if (prev_gray.empty() || curr_gray.empty()) {
    return out;
  }

  const cv::Mat mask = buildBevRoadMask(prev_gray, params);

  const bool try_ecc = params.method == BevRegistrationMethod::ECC ||
    params.method == BevRegistrationMethod::ECC_THEN_ORB;
  const bool try_orb = params.method == BevRegistrationMethod::ORB ||
    params.method == BevRegistrationMethod::ECC_THEN_ORB;

  BevRegistrationResult ecc_result;
  BevRegistrationResult orb_result;

  if (try_ecc) {
    ecc_result = registerEcc(prev_gray, curr_gray, mask, params);
    if (ecc_result.valid) {
      return ecc_result;
    }
  }
  if (try_orb) {
    orb_result = registerOrb(prev_gray, curr_gray, mask, params);
    if (orb_result.valid) {
      return orb_result;
    }
  }

  if (try_ecc && ecc_result.correlation > 0.0) {
    return ecc_result;
  }
  if (try_orb) {
    return orb_result;
  }
  return ecc_result;
}

cv::Mat alignCurrentToPrevious(
  const cv::Mat & prev_bgr,
  const cv::Mat & curr_bgr,
  const cv::Mat & relative_affine_2x3)
{
  if (prev_bgr.empty() || curr_bgr.empty() || relative_affine_2x3.empty()) {
    return {};
  }
  cv::Mat aligned;
  cv::warpAffine(curr_bgr, aligned, relative_affine_2x3, prev_bgr.size(), cv::INTER_LINEAR);
  return aligned;
}

BevAlignmentDebug makeBevAlignmentDebug(
  const cv::Mat & prev_bgr,
  const cv::Mat & curr_bgr,
  const BevRegistrationResult & reg,
  const BevRegistrationParams & params)
{
  BevAlignmentDebug debug;
  debug.prev_bgr = prev_bgr.clone();
  debug.curr_bgr = curr_bgr.clone();
  debug.valid_mask = buildBevRoadMask(toGray(prev_bgr), params);

  if (reg.relative_affine_2x3.empty()) {
    return debug;
  }

  debug.curr_aligned = alignCurrentToPrevious(prev_bgr, curr_bgr, reg.relative_affine_2x3);
  if (debug.curr_aligned.empty()) {
    return debug;
  }

  cv::addWeighted(prev_bgr, 0.5, debug.curr_aligned, 0.5, 0.0, debug.blend);
  cv::absdiff(prev_bgr, debug.curr_aligned, debug.abs_diff);

  debug.side_by_side = cv::Mat(prev_bgr.rows, prev_bgr.cols * 2, prev_bgr.type());
  prev_bgr.copyTo(debug.side_by_side(cv::Rect(0, 0, prev_bgr.cols, prev_bgr.rows)));
  debug.curr_aligned.copyTo(
    debug.side_by_side(cv::Rect(prev_bgr.cols, 0, prev_bgr.cols, prev_bgr.rows)));

  return debug;
}

}  // namespace lie_lane_detection
