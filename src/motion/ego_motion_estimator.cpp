#include "lie_lane_detection/motion/ego_motion_estimator.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <cmath>
#include <vector>

namespace lie_lane_detection
{

EgoMotionEstimator::EgoMotionEstimator(EgoMotionEstimatorParams params) : params_(std::move(params))
{
}

void EgoMotionEstimator::reset()
{
  prev_gray_.release();
  last_ = EgoMotionEstimate{};
  integrated_image_x_ = 0.0;
}

double EgoMotionEstimator::robustMedian(std::vector<double> values)
{
  if (values.empty()) {
    return 0.0;
  }
  const size_t mid = values.size() / 2;
  std::nth_element(values.begin(), values.begin() + static_cast<long>(mid), values.end());
  return values[mid];
}

double EgoMotionEstimator::imageDeltaToBevLateral(
  double delta_image_x, const PipelineParams & params, int bev_cols, double scale)
{
  if (params.ipm_src_points.size() < 4 || bev_cols <= 0) {
    return 0.0;
  }
  const double bl_x = params.ipm_src_points[0];
  const double br_x = params.ipm_src_points[2];
  const double span = std::max(30.0, br_x - bl_x);
  // Road texture moves opposite to ego lateral motion in image.
  return -delta_image_x * (static_cast<double>(bev_cols) / span) * scale;
}

EgoMotionEstimate EgoMotionEstimator::update(const cv::Mat & image_bgr, double /*dt*/)
{
  EgoMotionEstimate out;
  if (image_bgr.empty()) {
    last_ = out;
    return out;
  }

  cv::Mat gray;
  if (image_bgr.channels() == 3) {
    cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image_bgr;
  }

  if (prev_gray_.empty() || prev_gray_.size() != gray.size()) {
    prev_gray_ = gray.clone();
    last_ = out;
    return out;
  }

  const int w = gray.cols;
  const int h = gray.rows;
  const int y_min = static_cast<int>(h * params_.road_y_min_ratio);

  std::vector<cv::Point2f> pts0;
  pts0.reserve(400);
  for (int y = y_min; y < h - 8; y += params_.grid_step_px) {
    for (int x = static_cast<int>(w * 0.08); x < static_cast<int>(w * 0.92);
         x += params_.grid_step_px) {
      pts0.emplace_back(static_cast<float>(x), static_cast<float>(y));
    }
  }

  std::vector<cv::Point2f> pts1;
  std::vector<uchar> status;
  std::vector<float> err;
  cv::calcOpticalFlowPyrLK(prev_gray_, gray, pts0, pts1, status, err);

  std::vector<double> dx_vals;
  std::vector<double> dy_vals;
  std::vector<cv::Point2f> valid0;
  std::vector<cv::Point2f> valid1;
  dx_vals.reserve(pts0.size());
  dy_vals.reserve(pts0.size());

  for (size_t i = 0; i < pts0.size(); ++i) {
    if (!status[i]) {
      continue;
    }
    const double dx = pts1[i].x - pts0[i].x;
    const double dy = pts1[i].y - pts0[i].y;
    if (std::hypot(dx, dy) > params_.max_flow_px) {
      continue;
    }
    dx_vals.push_back(dx);
    dy_vals.push_back(dy);
    valid0.push_back(pts0[i]);
    valid1.push_back(pts1[i]);
  }

  prev_gray_ = gray.clone();

  if (dx_vals.size() < 12) {
    last_ = out;
    return out;
  }

  out.delta_image_x = robustMedian(dx_vals);
  out.delta_image_y = robustMedian(dy_vals);
  out.confidence =
    static_cast<double>(dx_vals.size()) / static_cast<double>(std::max<size_t>(1, pts0.size()));

  if (valid0.size() >= 6) {
    cv::Mat affine =
      cv::estimateAffinePartial2D(valid0, valid1, cv::noArray(), cv::RANSAC, 3.0, 2000, 0.99);
    if (!affine.empty()) {
      out.delta_yaw_rad = std::atan2(affine.at<double>(1, 0), affine.at<double>(0, 0));
    }
  }

  out.valid = out.confidence >= 0.25 && std::abs(out.delta_image_x) < params_.max_flow_px;

  if (out.valid) {
    integrated_image_x_ = params_.integral_decay * integrated_image_x_ + out.delta_image_x;
    const double max_integral = w * params_.max_integral_image_x_ratio;
    integrated_image_x_ = std::clamp(integrated_image_x_, -max_integral, max_integral);
  }

  last_ = out;
  return out;
}

void EgoMotionEstimator::applyIntegratedShiftToIpmRoi(
  PipelineParams & params, int cols, int rows) const
{
  if (params.ipm_src_points.size() < 8 || std::abs(integrated_image_x_) < 0.5) {
    return;
  }
  const double margin_x = cols * 0.04;
  const double margin_y = rows * 0.02;
  for (size_t i = 0; i < params.ipm_src_points.size(); i += 2) {
    double & x = params.ipm_src_points[i];
    double & y = params.ipm_src_points[i + 1];
    x = std::clamp(x + integrated_image_x_, margin_x, cols - margin_x);
    y = std::clamp(y, margin_y, rows - margin_y);
  }
}

}  // namespace lie_lane_detection
