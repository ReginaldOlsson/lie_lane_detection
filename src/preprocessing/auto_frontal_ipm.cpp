#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"
#include "lie_lane_detection/motion/ego_motion_estimator.hpp"

namespace lie_lane_detection
{

namespace
{

struct Line2D
{
  double x1{0.0};
  double y1{0.0};
  double x2{0.0};
  double y2{0.0};
  double length{0.0};
};

bool intersectLines(
  const Line2D & a,
  const Line2D & b,
  double & ix,
  double & iy)
{
  const double x1 = a.x1;
  const double y1 = a.y1;
  const double x2 = a.x2;
  const double y2 = a.y2;
  const double x3 = b.x1;
  const double y3 = b.y1;
  const double x4 = b.x2;
  const double y4 = b.y2;

  const double denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (std::abs(denom) < 1e-6) {
    return false;
  }

  ix = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom;
  iy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom;
  return std::isfinite(ix) && std::isfinite(iy);
}

cv::Point2f pointOnRay(double vx, double vy, double bx, double by, double y_target)
{
  if (std::abs(by - vy) < 1e-6) {
    return cv::Point2f(static_cast<float>(bx), static_cast<float>(by));
  }
  const double t = (y_target - vy) / (by - vy);
  return cv::Point2f(
    static_cast<float>(vx + t * (bx - vx)),
    static_cast<float>(y_target));
}

bool refineVotePeakCentroid(
  const std::vector<float> & votes,
  int vote_w,
  int vote_h,
  int peak_x,
  int peak_y,
  double & cx,
  double & cy,
  float & peak_vote,
  float & second_vote)
{
  const int radius = 3;
  double sum_w = 0.0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  peak_vote = votes[static_cast<size_t>(peak_y * vote_w + peak_x)];
  second_vote = 0.0f;

  for (int y = 0; y < vote_h; ++y) {
    for (int x = 0; x < vote_w; ++x) {
      const float v = votes[static_cast<size_t>(y * vote_w + x)];
      if (x == peak_x && y == peak_y) {
        continue;
      }
      if (v > second_vote) {
        second_vote = v;
      }
    }
  }

  for (int dy = -radius; dy <= radius; ++dy) {
    for (int dx = -radius; dx <= radius; ++dx) {
      const int x = peak_x + dx;
      const int y = peak_y + dy;
      if (x < 0 || y < 0 || x >= vote_w || y >= vote_h) {
        continue;
      }
      const float v = votes[static_cast<size_t>(y * vote_w + x)];
      if (v <= 0.0f) {
        continue;
      }
      sum_w += v;
      sum_x += v * x;
      sum_y += v * y;
    }
  }
  if (sum_w <= 0.0) {
    cx = peak_x;
    cy = peak_y;
    return false;
  }
  cx = sum_x / sum_w;
  cy = sum_y / sum_w;
  return true;
}

}  // namespace

VanishingPointTracker::VanishingPointTracker(VanishingPointTrackerParams params)
: params_(std::move(params))
{
}

void VanishingPointTracker::reset()
{
  filtered_ = VanishingPointEstimate{};
  p_x_ = 100.0;
  p_y_ = 100.0;
  init_count_ = 0;
  initialized_ = false;
}

bool VanishingPointTracker::isOutlier(const VanishingPointEstimate & measurement) const
{
  if (!initialized_ || !measurement.valid) {
    return false;
  }
  const double dx = measurement.x - filtered_.x;
  const double dy = measurement.y - filtered_.y;
  if (std::abs(dx) > params_.max_jump_px || std::abs(dy) > params_.max_jump_y_px) {
    return true;
  }
  const double dist = std::hypot(dx, dy);
  return dist > params_.max_jump_px;
}

VanishingPointEstimate VanishingPointTracker::update(const VanishingPointEstimate & measurement)
{
  VanishingPointEstimate output = measurement;
  output.rejected_as_outlier = false;
  output.used_temporal_prior = false;

  if (!measurement.valid) {
    if (initialized_) {
      output = filtered_;
      output.valid = true;
      output.used_temporal_prior = true;
      output.confidence = filtered_.confidence * 0.85;
    }
    return output;
  }

  if (!initialized_) {
    if (!measurement.valid) {
      return output;
    }
    if (init_count_ == 0) {
      filtered_ = measurement;
    } else {
      filtered_.x = 0.5 * (filtered_.x + measurement.x);
      filtered_.y = 0.5 * (filtered_.y + measurement.y);
      filtered_.confidence = std::max(filtered_.confidence, measurement.confidence);
    }
    ++init_count_;
    if (init_count_ >= params_.min_init_frames) {
      initialized_ = true;
    }
    output = filtered_;
    return output;
  }

  if (measurement.confidence < params_.min_confidence_ratio && isOutlier(measurement)) {
    output = filtered_;
    output.valid = true;
    output.rejected_as_outlier = true;
    output.used_temporal_prior = true;
    p_x_ += params_.process_noise_px;
    p_y_ += params_.process_noise_px;
    return output;
  }

  if (isOutlier(measurement)) {
    // Soft update: pull only partially toward a suspicious measurement.
    const double blend = 0.15;
    filtered_.x = (1.0 - blend) * filtered_.x + blend * measurement.x;
    filtered_.y = (1.0 - blend) * filtered_.y + blend * measurement.y;
    filtered_.confidence = std::max(filtered_.confidence, measurement.confidence * 0.5);
    output = filtered_;
    output.used_temporal_prior = true;
    return output;
  }

  const double r = params_.base_measurement_noise_px /
    std::max(0.2, measurement.confidence);
  const double kx = p_x_ / (p_x_ + r);
  const double ky = p_y_ / (p_y_ + r);
  filtered_.x += kx * (measurement.x - filtered_.x);
  filtered_.y += ky * (measurement.y - filtered_.y);
  filtered_.confidence = std::max(
    measurement.confidence,
    filtered_.confidence * 0.95);
  filtered_.valid = true;
  p_x_ = (1.0 - kx) * p_x_ + params_.process_noise_px;
  p_y_ = (1.0 - ky) * p_y_ + params_.process_noise_px;

  output = filtered_;
  return output;
}

VanishingPointEstimate estimateVanishingPoint(
  const cv::Mat & image_bgr,
  const PipelineParams & params)
{
  VanishingPointEstimate result;
  if (image_bgr.empty()) {
    return result;
  }

  cv::Mat gray;
  if (image_bgr.channels() == 3) {
    cv::cvtColor(image_bgr, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = image_bgr;
  }

  cv::Mat blurred;
  cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.2);
  cv::Mat edges;
  cv::Canny(blurred, edges, 60, 160);

  const int min_len = std::max(18, static_cast<int>(0.04 * std::min(gray.cols, gray.rows)));
  std::vector<cv::Vec4i> raw_lines;
  cv::HoughLinesP(
    edges, raw_lines, 1.0, CV_PI / 180.0,
    std::max(20, min_len), min_len, 12);

  std::vector<Line2D> lines;
  lines.reserve(raw_lines.size());
  const double y_min = gray.rows * 0.30;
  for (const auto & ln : raw_lines) {
    const double x1 = ln[0];
    const double y1 = ln[1];
    const double x2 = ln[2];
    const double y2 = ln[3];
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    const double length = std::hypot(dx, dy);
    if (length < min_len) {
      continue;
    }
    // Lane markings: mostly vertical in frontal view (converging upward).
    if (std::abs(dy) < std::abs(dx) * 0.55) {
      continue;
    }
    const double my = 0.5 * (y1 + y2);
    if (my < y_min) {
      continue;
    }
    lines.push_back({x1, y1, x2, y2, length});
  }

  if (lines.size() < 2) {
    return result;
  }

  const int vote_w = gray.cols;
  const int vote_h = std::max(1, static_cast<int>(gray.rows * 0.70));
  std::vector<float> votes(static_cast<size_t>(vote_w * vote_h), 0.0f);

  for (size_t i = 0; i < lines.size(); ++i) {
    for (size_t j = i + 1; j < lines.size(); ++j) {
      double ix = 0.0;
      double iy = 0.0;
      if (!intersectLines(lines[i], lines[j], ix, iy)) {
        continue;
      }
      if (ix < gray.cols * 0.10 || ix > gray.cols * 0.90) {
        continue;
      }
      if (iy < 0.0 || iy > gray.rows * 0.62) {
        continue;
      }
      const int vx = std::clamp(static_cast<int>(std::lround(ix)), 0, vote_w - 1);
      const int vy = std::clamp(static_cast<int>(std::lround(iy)), 0, vote_h - 1);
      const float w = static_cast<float>(lines[i].length * lines[j].length);
      votes[static_cast<size_t>(vy * vote_w + vx)] += w;
    }
  }

  float best_vote = 0.0f;
  int best_x = vote_w / 2;
  int best_y = static_cast<int>(gray.rows * 0.22);
  for (int y = 0; y < vote_h; ++y) {
    for (int x = 0; x < vote_w; ++x) {
      const float v = votes[static_cast<size_t>(y * vote_w + x)];
      if (v > best_vote) {
        best_vote = v;
        best_x = x;
        best_y = y;
      }
    }
  }

  if (best_vote <= 0.0f) {
    return result;
  }

  double cx = best_x;
  double cy = best_y;
  float second_vote = 0.0f;
  refineVotePeakCentroid(votes, vote_w, vote_h, best_x, best_y, cx, cy, best_vote, second_vote);

  result.x = cx;
  result.y = cy;
  const float conf_denom = best_vote + second_vote + 1e-6f;
  result.confidence = static_cast<double>(best_vote / conf_denom);
  result.valid = result.confidence >= 0.20;
  return result;
}

bool configureAutoIpmRoi(
  PipelineParams & params,
  int cols,
  int rows,
  double vp_x,
  double vp_y)
{
  if (cols < 32 || rows < 32) {
    return false;
  }

  const double w = static_cast<double>(cols);
  const double h = static_cast<double>(rows);

  vp_x = std::clamp(vp_x, 0.25 * w, 0.75 * w);
  vp_y = std::clamp(vp_y, 0.05 * h, 0.58 * h);

  const double bl_x = 0.06 * w;
  const double br_x = 0.94 * w;
  const double bl_y = 0.97 * h;
  const double br_y = bl_y;
  const double y_top = std::clamp(vp_y + 0.12 * h, 0.42 * h, 0.72 * h);

  const cv::Point2f tl = pointOnRay(vp_x, vp_y, bl_x, bl_y, y_top);
  const cv::Point2f tr = pointOnRay(vp_x, vp_y, br_x, br_y, y_top);

  params.bev_width_m = 12.0;
  params.bev_length_m = 40.0;
  params.bev_resolution_m_per_px = 0.05;
  params.ipm_dst_points = {-6.0, 0.0, 6.0, 0.0, 6.0, 40.0, -6.0, 40.0};
  params.ipm_src_points = {
    bl_x, bl_y,
    br_x, br_y,
    static_cast<double>(tr.x), static_cast<double>(tr.y),
    static_cast<double>(tl.x), static_cast<double>(tl.y),
  };
  return true;
}

FrontalHomographyResult estimateFrontalHomography(
  const cv::Mat & image_bgr,
  PipelineParams params,
  VanishingPointTracker * vp_tracker,
  EgoMotionEstimator * ego_motion)
{
  FrontalHomographyResult result;
  result.params = params;
  if (image_bgr.empty()) {
    return result;
  }

  const VanishingPointEstimate raw_vp = estimateVanishingPoint(image_bgr, params);
  VanishingPointEstimate vp = raw_vp;
  if (vp_tracker != nullptr) {
    vp = vp_tracker->update(raw_vp);
  }
  result.vanishing_point = vp;
  result.used_fallback_roi = !vp.valid;

  if (vp.valid) {
    configureAutoIpmRoi(params, image_bgr.cols, image_bgr.rows, vp.x, vp.y);
  } else {
    setDefaultHighwayIpmRoi(params, image_bgr.cols, image_bgr.rows);
  }
  if (ego_motion != nullptr) {
    ego_motion->applyIntegratedShiftToIpmRoi(
      params, image_bgr.cols, image_bgr.rows);
  }
  result.params = params;

  cv::Mat debug;
  result.debug_roi = image_bgr.clone();
  if (vp.valid) {
    cv::circle(
      result.debug_roi,
      cv::Point(static_cast<int>(std::lround(vp.x)), static_cast<int>(std::lround(vp.y))),
      8, cv::Scalar(0, 0, 255), 2);
    if (vp_tracker != nullptr && raw_vp.valid &&
      (std::abs(raw_vp.x - vp.x) > 2.0 || std::abs(raw_vp.y - vp.y) > 2.0))
    {
      cv::circle(
        result.debug_roi,
        cv::Point(static_cast<int>(std::lround(raw_vp.x)), static_cast<int>(std::lround(raw_vp.y))),
        5, cv::Scalar(255, 128, 0), 1);
    }
  }
  if (params.ipm_src_points.size() >= 8) {
    std::vector<cv::Point> poly(4);
    for (int i = 0; i < 4; ++i) {
      poly[static_cast<size_t>(i)] = cv::Point(
        static_cast<int>(params.ipm_src_points[2 * i]),
        static_cast<int>(params.ipm_src_points[2 * i + 1]));
    }
    const std::vector<std::vector<cv::Point>> polys = {poly};
    cv::polylines(result.debug_roi, polys, true, cv::Scalar(0, 255, 255), 2);
  }

  IPMTransformer ipm(params);
  if (!ipm.computeHomography(nullptr)) {
    return result;
  }
  result.H_img2bev = ipm.homography().clone();
  result.bev = ipm.warpToBev(image_bgr);
  if (result.bev.empty()) {
    return result;
  }
  result.bev = prepareBevImage(result.bev);
  result.valid = true;
  return result;
}

cv::Mat warpFrontalAutoIpm(
  const cv::Mat & image_bgr,
  PipelineParams & params,
  VanishingPointEstimate * vp_out,
  cv::Mat * debug_viz,
  VanishingPointTracker * vp_tracker)
{
  const VanishingPointEstimate raw_vp = estimateVanishingPoint(image_bgr, params);
  VanishingPointEstimate vp = raw_vp;
  if (vp_tracker != nullptr) {
    vp = vp_tracker->update(raw_vp);
  }
  if (!vp.valid) {
    setDefaultHighwayIpmRoi(params, image_bgr.cols, image_bgr.rows);
    if (vp_out) {
      *vp_out = vp;
    }
  } else {
    configureAutoIpmRoi(params, image_bgr.cols, image_bgr.rows, vp.x, vp.y);
    if (vp_out) {
      *vp_out = vp;
    }
  }

  if (debug_viz) {
    *debug_viz = image_bgr.clone();
    if (vp.valid) {
      cv::circle(
        *debug_viz,
        cv::Point(static_cast<int>(vp.x), static_cast<int>(vp.y)),
        8, cv::Scalar(0, 0, 255), 2);
    }
    if (params.ipm_src_points.size() >= 8) {
      std::vector<cv::Point> poly(4);
      for (int i = 0; i < 4; ++i) {
        poly[static_cast<size_t>(i)] = cv::Point(
          static_cast<int>(params.ipm_src_points[2 * i]),
          static_cast<int>(params.ipm_src_points[2 * i + 1]));
      }
      const std::vector<std::vector<cv::Point>> polys = {poly};
      cv::polylines(*debug_viz, polys, true, cv::Scalar(0, 255, 255), 2);
    }
  }

  configureParamsForPerspectiveIpm(params);
  cv::Mat warped = warpPerspectiveToBev(image_bgr, params);
  if (warped.empty()) {
    return cv::Mat();
  }
  return prepareBevImage(warped);
}

}  // namespace lie_lane_detection
