#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"

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

}  // namespace

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

  result.x = static_cast<double>(best_x);
  result.y = static_cast<double>(best_y);
  result.confidence = static_cast<double>(best_vote);
  result.valid = true;
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
  PipelineParams params)
{
  FrontalHomographyResult result;
  result.params = params;
  if (image_bgr.empty()) {
    return result;
  }

  VanishingPointEstimate vp = estimateVanishingPoint(image_bgr, params);
  result.vanishing_point = vp;
  result.used_fallback_roi = !vp.valid;

  if (vp.valid) {
    configureAutoIpmRoi(params, image_bgr.cols, image_bgr.rows, vp.x, vp.y);
  } else {
    setDefaultHighwayIpmRoi(params, image_bgr.cols, image_bgr.rows);
  }
  result.params = params;

  cv::Mat debug;
  result.debug_roi = image_bgr.clone();
  if (vp.valid) {
    cv::circle(
      result.debug_roi,
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
  cv::Mat * debug_viz)
{
  VanishingPointEstimate vp = estimateVanishingPoint(image_bgr, params);
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
