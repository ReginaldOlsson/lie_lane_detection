#include "lie_lane_detection/pipeline/detection_common.hpp"

#include <algorithm>

#include <opencv2/imgproc.hpp>

#include "lie_lane_detection/fitting/manifold_ransac.hpp"
#include "lie_lane_detection/common/parallel.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/preprocessing/edge_extractor.hpp"

namespace lie_lane_detection
{

bool isBorderLane(const LaneHypothesis & lane, const PipelineParams & params)
{
  const double span = params.se2_vx_max - params.se2_vx_min;
  const double margin = span * params.edge_border_margin_ratio;
  return lane.xi[0] < params.se2_vx_min + margin ||
         lane.xi[0] > params.se2_vx_max - margin;
}

bool isTooCloseToExisting(
  const LaneHypothesis & lane,
  const std::vector<LaneHypothesis> & existing,
  double min_sep_px)
{
  for (const auto & other : existing) {
    if (std::abs(lane.xi[0] - other.xi[0]) < min_sep_px) {
      return true;
    }
  }
  return false;
}

bool passesQualityGate(
  const LaneHypothesis & lane,
  const PipelineParams & params,
  double image_height)
{
  if (lane.inlier_ratio < params.min_inlier_ratio) {
    return false;
  }
  if (static_cast<int>(lane.supporting_edges.size()) < params.min_inliers) {
    return false;
  }
  if (isBorderLane(lane, params)) {
    return false;
  }

  if (!lane.supporting_edges.empty() && image_height > 1.0) {
    double y_min = lane.supporting_edges.front().y;
    double y_max = lane.supporting_edges.front().y;
    for (const auto & e : lane.supporting_edges) {
      y_min = std::min(y_min, e.y);
      y_max = std::max(y_max, e.y);
    }
    const double coverage = (y_max - y_min) / image_height;
    if (coverage < params.min_inlier_y_coverage) {
      return false;
    }
  }
  return true;
}

double bevEffectiveYMax(int bev_rows, const PipelineParams & params)
{
  const double exclude = std::max(0.0, params.bev_bottom_exclude_px);
  const double margin = std::max(0.0, params.bev_bottom_edge_margin_px);
  return std::max(1.0, static_cast<double>(bev_rows) - exclude - margin);
}

int bevBottomMaskRows(const PipelineParams & params)
{
  const double exclude = std::max(0.0, params.bev_bottom_exclude_px);
  if (exclude <= 0.0) {
    return 0;
  }
  const double margin = std::max(0.0, params.bev_bottom_edge_margin_px);
  return static_cast<int>(std::round(exclude + margin));
}

void maskBevBottomRows(cv::Mat & bev_bgr, int mask_rows)
{
  if (mask_rows <= 0 || bev_bgr.empty()) {
    return;
  }
  const int y0 = std::max(0, bev_bgr.rows - mask_rows);
  if (y0 < bev_bgr.rows) {
    bev_bgr.rowRange(y0, bev_bgr.rows).setTo(cv::Scalar(0, 0, 0));
  }
}

void maskBevBottomExclude(cv::Mat & bev_bgr, const PipelineParams & params)
{
  const int exclude = static_cast<int>(std::round(std::max(0.0, params.bev_bottom_exclude_px)));
  maskBevBottomRows(bev_bgr, exclude);
}

void maskBevBottomForDetection(cv::Mat & bev_bgr, const PipelineParams & params)
{
  maskBevBottomRows(bev_bgr, bevBottomMaskRows(params));
}

cv::Mat prepareBevForDetection(const cv::Mat & bev_bgr, const PipelineParams & params)
{
  if (bevBottomMaskRows(params) <= 0 || bev_bgr.empty()) {
    return bev_bgr;
  }
  cv::Mat masked = bev_bgr.clone();
  maskBevBottomForDetection(masked, params);
  return masked;
}

BevTrackingPrep prepareBevGrayForTracking(
  const cv::Mat & bev_bgr,
  const PipelineParams & params)
{
  BevTrackingPrep out;
  if (bev_bgr.empty()) {
    return out;
  }

  cv::Mat bgr = prepareBevForDetection(bev_bgr, params);
  if (bgr.channels() == 1) {
    out.gray = bgr.clone();
    cv::cvtColor(bgr, out.display_bgr, cv::COLOR_GRAY2BGR);
  } else {
    out.display_bgr = bgr.clone();
    cv::cvtColor(bgr, out.gray, cv::COLOR_BGR2GRAY);
  }

  const uchar min_road = static_cast<uchar>(std::clamp(params.bev_min_road_gray, 0, 255));
  const cv::Mat road_mask = out.gray >= min_road;
  out.gray.setTo(0, ~road_mask);
  return out;
}

BevPreprocessResult preprocessBevForLaneDetection(
  const cv::Mat & bev_bgr,
  const PipelineParams & params)
{
  BevPreprocessResult out;
  if (bev_bgr.empty()) {
    return out;
  }

  cv::Mat bgr = prepareBevForDetection(bev_bgr, params);
  if (bgr.channels() == 1) {
    cv::cvtColor(bgr, out.display_bgr, cv::COLOR_GRAY2BGR);
  } else {
    out.display_bgr = bgr.clone();
  }

  cv::Mat work = out.display_bgr.clone();
  if (params.bev_use_sharpen) {
    const cv::Mat kernel = (cv::Mat_<float>(3, 3) <<
      -1.f, -1.f, -1.f,
      -1.f,  9.f, -1.f,
      -1.f, -1.f, -1.f);
    cv::filter2D(work, work, -1, kernel);
  }

  cv::Mat gray;
  cv::cvtColor(work, gray, cv::COLOR_BGR2GRAY);

  const int blur_k = params.bev_gaussian_blur_ksize;
  if (blur_k >= 3) {
    const int k = blur_k | 1;
    cv::GaussianBlur(gray, gray, cv::Size(k, k), 0.0);
  }

  const uchar min_road = static_cast<uchar>(std::clamp(params.bev_min_road_gray, 0, 255));
  cv::Mat road_mask = gray >= min_road;

  // Grayscale keeps lane paint gradients for EdgeExtractor (Sobel / steerable bank).
  cv::Mat detect = gray.clone();
  detect.setTo(0, ~road_mask);

  cv::Mat filtered_debug = detect.clone();
  if (params.bev_use_otsu) {
    cv::Mat otsu_binary;
    out.otsu_threshold = EdgeExtractor::otsuThresholdMasked(gray, road_mask, otsu_binary);
    filtered_debug = otsu_binary;

    const int open_px = params.bev_morph_open_px;
    if (open_px >= 3) {
      const int k = open_px | 1;
      const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(k, k));
      cv::morphologyEx(filtered_debug, filtered_debug, cv::MORPH_OPEN, kernel);
      filtered_debug.setTo(0, ~road_mask);
    }

    if (params.bev_otsu_for_detection) {
      detect = filtered_debug.clone();
    }
  }

  out.detect_image = detect;
  out.filtered_debug = filtered_debug;
  return out;
}

std::vector<EdgePoint> filterBorderEdges(
  const std::vector<EdgePoint> & edges,
  double x_min,
  double x_max)
{
  std::vector<EdgePoint> filtered;
  filtered.reserve(edges.size());
  for (const auto & e : edges) {
    if (e.x >= x_min && e.x <= x_max) {
      filtered.push_back(e);
    }
  }
  return filtered;
}

std::vector<EdgePoint> filterBevYMaxEdges(
  const std::vector<EdgePoint> & edges,
  double y_max)
{
  std::vector<EdgePoint> filtered;
  filtered.reserve(edges.size());
  for (const auto & e : edges) {
    if (e.y < y_max) {
      filtered.push_back(e);
    }
  }
  return filtered;
}

std::vector<EdgePoint> peelEdgesNearCurve(
  const std::vector<EdgePoint> & edges,
  const TemplateCurve & curve,
  const XiVector & xi,
  double margin_px)
{
  std::vector<EdgePoint> remaining;
  remaining.reserve(edges.size());
  for (const auto & e : edges) {
    const Vec2 p(e.x, e.y);
    if (curve.distanceToCurve(xi, p) > margin_px) {
      remaining.push_back(e);
    }
  }
  return remaining;
}

std::vector<EdgePoint> subsampleEdges(const std::vector<EdgePoint> & edges, size_t max_count)
{
  if (edges.size() <= max_count) {
    return edges;
  }
  std::vector<EdgePoint> sampled;
  sampled.reserve(max_count);
  const double stride = static_cast<double>(edges.size()) / static_cast<double>(max_count);
  for (size_t i = 0; i < max_count; ++i) {
    sampled.push_back(edges[static_cast<size_t>(i * stride)]);
  }
  return sampled;
}

std::vector<LineSegment> filterBorderLines(
  const std::vector<LineSegment> & lines,
  double x_min,
  double x_max)
{
  std::vector<LineSegment> filtered;
  filtered.reserve(lines.size());
  for (const auto & line : lines) {
    if (line.mx >= x_min && line.mx <= x_max) {
      filtered.push_back(line);
    }
  }
  return filtered;
}

std::vector<LineSegment> filterBevYMaxLines(
  const std::vector<LineSegment> & lines,
  double y_max)
{
  std::vector<LineSegment> filtered;
  filtered.reserve(lines.size());
  for (const auto & line : lines) {
    if (line.my < y_max && std::min(line.y1, line.y2) < y_max) {
      filtered.push_back(line);
    }
  }
  return filtered;
}

std::vector<LineSegment> peelLinesNearCurve(
  const std::vector<LineSegment> & lines,
  const TemplateCurve & curve,
  const XiVector & xi,
  double margin_px)
{
  std::vector<LineSegment> remaining;
  remaining.reserve(lines.size());
  for (const auto & line : lines) {
    const Vec2 p(line.mx, line.my);
    if (curve.distanceToCurve(xi, p) > margin_px) {
      remaining.push_back(line);
    }
  }
  return remaining;
}

std::vector<EdgePoint> linesToRefineEdges(const std::vector<LineSegment> & lines)
{
  std::vector<EdgePoint> edges;
  edges.reserve(lines.size() * 3);
  for (const auto & line : lines) {
    auto add = [&](double x, double y) {
        EdgePoint ep;
        ep.x = x;
        ep.y = y;
        ep.magnitude = line.length;
        ep.orientation = line.angle;
        edges.push_back(ep);
      };
    add(line.mx, line.my);
    if (line.length > 25.0) {
      add(line.x1, line.y1);
      add(line.x2, line.y2);
    }
  }
  return edges;
}

LaneHypothesis pickBestSeed(
  const std::vector<LaneHypothesis> & seeds,
  const std::vector<EdgePoint> & edges,
  ManifoldRansac & ransac,
  const std::vector<LaneHypothesis> & accepted,
  const PipelineParams & params,
  double image_height)
{
  // Rank all seeds cheaply with the raw RANSAC consensus (no Ceres), then pay
  // for the expensive non-linear refinement on the single winner only. Fitting
  // every seed fully wasted most of the Ceres work on hypotheses that are
  // immediately discarded.
  std::vector<LaneHypothesis> fitted(seeds.size());
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, seeds.size()),
    [&](const tbb::blocked_range<size_t> & range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        fitted[i] = ransac.fit(seeds[i], edges, /*refine=*/false);
      }
    });

  int best_idx = -1;
  double best_metric = -1.0;
  for (size_t i = 0; i < fitted.size(); ++i) {
    const auto & hyp = fitted[i];
    if (!passesQualityGate(hyp, params, image_height)) {
      continue;
    }
    if (isTooCloseToExisting(hyp, accepted, params.min_lane_separation_px)) {
      continue;
    }
    const double metric = hyp.score * hyp.inlier_ratio;
    if (metric > best_metric) {
      best_metric = metric;
      best_idx = static_cast<int>(i);
    }
  }

  if (best_idx < 0) {
    return LaneHypothesis{};
  }

  // Full refinement on the winning seed, then re-validate.
  LaneHypothesis best = ransac.fit(seeds[static_cast<size_t>(best_idx)], edges, /*refine=*/true);
  if (!passesQualityGate(best, params, image_height) ||
    isTooCloseToExisting(best, accepted, params.min_lane_separation_px))
  {
    return LaneHypothesis{};
  }
  best.score = best.score * best.inlier_ratio;
  return best;
}

}  // namespace lie_lane_detection
