#include "lie_lane_detection/pipeline/detection_common.hpp"

#include <algorithm>

#include <opencv2/core.hpp>

#include "lie_lane_detection/fitting/manifold_ransac.hpp"
#include "lie_lane_detection/common/parallel.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"

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
  return std::max(1.0, static_cast<double>(bev_rows) - exclude);
}

void maskBevBottomExclude(cv::Mat & bev_bgr, const PipelineParams & params)
{
  const int exclude = static_cast<int>(std::round(std::max(0.0, params.bev_bottom_exclude_px)));
  if (exclude <= 0 || bev_bgr.empty()) {
    return;
  }
  const int y0 = std::max(0, bev_bgr.rows - exclude);
  if (y0 < bev_bgr.rows) {
    bev_bgr.rowRange(y0, bev_bgr.rows).setTo(cv::Scalar(0, 0, 0));
  }
}

cv::Mat prepareBevForDetection(const cv::Mat & bev_bgr, const PipelineParams & params)
{
  if (params.bev_bottom_exclude_px <= 0.0 || bev_bgr.empty()) {
    return bev_bgr;
  }
  cv::Mat masked = bev_bgr.clone();
  maskBevBottomExclude(masked, params);
  return masked;
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
  std::vector<LaneHypothesis> fitted(seeds.size());
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, seeds.size()),
    [&](const tbb::blocked_range<size_t> & range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        fitted[i] = ransac.fit(seeds[i], edges);
      }
    });

  LaneHypothesis best;
  double best_metric = -1.0;
  for (const auto & hyp : fitted) {
    if (!passesQualityGate(hyp, params, image_height)) {
      continue;
    }
    if (isTooCloseToExisting(hyp, accepted, params.min_lane_separation_px)) {
      continue;
    }
    const double metric = hyp.score * hyp.inlier_ratio;
    if (metric > best_metric) {
      best_metric = metric;
      best = hyp;
    }
  }
  best.score = best_metric;
  return best;
}

}  // namespace lie_lane_detection
