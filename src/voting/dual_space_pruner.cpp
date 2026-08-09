#include "lie_lane_detection/voting/dual_space_pruner.hpp"

#include <opencv2/core.hpp>

#include <cmath>

namespace lie_lane_detection
{

DualSpacePruner::DualSpacePruner(const PipelineParams & params) : params_(params)
{
}

std::vector<LineSegment> DualSpacePruner::prune(const std::vector<LineSegment> & lines) const
{
  if (lines.size() < 3) {
    return lines;
  }

  // Estimate dominant lane direction from long segments.
  double sum_angle = 0.0;
  double sum_w = 0.0;
  for (const auto & line : lines) {
    if (line.length < params_.line_min_length_px * 0.5) {
      continue;
    }
    sum_angle += line.angle * line.length;
    sum_w += line.length;
  }
  if (sum_w < 1e-6) {
    return lines;
  }
  const double dom_angle = sum_angle / sum_w;
  const double angle_tol = params_.line_angle_threshold_rad * 1.5;

  std::vector<LineSegment> kept;
  kept.reserve(lines.size());
  for (const auto & line : lines) {
    double da = std::abs(line.angle - dom_angle);
    while (da > CV_PI) {
      da -= CV_PI;
    }
    da = std::min(da, CV_PI - da);
    if (da <= angle_tol) {
      kept.push_back(line);
    }
  }
  return kept.empty() ? lines : kept;
}

}  // namespace lie_lane_detection
