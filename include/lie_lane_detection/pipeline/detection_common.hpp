#pragma once

#include <vector>

#include "lie_lane_detection/fitting/manifold_ransac.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

bool isBorderLane(const LaneHypothesis & lane, const PipelineParams & params);

bool isTooCloseToExisting(
  const LaneHypothesis & lane,
  const std::vector<LaneHypothesis> & existing,
  double min_sep_px);

bool passesQualityGate(
  const LaneHypothesis & lane,
  const PipelineParams & params,
  double image_height);

std::vector<EdgePoint> filterBorderEdges(
  const std::vector<EdgePoint> & edges,
  double x_min,
  double x_max);

std::vector<EdgePoint> peelEdgesNearCurve(
  const std::vector<EdgePoint> & edges,
  const TemplateCurve & curve,
  const XiVector & xi,
  double margin_px);

std::vector<EdgePoint> subsampleEdges(const std::vector<EdgePoint> & edges, size_t max_count);

std::vector<LineSegment> filterBorderLines(
  const std::vector<LineSegment> & lines,
  double x_min,
  double x_max);

std::vector<LineSegment> peelLinesNearCurve(
  const std::vector<LineSegment> & lines,
  const TemplateCurve & curve,
  const XiVector & xi,
  double margin_px);

/// Sample line midpoints (and endpoints for long segments) as pseudo-edges for RANSAC.
std::vector<EdgePoint> linesToRefineEdges(const std::vector<LineSegment> & lines);

LaneHypothesis pickBestSeed(
  const std::vector<LaneHypothesis> & seeds,
  const std::vector<EdgePoint> & edges,
  ManifoldRansac & ransac,
  const std::vector<LaneHypothesis> & accepted,
  const PipelineParams & params,
  double image_height);

}  // namespace lie_lane_detection
