#pragma once

#include "lie_lane_detection/core/types.hpp"
#include "lie_lane_detection/fitting/observation_association.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"

#include <utility>
#include <vector>

namespace lie_lane_detection
{

struct RansacGateResult
{
  XiVector xi{XiVector::Zero()};
  std::vector<LineSegment> inlier_lines;
  double inlier_ratio{0.0};
  bool valid{false};
};

/// Fast 2-line RANSAC gate: prune crosswalk outliers before Ceres initialization.
class LineSegmentRansacGate
{
public:
  LineSegmentRansacGate(const PipelineParams & params, TemplateCurve * template_curve);

  RansacGateResult filterSeed(
    const LaneHypothesis & seed, const std::vector<LineSegment> & lines) const;

private:
  bool fitFromTwoLines(const LineSegment & a, const LineSegment & b, XiVector & xi_out) const;

  int countInliers(
    const XiVector & xi, const std::vector<LineSegment> & candidates,
    std::vector<bool> * mask) const;

  PipelineParams params_;
  TemplateCurve * template_curve_{nullptr};
  ObservationAssociation association_;
};

}  // namespace lie_lane_detection
