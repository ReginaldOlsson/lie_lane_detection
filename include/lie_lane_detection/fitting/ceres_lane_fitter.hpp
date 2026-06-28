#pragma once

#include <vector>

#include "lie_lane_detection/fitting/observation_association.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

/// Continuous Levenberg–Marquardt lane fit on M = SE(2) × ℝ² via Ceres.
class CeresLaneFitter
{
public:
  CeresLaneFitter(const PipelineParams & params, TemplateCurve * template_curve);

  void setTemplateCurve(TemplateCurve * template_curve) {template_curve_ = template_curve;}

  LaneHypothesis fitEdges(const LaneHypothesis & seed, const std::vector<EdgePoint> & edges) const;

  LaneHypothesis fitLines(const LaneHypothesis & seed, const std::vector<LineSegment> & lines) const;

private:
  bool optimizeXi(
    XiVector & xi,
    const std::vector<EdgePoint> & points,
    const std::vector<double> & weights) const;

  bool optimizeXiFromEdges(XiVector & xi, const std::vector<AssociatedEdge> & edges) const;

  bool optimizeXiFromLines(XiVector & xi, const std::vector<AssociatedLine> & lines) const;

  PipelineParams params_;
  TemplateCurve * template_curve_{nullptr};
};

}  // namespace lie_lane_detection
