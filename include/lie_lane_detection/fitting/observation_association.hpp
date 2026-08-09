#pragma once

#include "lie_lane_detection/core/types.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"

#include <vector>

namespace lie_lane_detection
{

struct AssociatedLine
{
  LineSegment segment;
  double soft_weight{1.0};
  double length_weight{1.0};
};

struct AssociatedEdge
{
  EdgePoint edge;
  double soft_weight{1.0};
  double length_weight{1.0};
  double t_near{0.0};
};

/// Hard-gated observation association for Ceres (distance + heading + longitudinal).
class ObservationAssociation
{
public:
  ObservationAssociation(const PipelineParams & params, TemplateCurve * template_curve);

  std::vector<AssociatedLine> associateLines(
    const XiVector & seed_xi, const std::vector<LineSegment> & lines) const;

  std::vector<AssociatedEdge> associateEdges(
    const XiVector & seed_xi, const std::vector<EdgePoint> & edges) const;

  bool lineSupportsSeed(const XiVector & seed_xi, const LineSegment & line) const;

  bool edgeSupportsSeed(const XiVector & seed_xi, const EdgePoint & edge) const;

private:
  PipelineParams params_;
  TemplateCurve * template_curve_{nullptr};
};

}  // namespace lie_lane_detection
