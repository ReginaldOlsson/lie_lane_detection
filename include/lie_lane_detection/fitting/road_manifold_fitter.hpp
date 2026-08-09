#pragma once

#include "lie_lane_detection/core/types.hpp"
#include "lie_lane_detection/fitting/ceres_lane_fitter.hpp"
#include "lie_lane_detection/fitting/observation_association.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"

#include <vector>

namespace lie_lane_detection
{

/// Joint road manifold: shared T_ego, kappa, sigma, w_lane with fixed lane-index offsets.
class RoadManifoldFitter
{
public:
  RoadManifoldFitter(const PipelineParams & params, TemplateCurve * template_curve);

  void setTemplateCurve(TemplateCurve * template_curve) { template_curve_ = template_curve; }

  /// Refine peeled lane hypotheses in-place using joint optimization on line segments.
  bool refine(std::vector<LaneHypothesis> & lanes, const std::vector<LineSegment> & lines) const;

  /// Refine peeled lane hypotheses using edge points (edge pipeline).
  bool refine(std::vector<LaneHypothesis> & lanes, const std::vector<EdgePoint> & edges) const;

  static double laneOffsetMultiplier(int lane_index);

  static int laneSlotForRank(int rank, int total_lanes);

  static XiVector decodeLaneXi(
    const double * ego_xi, int lane_index, double kappa, double sigma, double w_lane);

private:
  struct LaneAssignment
  {
    int lane_slot{0};
    std::vector<AssociatedLine> lines;
    std::vector<AssociatedEdge> edges;
  };

  std::vector<LaneAssignment> assignLanes(
    const std::vector<LaneHypothesis> & lanes, const std::vector<LineSegment> & lines) const;

  std::vector<LaneAssignment> assignLanes(
    const std::vector<LaneHypothesis> & lanes, const std::vector<EdgePoint> & edges) const;

  bool optimizeRoad(
    const std::vector<LaneHypothesis> & seeds, const std::vector<LaneAssignment> & assignments,
    double * road_params) const;

  PipelineParams params_;
  TemplateCurve * template_curve_{nullptr};
  CeresLaneFitter ceres_fitter_;
};

}  // namespace lie_lane_detection
