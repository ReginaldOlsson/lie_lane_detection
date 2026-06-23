#pragma once

#include <vector>

#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

class MergeTopology
{
public:
  MergeTopology(const PipelineParams & params, TemplateCurve * template_curve);

  void setTemplateCurve(TemplateCurve * template_curve) {template_curve_ = template_curve;}

  std::vector<MergeEvent> analyze(std::vector<LaneHypothesis> & lanes) const;

  void completeGaps(LaneHypothesis & lane) const;

private:
  MergeTopologyType classifyPair(const LaneHypothesis & a, const LaneHypothesis & b, Vec2 * merge_point) const;

  PipelineParams params_;
  TemplateCurve * template_curve_{nullptr};
};

}  // namespace lie_lane_detection
