#pragma once

#include <vector>

#include "lie_lane_detection/template_curve.hpp"
#include "lie_lane_detection/types.hpp"

namespace lie_lane_detection
{

class ManifoldRansac
{
public:
  ManifoldRansac(const PipelineParams & params, TemplateCurve * template_curve);

  void setTemplateCurve(TemplateCurve * template_curve) {template_curve_ = template_curve;}

  LaneHypothesis fit(
    const LaneHypothesis & seed,
    const std::vector<EdgePoint> & edges) const;

  void refineGaussNewton(XiVector & xi, const std::vector<EdgePoint> & inliers) const;

private:
  std::vector<EdgePoint> collectCandidates(const XiVector & xi, const std::vector<EdgePoint> & edges) const;

  bool fitMinimal(const std::vector<EdgePoint> & sample, XiVector & xi_out) const;

  int countInliers(const XiVector & xi, const std::vector<EdgePoint> & edges, std::vector<bool> * mask) const;

  PipelineParams params_;
  TemplateCurve * template_curve_{nullptr};
};

}  // namespace lie_lane_detection
