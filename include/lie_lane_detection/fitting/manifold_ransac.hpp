#pragma once

#include <vector>

#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

class ManifoldRansac
{
public:
  ManifoldRansac(const PipelineParams & params, TemplateCurve * template_curve);

  void setTemplateCurve(TemplateCurve * template_curve) {template_curve_ = template_curve;}

  /// Fit a lane to the edges from a Hough seed. When refine is false the costly
  /// non-linear refinement (Ceres / Gauss-Newton) is skipped, yielding the raw
  /// RANSAC consensus - useful for cheaply ranking many candidate seeds before
  /// paying for full refinement on the winner only.
  LaneHypothesis fit(
    const LaneHypothesis & seed,
    const std::vector<EdgePoint> & edges,
    bool refine = true) const;

  void refineGaussNewton(XiVector & xi, const std::vector<EdgePoint> & inliers) const;

private:
  std::vector<EdgePoint> collectCandidates(const XiVector & xi, const std::vector<EdgePoint> & edges) const;

  bool fitMinimal(const std::vector<EdgePoint> & sample, XiVector & xi_out) const;

  int countInliers(
    const XiVector & xi, const std::vector<EdgePoint> & edges, std::vector<bool> * mask,
    double * weighted_inliers = nullptr) const;

  PipelineParams params_;
  TemplateCurve * template_curve_{nullptr};
};

}  // namespace lie_lane_detection
