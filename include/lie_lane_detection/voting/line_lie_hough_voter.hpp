#pragma once

#include "lie_lane_detection/core/types.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"

#include <vector>

namespace lie_lane_detection
{

/// Lie-Hough voter using line segments (position + tangent constraints).
class LineLieHoughVoter
{
public:
  LineLieHoughVoter(const PipelineParams & params, TemplateCurve * template_curve);

  void setTemplateCurve(TemplateCurve * template_curve) { template_curve_ = template_curve; }

  std::vector<LaneHypothesis> vote(
    const std::vector<LineSegment> & lines, cv::Mat * hough_debug_slice = nullptr);

private:
  struct SE2Peak
  {
    int ix{0};
    int iy{0};
    int io{0};
    double votes{0.0};
    XiVector xi{XiVector::Zero()};
  };

  XiVector binToXi(int ix, int iy, int io, int ik, int is) const;
  void binIndicesToXiComponents(
    int ix, int iy, int io, int ik, int is, double & vx, double & vy, double & omega,
    double & kappa, double & sigma) const;

  bool isPeakSeparated(const XiVector & a, const XiVector & b) const;

  bool lineSupportsXi(
    const LineSegment & line, const XiVector & xi, double vote_thresh_sq,
    double angle_thresh) const;

  PipelineParams params_;
  TemplateCurve * template_curve_{nullptr};
};

}  // namespace lie_lane_detection
