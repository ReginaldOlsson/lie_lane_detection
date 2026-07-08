#pragma once

#include <utility>

#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

class TemplateCurve
{
public:
  explicit TemplateCurve(const PipelineParams & params);

  void setBevExtents(double y_min, double y_max, double x_min, double x_max);

  Vec2 deformTemplate(double t, double kappa, double sigma) const;

  Vec2 sample(const XiVector & xi, double t) const;

  std::vector<Vec2> samplePolyline(const XiVector & xi, int num_samples = -1) const;

  double distanceToCurve(const XiVector & xi, const Vec2 & p, double * nearest_t = nullptr) const;

  /// Distance from a point already expressed in the SE(2)-local frame (a=lateral,
  /// b=forward) to the deformed template x_c(t)=A t^2 + B t, y_c(t)=y_min+y_span t
  /// (A=kappa*y_span, B=sigma*y_span). Callers that evaluate many (kappa,sigma)
  /// values for a fixed pose can transform the point once and reuse it here.
  double distanceInLocalFrame(
    double a, double b, double kappa, double sigma, double * nearest_t = nullptr) const;

  /// Minimum distance from a line segment to the curve (not midpoint-only).
  double segmentDistanceToCurve(const XiVector & xi, const LineSegment & seg) const;

  Vec2 nearestPoint(const XiVector & xi, const Vec2 & p, double * nearest_t = nullptr) const;

  Vec2 tangentAt(const XiVector & xi, double t) const;

  double lateralAtY(const XiVector & xi, double y) const;

  /// Local-frame forward-axis origin and span used by distanceInLocalFrame.
  /// Exposed so GPU kernels can replicate the exact template geometry.
  double localFrameYMin() const {return y_min_;}
  double localFrameYSpan() const
  {
    const double span = y_max_ - y_min_;
    return span > 1e-6 ? span : 1e-6;
  }

private:
  PipelineParams params_;
  double y_min_{0.0};
  double y_max_{1.0};
  double x_min_{0.0};
  double x_max_{1.0};
};

}  // namespace lie_lane_detection
