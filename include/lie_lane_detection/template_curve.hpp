#pragma once

#include <utility>

#include "lie_lane_detection/types.hpp"

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

  Vec2 nearestPoint(const XiVector & xi, const Vec2 & p, double * nearest_t = nullptr) const;

  double lateralAtY(const XiVector & xi, double y) const;

private:
  PipelineParams params_;
  double y_min_{0.0};
  double y_max_{1.0};
  double x_min_{0.0};
  double x_max_{1.0};
};

}  // namespace lie_lane_detection
