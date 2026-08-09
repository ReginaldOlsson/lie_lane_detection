#pragma once

#include "lie_lane_detection/core/types.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"

#include <cmath>
#include <random>
#include <vector>

namespace lie_lane_detection
{
namespace test_helpers
{

inline std::vector<EdgePoint> samplePointsOnCurve(
  const TemplateCurve & curve, const XiVector & xi, int num_points, double noise_std = 0.0,
  unsigned seed = 42)
{
  std::mt19937 rng(seed);
  std::normal_distribution<double> noise(0.0, noise_std);

  std::vector<EdgePoint> edges;
  edges.reserve(static_cast<size_t>(num_points));
  for (int i = 0; i < num_points; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(num_points - 1);
    Vec2 p = curve.sample(xi, t);
    p.x() += noise(rng);
    p.y() += noise(rng);
    const double t1 = std::clamp(t + 1e-3, 0.0, 1.0);
    const Vec2 p1 = curve.sample(xi, t1);
    EdgePoint ep;
    ep.x = p.x();
    ep.y = p.y();
    ep.magnitude = 1.0;
    ep.orientation = std::atan2(p1.y() - p.y(), p1.x() - p.x());
    edges.push_back(ep);
  }
  return edges;
}

}  // namespace test_helpers
}  // namespace lie_lane_detection
