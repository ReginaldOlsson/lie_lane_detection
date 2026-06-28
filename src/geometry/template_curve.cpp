#include "lie_lane_detection/geometry/template_curve.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lie_lane_detection
{

TemplateCurve::TemplateCurve(const PipelineParams & params)
: params_(params)
{
  setBevExtents(0.0, 1.0, -1.0, 1.0);
}

void TemplateCurve::setBevExtents(double y_min, double y_max, double x_min, double x_max)
{
  y_min_ = y_min;
  y_max_ = y_max;
  x_min_ = x_min;
  x_max_ = x_max;
}

Vec2 TemplateCurve::deformTemplate(double t, double kappa, double sigma) const
{
  const double y_span = std::max(y_max_ - y_min_, 1e-6);
  const double y = y_min_ + t * y_span;
  const double yn = (y - y_min_) / y_span;
  // kappa/sigma scaled to BEV pixels: quadratic + linear lateral drift vs forward
  const double x = kappa * yn * yn * y_span + sigma * yn * y_span;
  return Vec2(x, y);
}

Vec2 TemplateCurve::sample(const XiVector & xi, double t) const
{
  const Sophus::SE2d g = xiToSE2(xi);
  return g * deformTemplate(t, xi[3], xi[4]);
}

std::vector<Vec2> TemplateCurve::samplePolyline(const XiVector & xi, int num_samples) const
{
  if (num_samples < 0) {
    num_samples = params_.template_samples;
  }
  std::vector<Vec2> polyline;
  polyline.reserve(static_cast<size_t>(num_samples));
  for (int i = 0; i < num_samples; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(num_samples - 1);
    polyline.push_back(sample(xi, t));
  }
  return polyline;
}

double TemplateCurve::distanceToCurve(const XiVector & xi, const Vec2 & p, double * nearest_t) const
{
  constexpr int kCoarseSteps = 24;
  double best_dist = std::numeric_limits<double>::max();
  double best_t = 0.0;
  int best_i = 0;
  for (int i = 0; i <= kCoarseSteps; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(kCoarseSteps);
    const Vec2 q = sample(xi, t);
    const double d = (p - q).norm();
    if (d < best_dist) {
      best_dist = d;
      best_t = t;
      best_i = i;
    }
  }

  constexpr int kFineSteps = 12;
  const double t_lo = std::max(
    0.0, static_cast<double>(best_i - 1) / static_cast<double>(kCoarseSteps));
  const double t_hi = std::min(
    1.0, static_cast<double>(best_i + 1) / static_cast<double>(kCoarseSteps));
  for (int i = 0; i <= kFineSteps; ++i) {
    const double t = t_lo + (t_hi - t_lo) * static_cast<double>(i) / static_cast<double>(kFineSteps);
    const Vec2 q = sample(xi, t);
    const double d = (p - q).norm();
    if (d < best_dist) {
      best_dist = d;
      best_t = t;
    }
  }

  if (nearest_t) {
    *nearest_t = best_t;
  }
  return best_dist;
}

Vec2 TemplateCurve::nearestPoint(const XiVector & xi, const Vec2 & p, double * nearest_t) const
{
  if (std::abs(xi[3]) < 1e-4 && std::abs(xi[4]) < 1e-4 && std::abs(xi[2]) < 0.03 && std::abs(xi[1]) < 0.5) {
    const double y = std::clamp(p.y(), y_min_, y_max_);
    const double t = (y_max_ > y_min_) ? (y - y_min_) / (y_max_ - y_min_) : 0.0;
    if (nearest_t) {
      *nearest_t = t;
    }
    return Vec2(xi[0], y);
  }

  double t = 0.0;
  distanceToCurve(xi, p, &t);
  if (nearest_t) {
    *nearest_t = t;
  }
  return sample(xi, t);
}

double TemplateCurve::segmentDistanceToCurve(const XiVector & xi, const LineSegment & seg) const
{
  const Vec2 p0(seg.x1, seg.y1);
  const Vec2 p1(seg.x2, seg.y2);
  const double d0 = distanceToCurve(xi, p0);
  const double d1 = distanceToCurve(xi, p1);
  const Vec2 pm(seg.mx, seg.my);
  const double dm = distanceToCurve(xi, pm);
  constexpr int kInterior = 4;
  double best = std::min({d0, d1, dm});
  for (int i = 1; i < kInterior; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(kInterior);
    const Vec2 p = p0 + u * (p1 - p0);
    best = std::min(best, distanceToCurve(xi, p));
  }
  return best;
}

double TemplateCurve::lateralAtY(const XiVector & xi, double y) const
{
  constexpr int kSearchSteps = 100;
  double best_t = 0.0;
  double best_dy = std::numeric_limits<double>::max();
  for (int i = 0; i <= kSearchSteps; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(kSearchSteps);
    const Vec2 q = sample(xi, t);
    const double dy = std::abs(q.y() - y);
    if (dy < best_dy) {
      best_dy = dy;
      best_t = t;
    }
  }
  return sample(xi, best_t).x();
}

Vec2 TemplateCurve::tangentAt(const XiVector & xi, double t) const
{
  const double dt = 1e-3;
  const double t0 = std::clamp(t - dt, 0.0, 1.0);
  const double t1 = std::clamp(t + dt, 0.0, 1.0);
  const Vec2 a = sample(xi, t0);
  const Vec2 b = sample(xi, t1);
  const Vec2 d = b - a;
  const double n = d.norm();
  if (n < 1e-9) {
    return Vec2(0.0, 1.0);
  }
  return d / n;
}

}  // namespace lie_lane_detection
