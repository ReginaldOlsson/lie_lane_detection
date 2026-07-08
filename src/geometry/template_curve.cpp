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
  // Work in the SE(2)-local frame where the deformed template is an explicit
  // quadratic in the curve parameter t:
  //   x_c(t) = A t^2 + B t,   y_c(t) = y_min + y_span t,   yn == t
  // with A = kappa * y_span, B = sigma * y_span. Rotation is an isometry, so the
  // Euclidean distance is identical in the local and world frames. We seed t from
  // the (linear) y-inversion and refine with a few Newton steps on d^2(t), which
  // replaces the previous 36-sample brute-force scan (each an SE(2) multiply).
  const Sophus::SE2d g = xiToSE2(xi);
  const Vec2 p_local = g.inverse() * p;
  return distanceInLocalFrame(p_local.x(), p_local.y(), xi[3], xi[4], nearest_t);
}

double TemplateCurve::distanceInLocalFrame(
  double a, double b, double kappa, double sigma, double * nearest_t) const
{
  const double y_span = std::max(y_max_ - y_min_, 1e-6);
  const double A = kappa * y_span;
  const double B = sigma * y_span;

  auto distSq = [&](double t) {
    const double xc = (A * t + B) * t;
    const double yc = y_min_ + y_span * t;
    const double dx = a - xc;
    const double dy = b - yc;
    return dx * dx + dy * dy;
  };

  // Seed from the linear y-coordinate inversion, then refine with Newton on
  // d^2(t). Since kappa/sigma are bounded small the objective is near-convex on
  // [0, 1], so this converges to the true minimizer. We deliberately keep t a
  // smooth (converged) function of xi (no discrete grid / endpoint-min branch):
  // downstream Ceres residuals use sample(xi, t) and tangentAt(xi, t), and a
  // discontinuous t breaks the numeric-diff Jacobians (notably along the lane's
  // gauge-free longitudinal direction).
  double t = std::clamp((b - y_min_) / y_span, 0.0, 1.0);
  constexpr int kNewtonIters = 8;
  for (int iter = 0; iter < kNewtonIters; ++iter) {
    const double xc = (A * t + B) * t;
    const double yc = y_min_ + y_span * t;
    const double xc_p = 2.0 * A * t + B;   // x_c'(t)
    const double yc_p = y_span;            // y_c'(t)
    // d(d^2)/dt = 2[(xc-a) xc' + (yc-b) yc']
    const double grad = (xc - a) * xc_p + (yc - b) * yc_p;
    // d^2(d^2)/dt^2 = 2[xc'^2 + (xc-a) xc'' + yc'^2], with xc'' = 2A
    const double hess = xc_p * xc_p + (xc - a) * (2.0 * A) + yc_p * yc_p;
    if (std::abs(hess) < 1e-12) {
      break;
    }
    const double t_new = std::clamp(t - grad / hess, 0.0, 1.0);
    const double dt = t_new - t;
    t = t_new;
    if (std::abs(dt) < 1e-6) {
      break;
    }
  }

  if (nearest_t) {
    *nearest_t = t;
  }
  return std::sqrt(distSq(t));
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
