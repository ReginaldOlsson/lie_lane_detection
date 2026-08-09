#include "lie_lane_detection/fitting/ceres_lane_fitter.hpp"

#include "lie_lane_detection/fitting/observation_association.hpp"

#include <ceres/ceres.h>
#include <ceres/loss_function.h>

#include <algorithm>
#include <cmath>
#include <vector>

namespace lie_lane_detection
{

namespace
{

class EdgeObservationFunctor
{
public:
  EdgeObservationFunctor(
    double px, double py, double orientation, double soft_w, double len_w, double heading_w,
    TemplateCurve * curve)
  : px_(px),
    py_(py),
    orientation_(orientation),
    soft_w_(soft_w),
    len_w_(len_w),
    heading_w_(heading_w),
    curve_(curve)
  {
  }

  bool operator()(const double * const xi, double * residual) const
  {
    XiVector xi_e = XiVector::Zero();
    for (int i = 0; i < 5; ++i) {
      xi_e[i] = xi[i];
    }
    const Sophus::SE2d g_inv = xiToSE2(xi_e).inverse();
    const Vec2 local_p = g_inv * Vec2(px_, py_);

    XiVector xi_local = XiVector::Zero();
    xi_local[3] = xi_e[3];
    xi_local[4] = xi_e[4];

    double t = 0.0;
    curve_->nearestPoint(xi_local, local_p, &t);
    const Vec2 q = curve_->sample(xi_local, t);
    const Vec2 tau = curve_->tangentAt(xi_local, t);
    const double curve_angle = std::atan2(tau.y(), tau.x());
    const Eigen::Vector2d world_dir(std::cos(orientation_), std::sin(orientation_));
    const Eigen::Vector2d local_dir = g_inv.so2().matrix() * world_dir;
    const double edge_angle_local = std::atan2(local_dir.y(), local_dir.x());

    const double w = len_w_ * soft_w_;
    residual[0] = w * (local_p.x() - q.x());
    residual[1] = w * heading_w_ * std::sin(edge_angle_local - curve_angle);
    return true;
  }

private:
  double px_;
  double py_;
  double orientation_;
  double soft_w_;
  double len_w_;
  double heading_w_;
  TemplateCurve * curve_;
};

class LineSegmentFunctor
{
public:
  LineSegmentFunctor(
    double x1, double y1, double x2, double y2, double soft_w, double len_w, double heading_w,
    TemplateCurve * curve)
  : x1_(x1),
    y1_(y1),
    x2_(x2),
    y2_(y2),
    soft_w_(soft_w),
    len_w_(len_w),
    heading_w_(heading_w),
    curve_(curve)
  {
  }

  bool operator()(const double * const xi, double * residual) const
  {
    XiVector xi_e = XiVector::Zero();
    for (int i = 0; i < 5; ++i) {
      xi_e[i] = xi[i];
    }
    const Sophus::SE2d g_inv = xiToSE2(xi_e).inverse();
    const Vec2 lp0 = g_inv * Vec2(x1_, y1_);
    const Vec2 lp1 = g_inv * Vec2(x2_, y2_);

    XiVector xi_local = XiVector::Zero();
    xi_local[3] = xi_e[3];
    xi_local[4] = xi_e[4];

    double t0 = 0.0;
    double t1 = 0.0;
    curve_->nearestPoint(xi_local, lp0, &t0);
    curve_->nearestPoint(xi_local, lp1, &t1);
    const Vec2 q0 = curve_->sample(xi_local, t0);
    const Vec2 q1 = curve_->sample(xi_local, t1);
    const Vec2 tau0 = curve_->tangentAt(xi_local, t0);
    const Vec2 tau1 = curve_->tangentAt(xi_local, t1);
    const double curve_ang0 = std::atan2(tau0.y(), tau0.x());
    const double curve_ang1 = std::atan2(tau1.y(), tau1.x());
    const double line_ang = std::atan2(lp1.y() - lp0.y(), lp1.x() - lp0.x());

    const double w = len_w_ * soft_w_;
    residual[0] = w * (lp0.x() - q0.x());
    residual[1] = w * heading_w_ * std::sin(line_ang - curve_ang0);
    residual[2] = w * (lp1.x() - q1.x());
    residual[3] = w * heading_w_ * std::sin(line_ang - curve_ang1);
    return true;
  }

private:
  double x1_;
  double y1_;
  double x2_;
  double y2_;
  double soft_w_;
  double len_w_;
  double heading_w_;
  TemplateCurve * curve_;
};

}  // namespace

CeresLaneFitter::CeresLaneFitter(const PipelineParams & params, TemplateCurve * template_curve)
: params_(params), template_curve_(template_curve)
{
}

bool CeresLaneFitter::optimizeXi(
  XiVector & xi, const std::vector<EdgePoint> & points, const std::vector<double> & weights) const
{
  ObservationAssociation assoc(params_, template_curve_);
  std::vector<AssociatedEdge> associated;
  associated.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    if (!assoc.edgeSupportsSeed(xi, points[i])) {
      continue;
    }
    AssociatedEdge a;
    a.edge = points[i];
    a.soft_weight = i < weights.size() ? weights[i] : 1.0;
    a.length_weight =
      std::sqrt(std::max(points[i].magnitude, params_.ceres_length_weight_floor_px));
    double t = 0.0;
    template_curve_->distanceToCurve(xi, Vec2(points[i].x, points[i].y), &t);
    a.t_near = t;
    associated.push_back(a);
  }
  return optimizeXiFromEdges(xi, associated);
}

bool CeresLaneFitter::optimizeXiFromEdges(
  XiVector & xi, const std::vector<AssociatedEdge> & edges) const
{
  if (edges.empty() || template_curve_ == nullptr) {
    return false;
  }

  double xi_params[5];
  for (int i = 0; i < 5; ++i) {
    xi_params[i] = xi[i];
  }

  for (int outer = 0; outer < 3; ++outer) {
    ceres::Problem problem;
    ceres::LossFunction * loss = new ceres::HuberLoss(params_.ceres_huber_delta_px);
    for (const auto & a : edges) {
      ceres::CostFunction * cost =
        new ceres::NumericDiffCostFunction<EdgeObservationFunctor, ceres::CENTRAL, 2, 5>(
          new EdgeObservationFunctor(
            a.edge.x, a.edge.y, a.edge.orientation, a.soft_weight, a.length_weight,
            params_.ceres_heading_weight, template_curve_));
      problem.AddResidualBlock(cost, loss, xi_params);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = params_.ceres_max_iterations;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    XiVector xi_tmp = XiVector::Zero();
    for (int d = 0; d < 5; ++d) {
      xi_tmp[d] = xi_params[d];
    }
    const Sophus::SE2d g = xiToSE2(xi_tmp);
    for (int d = 0; d < 3; ++d) {
      xi_params[d] = g.log()[d];
    }
    xi_params[3] = std::clamp(xi_params[3], params_.kappa_min, params_.kappa_max);
    xi_params[4] = std::clamp(xi_params[4], params_.sigma_min, params_.sigma_max);
  }

  for (int i = 0; i < 5; ++i) {
    xi[i] = xi_params[i];
  }
  return true;
}

bool CeresLaneFitter::optimizeXiFromLines(
  XiVector & xi, const std::vector<AssociatedLine> & lines) const
{
  if (lines.empty() || template_curve_ == nullptr) {
    return false;
  }

  double xi_params[5];
  for (int i = 0; i < 5; ++i) {
    xi_params[i] = xi[i];
  }

  for (int outer = 0; outer < 3; ++outer) {
    ceres::Problem problem;
    ceres::LossFunction * loss = new ceres::HuberLoss(params_.ceres_huber_delta_px);
    for (const auto & a : lines) {
      ceres::CostFunction * cost =
        new ceres::NumericDiffCostFunction<LineSegmentFunctor, ceres::CENTRAL, 4, 5>(
          new LineSegmentFunctor(
            a.segment.x1, a.segment.y1, a.segment.x2, a.segment.y2, a.soft_weight, a.length_weight,
            params_.ceres_heading_weight, template_curve_));
      problem.AddResidualBlock(cost, loss, xi_params);
    }

    ceres::Solver::Options options;
    options.max_num_iterations = params_.ceres_max_iterations;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    XiVector xi_tmp = XiVector::Zero();
    for (int d = 0; d < 5; ++d) {
      xi_tmp[d] = xi_params[d];
    }
    const Sophus::SE2d g = xiToSE2(xi_tmp);
    for (int d = 0; d < 3; ++d) {
      xi_params[d] = g.log()[d];
    }
    xi_params[3] = std::clamp(xi_params[3], params_.kappa_min, params_.kappa_max);
    xi_params[4] = std::clamp(xi_params[4], params_.sigma_min, params_.sigma_max);
  }

  for (int i = 0; i < 5; ++i) {
    xi[i] = xi_params[i];
  }
  return true;
}

LaneHypothesis CeresLaneFitter::fitEdges(
  const LaneHypothesis & seed, const std::vector<EdgePoint> & edges) const
{
  LaneHypothesis out = seed;
  ObservationAssociation assoc(params_, template_curve_);
  const auto candidates = assoc.associateEdges(seed.xi, edges);
  if (candidates.size() < static_cast<size_t>(std::max(3, params_.min_inliers / 3))) {
    return out;
  }

  XiVector xi = seed.xi;
  if (optimizeXiFromEdges(xi, candidates)) {
    out.xi = xi;
    out.polyline = template_curve_->samplePolyline(out.xi);
    int inliers = 0;
    for (const auto & a : candidates) {
      if (
        template_curve_->distanceToCurve(out.xi, Vec2(a.edge.x, a.edge.y)) <
        params_.inlier_threshold_px) {
        ++inliers;
      }
    }
    out.inlier_ratio = static_cast<double>(inliers) / static_cast<double>(candidates.size());
    out.score = seed.score * std::max(0.01, out.inlier_ratio);
    out.supporting_edges.reserve(candidates.size());
    for (const auto & a : candidates) {
      out.supporting_edges.push_back(a.edge);
    }
  }
  return out;
}

LaneHypothesis CeresLaneFitter::fitLines(
  const LaneHypothesis & seed, const std::vector<LineSegment> & lines) const
{
  LaneHypothesis out = seed;
  ObservationAssociation assoc(params_, template_curve_);
  const auto candidates = assoc.associateLines(seed.xi, lines);
  if (candidates.empty()) {
    return seed;
  }
  if (candidates.size() < static_cast<size_t>(std::max(2, params_.min_inliers / 4))) {
    return out;
  }

  XiVector xi = seed.xi;
  if (optimizeXiFromLines(xi, candidates)) {
    out.xi = xi;
    out.polyline = template_curve_->samplePolyline(out.xi);
    int inliers = 0;
    for (const auto & a : candidates) {
      if (
        template_curve_->segmentDistanceToCurve(out.xi, a.segment) < params_.inlier_threshold_px) {
        ++inliers;
      }
    }
    out.inlier_ratio = static_cast<double>(inliers) / static_cast<double>(candidates.size());
    out.score = seed.score * std::max(0.01, out.inlier_ratio);
  }
  return out;
}

}  // namespace lie_lane_detection
