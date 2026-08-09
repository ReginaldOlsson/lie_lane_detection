#include "lie_lane_detection/fitting/observation_association.hpp"

#include <algorithm>
#include <cmath>

namespace lie_lane_detection
{

ObservationAssociation::ObservationAssociation(
  const PipelineParams & params, TemplateCurve * template_curve)
: params_(params), template_curve_(template_curve)
{
}

bool ObservationAssociation::lineSupportsSeed(
  const XiVector & seed_xi, const LineSegment & line) const
{
  if (template_curve_ == nullptr) {
    return false;
  }
  if (
    params_.use_longitudinal_line_filter &&
    !isLongitudinalSegment(line.angle, params_.longitudinal_max_deviation_rad)) {
    return false;
  }

  const Sophus::SE2d g_inv = xiToSE2(seed_xi).inverse();
  const Vec2 p0 = g_inv * Vec2(line.x1, line.y1);
  const Vec2 p1 = g_inv * Vec2(line.x2, line.y2);
  const Vec2 pm = g_inv * Vec2(line.mx, line.my);

  if (
    params_.ceres_reject_corridor_center &&
    std::abs(pm.x()) < params_.ceres_corridor_half_width_px) {
    return false;
  }

  XiVector xi_local = XiVector::Zero();
  xi_local[3] = seed_xi[3];
  xi_local[4] = seed_xi[4];

  LineSegment local_line = line;
  local_line.x1 = p0.x();
  local_line.y1 = p0.y();
  local_line.x2 = p1.x();
  local_line.y2 = p1.y();
  local_line.mx = pm.x();
  local_line.my = pm.y();

  const double dist = template_curve_->segmentDistanceToCurve(xi_local, local_line);
  if (dist > params_.ceres_hard_gate_dist_px) {
    return false;
  }

  double t = 0.0;
  template_curve_->nearestPoint(xi_local, pm, &t);
  const Vec2 tau = template_curve_->tangentAt(xi_local, t);
  const double curve_angle = std::atan2(tau.y(), tau.x());
  const double local_line_angle = std::atan2(p1.y() - p0.y(), p1.x() - p0.x());
  return angleDiffRad(curve_angle, local_line_angle) <= params_.ceres_hard_gate_angle_rad;
}

bool ObservationAssociation::edgeSupportsSeed(
  const XiVector & seed_xi, const EdgePoint & edge) const
{
  if (template_curve_ == nullptr) {
    return false;
  }
  if (
    params_.use_longitudinal_line_filter &&
    !isLongitudinalSegment(edge.orientation, params_.longitudinal_max_deviation_rad)) {
    return false;
  }

  const Vec2 p(edge.x, edge.y);
  const double dist = template_curve_->distanceToCurve(seed_xi, p);
  if (dist > params_.ceres_hard_gate_dist_px) {
    return false;
  }

  const Sophus::SE2d g_inv = xiToSE2(seed_xi).inverse();
  const Vec2 local_p = g_inv * p;
  if (
    params_.ceres_reject_corridor_center &&
    std::abs(local_p.x()) < params_.ceres_corridor_half_width_px) {
    return false;
  }

  XiVector xi_local = XiVector::Zero();
  xi_local[3] = seed_xi[3];
  xi_local[4] = seed_xi[4];
  double t = 0.0;
  template_curve_->nearestPoint(xi_local, local_p, &t);
  const Vec2 tau = template_curve_->tangentAt(xi_local, t);
  const double curve_angle = std::atan2(tau.y(), tau.x());
  const Eigen::Vector2d world_dir(std::cos(edge.orientation), std::sin(edge.orientation));
  const Eigen::Vector2d local_dir = g_inv.so2().matrix() * world_dir;
  const double edge_angle_local = std::atan2(local_dir.y(), local_dir.x());
  return angleDiffRad(curve_angle, edge_angle_local) <= params_.ceres_hard_gate_angle_rad;
}

std::vector<AssociatedLine> ObservationAssociation::associateLines(
  const XiVector & seed_xi, const std::vector<LineSegment> & lines) const
{
  std::vector<AssociatedLine> out;
  out.reserve(lines.size());
  const double sigma_sq = std::max(1.0, params_.soft_vote_sigma_px * params_.soft_vote_sigma_px);
  const double len_floor = std::max(1.0, params_.ceres_length_weight_floor_px);

  for (const auto & line : lines) {
    if (!lineSupportsSeed(seed_xi, line)) {
      continue;
    }
    const double d = template_curve_->segmentDistanceToCurve(seed_xi, line);
    AssociatedLine a;
    a.segment = line;
    a.soft_weight = std::max(1e-3, line.length * std::exp(-d * d / (2.0 * sigma_sq)));
    a.length_weight = std::sqrt(std::max(line.length, len_floor));
    out.push_back(a);
  }
  return out;
}

std::vector<AssociatedEdge> ObservationAssociation::associateEdges(
  const XiVector & seed_xi, const std::vector<EdgePoint> & edges) const
{
  std::vector<AssociatedEdge> out;
  out.reserve(edges.size());
  const double sigma_sq = std::max(1.0, params_.soft_vote_sigma_px * params_.soft_vote_sigma_px);
  const double len_floor = std::sqrt(std::max(1.0, params_.ceres_length_weight_floor_px));

  for (const auto & e : edges) {
    if (!edgeSupportsSeed(seed_xi, e)) {
      continue;
    }
    const Vec2 p(e.x, e.y);
    double t_near = 0.0;
    const double d = template_curve_->distanceToCurve(seed_xi, p, &t_near);
    AssociatedEdge a;
    a.edge = e;
    a.t_near = t_near;
    a.soft_weight = std::max(1e-3, e.magnitude * std::exp(-d * d / (2.0 * sigma_sq)));
    a.length_weight = std::max(len_floor, std::sqrt(std::max(e.magnitude, 1.0)));
    out.push_back(a);
  }
  return out;
}

}  // namespace lie_lane_detection
