#include "lie_lane_detection/fitting/road_manifold_fitter.hpp"

#include <ceres/ceres.h>
#include <ceres/loss_function.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace lie_lane_detection
{

namespace
{

constexpr int kRoadParamSize = 6;

class RoadLineSegmentFunctor
{
public:
  RoadLineSegmentFunctor(
    int lane_slot, double x1, double y1, double x2, double y2, double soft_w, double len_w,
    double heading_w, TemplateCurve * curve)
  : lane_slot_(lane_slot),
    x1_(x1),
    y1_(y1),
    x2_(x2),
    y2_(y2),
    soft_w_(soft_w),
    len_w_(len_w),
    heading_w_(heading_w),
    curve_(curve)
  {
  }

  bool operator()(const double * const road, double * residual) const
  {
    const double * ego = road;
    const double kappa = road[3];
    const double sigma = road[4];
    const double w_lane = road[5];

    XiVector xi_e = RoadManifoldFitter::decodeLaneXi(ego, lane_slot_, kappa, sigma, w_lane);
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
  int lane_slot_;
  double x1_;
  double y1_;
  double x2_;
  double y2_;
  double soft_w_;
  double len_w_;
  double heading_w_;
  TemplateCurve * curve_;
};

class RoadEdgeFunctor
{
public:
  RoadEdgeFunctor(
    int lane_slot, double px, double py, double orientation, double soft_w, double len_w,
    double heading_w, TemplateCurve * curve)
  : lane_slot_(lane_slot),
    px_(px),
    py_(py),
    orientation_(orientation),
    soft_w_(soft_w),
    len_w_(len_w),
    heading_w_(heading_w),
    curve_(curve)
  {
  }

  bool operator()(const double * const road, double * residual) const
  {
    const double * ego = road;
    const double kappa = road[3];
    const double sigma = road[4];
    const double w_lane = road[5];

    XiVector xi_e = RoadManifoldFitter::decodeLaneXi(ego, lane_slot_, kappa, sigma, w_lane);
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
  int lane_slot_;
  double px_;
  double py_;
  double orientation_;
  double soft_w_;
  double len_w_;
  double heading_w_;
  TemplateCurve * curve_;
};

class LaneWidthPriorFunctor
{
public:
  LaneWidthPriorFunctor(double w_init, double weight) : w_init_(w_init), weight_(weight) {}

  bool operator()(const double * const road, double * residual) const
  {
    residual[0] = weight_ * (road[5] - w_init_);
    return true;
  }

private:
  double w_init_;
  double weight_;
};

}  // namespace

RoadManifoldFitter::RoadManifoldFitter(
  const PipelineParams & params, TemplateCurve * template_curve)
: params_(params), template_curve_(template_curve), ceres_fitter_(params, template_curve)
{
}

double RoadManifoldFitter::laneOffsetMultiplier(int lane_index)
{
  static constexpr double kMul[] = {-1.5, -0.5, 0.5, 1.5};
  const int idx = std::clamp(lane_index, 0, 3);
  return kMul[idx];
}

int RoadManifoldFitter::laneSlotForRank(int rank, int total_lanes)
{
  if (total_lanes <= 1) {
    return 1;
  }
  if (total_lanes == 2) {
    return rank == 0 ? 1 : 2;
  }
  if (total_lanes == 3) {
    return rank + 1;
  }
  return std::clamp(rank, 0, 3);
}

XiVector RoadManifoldFitter::decodeLaneXi(
  const double * ego_xi, int lane_index, double kappa, double sigma, double w_lane)
{
  Sophus::SE2d::Tangent ego_tangent;
  ego_tangent << ego_xi[0], ego_xi[1], ego_xi[2];
  Sophus::SE2d::Tangent offset_tangent;
  offset_tangent << laneOffsetMultiplier(lane_index) * w_lane, 0.0, 0.0;
  const Sophus::SE2d T = Sophus::SE2d::exp(ego_tangent) * Sophus::SE2d::exp(offset_tangent);

  XiVector xi = XiVector::Zero();
  const auto log_t = T.log();
  xi[0] = log_t[0];
  xi[1] = log_t[1];
  xi[2] = log_t[2];
  xi[3] = kappa;
  xi[4] = sigma;
  return xi;
}

std::vector<RoadManifoldFitter::LaneAssignment> RoadManifoldFitter::assignLanes(
  const std::vector<LaneHypothesis> & lanes, const std::vector<LineSegment> & lines) const
{
  std::vector<LaneAssignment> assignments(lanes.size());
  ObservationAssociation assoc(params_, template_curve_);

  std::vector<size_t> order(lanes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return lanes[a].xi[0] < lanes[b].xi[0];
  });

  for (size_t rank = 0; rank < order.size(); ++rank) {
    const size_t li = order[rank];
    const int slot = laneSlotForRank(static_cast<int>(rank), static_cast<int>(lanes.size()));
    assignments[li].lane_slot = slot;
    assignments[li].lines = assoc.associateLines(lanes[li].xi, lines);
  }
  return assignments;
}

std::vector<RoadManifoldFitter::LaneAssignment> RoadManifoldFitter::assignLanes(
  const std::vector<LaneHypothesis> & lanes, const std::vector<EdgePoint> & edges) const
{
  std::vector<LaneAssignment> assignments(lanes.size());
  ObservationAssociation assoc(params_, template_curve_);

  std::vector<size_t> order(lanes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return lanes[a].xi[0] < lanes[b].xi[0];
  });

  for (size_t rank = 0; rank < order.size(); ++rank) {
    const size_t li = order[rank];
    const int slot = laneSlotForRank(static_cast<int>(rank), static_cast<int>(lanes.size()));
    assignments[li].lane_slot = slot;
    assignments[li].edges = assoc.associateEdges(lanes[li].xi, edges);
  }
  return assignments;
}

bool RoadManifoldFitter::optimizeRoad(
  const std::vector<LaneHypothesis> & seeds, const std::vector<LaneAssignment> & assignments,
  double * road_params) const
{
  if (template_curve_ == nullptr) {
    return false;
  }

  ceres::Problem problem;
  ceres::LossFunction * loss = new ceres::HuberLoss(params_.ceres_huber_delta_px);
  int residual_blocks = 0;

  for (size_t i = 0; i < assignments.size(); ++i) {
    const int slot = assignments[i].lane_slot;
    for (const auto & a : assignments[i].lines) {
      ceres::CostFunction * cost = new ceres::NumericDiffCostFunction<
        RoadLineSegmentFunctor, ceres::CENTRAL, 4, kRoadParamSize>(new RoadLineSegmentFunctor(
        slot, a.segment.x1, a.segment.y1, a.segment.x2, a.segment.y2, a.soft_weight,
        a.length_weight, params_.ceres_heading_weight, template_curve_));
      problem.AddResidualBlock(cost, loss, road_params);
      ++residual_blocks;
    }
    for (const auto & a : assignments[i].edges) {
      ceres::CostFunction * cost =
        new ceres::NumericDiffCostFunction<RoadEdgeFunctor, ceres::CENTRAL, 2, kRoadParamSize>(
          new RoadEdgeFunctor(
            slot, a.edge.x, a.edge.y, a.edge.orientation, a.soft_weight, a.length_weight,
            params_.ceres_heading_weight, template_curve_));
      problem.AddResidualBlock(cost, loss, road_params);
      ++residual_blocks;
    }
    (void)i;
    (void)seeds;
  }

  if (residual_blocks == 0) {
    return false;
  }

  ceres::CostFunction * prior =
    new ceres::NumericDiffCostFunction<LaneWidthPriorFunctor, ceres::CENTRAL, 1, kRoadParamSize>(
      new LaneWidthPriorFunctor(road_params[5], 0.05));
  problem.AddResidualBlock(prior, nullptr, road_params);

  problem.SetParameterLowerBound(road_params, 5, params_.min_lane_separation_px);
  problem.SetParameterUpperBound(road_params, 5, params_.min_lane_separation_px * 4.0);
  problem.SetParameterLowerBound(road_params, 3, params_.kappa_min);
  problem.SetParameterUpperBound(road_params, 3, params_.kappa_max);
  problem.SetParameterLowerBound(road_params, 4, params_.sigma_min);
  problem.SetParameterUpperBound(road_params, 4, params_.sigma_max);

  ceres::Solver::Options options;
  options.max_num_iterations = params_.ceres_max_iterations;
  options.linear_solver_type = ceres::DENSE_QR;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  Sophus::SE2d::Tangent ego_tangent;
  ego_tangent << road_params[0], road_params[1], road_params[2];
  const auto log_t = Sophus::SE2d::exp(ego_tangent).log();
  road_params[0] = log_t[0];
  road_params[1] = log_t[1];
  road_params[2] = log_t[2];
  return summary.IsSolutionUsable();
}

bool RoadManifoldFitter::refine(
  std::vector<LaneHypothesis> & lanes, const std::vector<LineSegment> & lines) const
{
  if (!params_.use_road_manifold_joint || lanes.size() < 2 || template_curve_ == nullptr) {
    return false;
  }

  const auto assignments = assignLanes(lanes, lines);
  size_t obs_count = 0;
  for (const auto & a : assignments) {
    obs_count += a.lines.size();
  }
  if (obs_count < static_cast<size_t>(std::max(4, params_.min_inliers / 2))) {
    return false;
  }

  size_t best_idx = 0;
  double best_score = -1.0;
  for (size_t i = 0; i < lanes.size(); ++i) {
    if (lanes[i].score > best_score) {
      best_score = lanes[i].score;
      best_idx = i;
    }
  }

  std::vector<size_t> order(lanes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return lanes[a].xi[0] < lanes[b].xi[0];
  });

  double w_init = params_.min_lane_separation_px * 2.0;
  if (lanes.size() == 2) {
    w_init = std::abs(lanes[order[1]].xi[0] - lanes[order[0]].xi[0]);
  } else if (lanes.size() >= 2) {
    std::vector<double> spacings;
    for (size_t i = 1; i < order.size(); ++i) {
      const int s0 = laneSlotForRank(static_cast<int>(i - 1), static_cast<int>(lanes.size()));
      const int s1 = laneSlotForRank(static_cast<int>(i), static_cast<int>(lanes.size()));
      const double slot_delta = std::abs(laneOffsetMultiplier(s1) - laneOffsetMultiplier(s0));
      if (slot_delta > 1e-6) {
        spacings.push_back(
          std::abs(lanes[order[i]].xi[0] - lanes[order[i - 1]].xi[0]) / slot_delta);
      }
    }
    if (!spacings.empty()) {
      std::nth_element(spacings.begin(), spacings.begin() + spacings.size() / 2, spacings.end());
      w_init = std::max(params_.min_lane_separation_px, spacings[spacings.size() / 2]);
    }
  }

  double road_params[kRoadParamSize];
  double lat_mean = 0.0;
  double y_mean = 0.0;
  double omega_mean = 0.0;
  for (const auto & lane : lanes) {
    lat_mean += lane.xi[0];
    y_mean += lane.xi[1];
    omega_mean += lane.xi[2];
  }
  const double inv_n = 1.0 / static_cast<double>(lanes.size());
  road_params[0] = lat_mean * inv_n;
  road_params[1] = y_mean * inv_n;
  road_params[2] = omega_mean * inv_n;
  (void)best_idx;

  double k_sum = 0.0;
  double s_sum = 0.0;
  double w_sum = 0.0;
  for (const auto & lane : lanes) {
    const double lw = std::max(1.0, lane.score);
    k_sum += lane.xi[3] * lw;
    s_sum += lane.xi[4] * lw;
    w_sum += lw;
  }
  road_params[3] = w_sum > 0.0 ? k_sum / w_sum : 0.0;
  road_params[4] = w_sum > 0.0 ? s_sum / w_sum : 0.0;
  road_params[5] = w_init;

  if (!optimizeRoad(lanes, assignments, road_params)) {
    return false;
  }

  for (size_t i = 0; i < lanes.size(); ++i) {
    const int slot = assignments[i].lane_slot;
    lanes[i].xi = decodeLaneXi(road_params, slot, road_params[3], road_params[4], road_params[5]);
    lanes[i].polyline = template_curve_->samplePolyline(lanes[i].xi);
  }
  return true;
}

bool RoadManifoldFitter::refine(
  std::vector<LaneHypothesis> & lanes, const std::vector<EdgePoint> & edges) const
{
  if (!params_.use_road_manifold_joint || lanes.size() < 2 || template_curve_ == nullptr) {
    return false;
  }

  const auto assignments = assignLanes(lanes, edges);
  size_t obs_count = 0;
  for (const auto & a : assignments) {
    obs_count += a.edges.size();
  }
  if (obs_count < static_cast<size_t>(std::max(4, params_.min_inliers / 2))) {
    return false;
  }

  size_t best_idx = 0;
  double best_score = -1.0;
  for (size_t i = 0; i < lanes.size(); ++i) {
    if (lanes[i].score > best_score) {
      best_score = lanes[i].score;
      best_idx = i;
    }
  }

  std::vector<size_t> order(lanes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
    return lanes[a].xi[0] < lanes[b].xi[0];
  });

  double w_init = params_.min_lane_separation_px * 2.0;
  if (lanes.size() == 2) {
    w_init = std::abs(lanes[order[1]].xi[0] - lanes[order[0]].xi[0]);
  } else if (lanes.size() >= 2) {
    std::vector<double> spacings;
    for (size_t i = 1; i < order.size(); ++i) {
      const int s0 = laneSlotForRank(static_cast<int>(i - 1), static_cast<int>(lanes.size()));
      const int s1 = laneSlotForRank(static_cast<int>(i), static_cast<int>(lanes.size()));
      const double slot_delta = std::abs(laneOffsetMultiplier(s1) - laneOffsetMultiplier(s0));
      if (slot_delta > 1e-6) {
        spacings.push_back(
          std::abs(lanes[order[i]].xi[0] - lanes[order[i - 1]].xi[0]) / slot_delta);
      }
    }
    if (!spacings.empty()) {
      std::nth_element(spacings.begin(), spacings.begin() + spacings.size() / 2, spacings.end());
      w_init = std::max(params_.min_lane_separation_px, spacings[spacings.size() / 2]);
    }
  }

  double road_params[kRoadParamSize];
  double lat_mean = 0.0;
  double y_mean = 0.0;
  double omega_mean = 0.0;
  for (const auto & lane : lanes) {
    lat_mean += lane.xi[0];
    y_mean += lane.xi[1];
    omega_mean += lane.xi[2];
  }
  const double inv_n = 1.0 / static_cast<double>(lanes.size());
  road_params[0] = lat_mean * inv_n;
  road_params[1] = y_mean * inv_n;
  road_params[2] = omega_mean * inv_n;
  (void)best_idx;

  double k_sum = 0.0;
  double s_sum = 0.0;
  double w_sum = 0.0;
  for (const auto & lane : lanes) {
    const double lw = std::max(1.0, lane.score);
    k_sum += lane.xi[3] * lw;
    s_sum += lane.xi[4] * lw;
    w_sum += lw;
  }
  road_params[3] = w_sum > 0.0 ? k_sum / w_sum : 0.0;
  road_params[4] = w_sum > 0.0 ? s_sum / w_sum : 0.0;
  road_params[5] = w_init;

  if (!optimizeRoad(lanes, assignments, road_params)) {
    return false;
  }

  for (size_t i = 0; i < lanes.size(); ++i) {
    const int slot = assignments[i].lane_slot;
    lanes[i].xi = decodeLaneXi(road_params, slot, road_params[3], road_params[4], road_params[5]);
    lanes[i].polyline = template_curve_->samplePolyline(lanes[i].xi);
  }
  return true;
}

}  // namespace lie_lane_detection
