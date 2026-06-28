#include "lie_lane_detection/fitting/manifold_ransac.hpp"

#include <algorithm>
#include <cmath>
#include <random>

#include "lie_lane_detection/common/parallel.hpp"
#include "lie_lane_detection/fitting/ceres_lane_fitter.hpp"

namespace lie_lane_detection
{

ManifoldRansac::ManifoldRansac(const PipelineParams & params, TemplateCurve * template_curve)
: params_(params), template_curve_(template_curve)
{
}

std::vector<EdgePoint> ManifoldRansac::collectCandidates(
  const XiVector & xi,
  const std::vector<EdgePoint> & edges) const
{
  std::vector<EdgePoint> candidates;
  const double gate = params_.inlier_threshold_px * 3.0;
  for (const auto & e : edges) {
    const Vec2 p(e.x, e.y);
    if (template_curve_->distanceToCurve(xi, p) < gate) {
      candidates.push_back(e);
    }
  }
  return candidates;
}

bool ManifoldRansac::fitMinimal(const std::vector<EdgePoint> & sample, XiVector & xi_out) const
{
  if (sample.size() < 3) {
    return false;
  }

  std::vector<Vec2> source;
  std::vector<Vec2> target;
  source.reserve(sample.size());
  target.reserve(sample.size());

  for (size_t i = 0; i < sample.size(); ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(sample.size() - 1);
    source.push_back(template_curve_->deformTemplate(t, 0.0, 0.0));
    target.emplace_back(sample[i].x, sample[i].y);
  }

  Vec2 src_mean = Vec2::Zero();
  Vec2 tgt_mean = Vec2::Zero();
  for (size_t i = 0; i < source.size(); ++i) {
    src_mean += source[i];
    tgt_mean += target[i];
  }
  src_mean /= static_cast<double>(source.size());
  tgt_mean /= static_cast<double>(source.size());

  double num = 0.0;
  double den = 0.0;
  for (size_t i = 0; i < source.size(); ++i) {
    const Vec2 ps = source[i] - src_mean;
    const Vec2 pt = target[i] - tgt_mean;
    num += ps.x() * pt.y() - ps.y() * pt.x();
    den += ps.squaredNorm();
  }
  if (std::abs(den) < 1e-9) {
    return false;
  }
  const double omega = std::atan2(num, den);
  const double c = std::cos(omega);
  const double s = std::sin(omega);
  const Vec2 rotated_src_mean(c * src_mean.x() - s * src_mean.y(), s * src_mean.x() + c * src_mean.y());
  const Vec2 trans = tgt_mean - rotated_src_mean;

  const double keep_kappa = xi_out[3];
  const double keep_sigma = xi_out[4];
  xi_out = XiVector::Zero();
  xi_out[0] = trans.x();
  xi_out[1] = trans.y();
  xi_out[2] = omega;
  xi_out[3] = keep_kappa;
  xi_out[4] = keep_sigma;
  return true;
}

int ManifoldRansac::countInliers(
  const XiVector & xi,
  const std::vector<EdgePoint> & edges,
  std::vector<bool> * mask) const
{
  int count = 0;
  if (mask) {
    mask->assign(edges.size(), false);
  }
  for (size_t i = 0; i < edges.size(); ++i) {
    const Vec2 p(edges[i].x, edges[i].y);
    if (template_curve_->distanceToCurve(xi, p) < params_.inlier_threshold_px) {
      ++count;
      if (mask) {
        (*mask)[i] = true;
      }
    }
  }
  return count;
}

void ManifoldRansac::refineGaussNewton(XiVector & xi, const std::vector<EdgePoint> & inliers) const
{
  if (inliers.empty()) {
    return;
  }

  if (params_.use_ceres_fitter) {
    CeresLaneFitter fitter(params_, template_curve_);
    LaneHypothesis seed;
    seed.xi = xi;
    seed.score = 1.0;
    const LaneHypothesis refined = fitter.fitEdges(seed, inliers);
    xi = refined.xi;
    return;
  }

  const int kMaxIter = (std::abs(xi[3]) > 0.03 || std::abs(xi[4]) > 0.05) ? 22 : 15;
  constexpr double kLambda = 1e-2;
  for (int iter = 0; iter < kMaxIter; ++iter) {
    Eigen::Matrix<double, 5, 5> H = Eigen::Matrix<double, 5, 5>::Zero();
    Eigen::Matrix<double, 5, 1> b = Eigen::Matrix<double, 5, 1>::Zero();

    for (const auto & e : inliers) {
      const Vec2 p(e.x, e.y);
      double t_near = 0.0;
      template_curve_->distanceToCurve(xi, p, &t_near);
      const Vec2 q = template_curve_->sample(xi, t_near);
      const Eigen::Vector2d residual = p - q;

      Eigen::Matrix<double, 2, 5> J = Eigen::Matrix<double, 2, 5>::Zero();
      constexpr double eps = 1e-3;
      for (int d = 0; d < 5; ++d) {
        XiVector xi_eps = xi;
        xi_eps[d] += eps;
        const Vec2 q_eps = template_curve_->sample(xi_eps, t_near);
        J.col(d) = (q_eps - q) / eps;
      }

      H += J.transpose() * J;
      b += J.transpose() * residual;
    }

    H += kLambda * Eigen::Matrix<double, 5, 5>::Identity();
    Eigen::Matrix<double, 5, 1> delta = H.ldlt().solve(b);
    if (!delta.allFinite()) {
      break;
    }

    constexpr double kMaxStep = 0.5;
    if (delta.norm() > kMaxStep) {
      delta *= kMaxStep / delta.norm();
    }

    Sophus::SE2d g = xiToSE2(xi);
    Sophus::SE2d::Tangent se2_delta;
    se2_delta << delta[0], delta[1], delta[2];
    g = Sophus::SE2d::exp(se2_delta) * g;
    const auto se2_xi = g.log();
    xi[0] = se2_xi[0];
    xi[1] = se2_xi[1];
    xi[2] = se2_xi[2];
    xi[3] += delta[3];
    xi[4] += delta[4];
    xi[3] = std::clamp(xi[3], params_.kappa_min, params_.kappa_max);
    xi[4] = std::clamp(xi[4], params_.sigma_min, params_.sigma_max);

    if (delta.norm() < 1e-4) {
      break;
    }
  }
}

LaneHypothesis ManifoldRansac::fit(
  const LaneHypothesis & seed,
  const std::vector<EdgePoint> & edges) const
{
  LaneHypothesis best = seed;
  const std::vector<EdgePoint> candidates = collectCandidates(seed.xi, edges);
  if (candidates.size() < static_cast<size_t>(params_.min_inliers)) {
    best.polyline = template_curve_->samplePolyline(best.xi);
    return best;
  }

  struct RansacBest
  {
    int inliers{0};
    XiVector xi{XiVector::Zero()};
    std::vector<bool> mask;
  };

  const RansacBest best_state = tbb::parallel_reduce(
    tbb::blocked_range<int>(0, params_.ransac_iterations),
    RansacBest{},
    [&](const tbb::blocked_range<int> & range, RansacBest local) {
      std::mt19937 rng(static_cast<unsigned>(42 + range.begin()));
      for (int it = range.begin(); it != range.end(); ++it) {
        std::vector<EdgePoint> sample;
        std::sample(
          candidates.begin(), candidates.end(), std::back_inserter(sample), 5,
          rng);
        if (sample.size() < 5) {
          continue;
        }

        XiVector xi_try = seed.xi;
        if (!fitMinimal(sample, xi_try)) {
          continue;
        }

        std::vector<bool> mask;
        const int inliers = countInliers(xi_try, candidates, &mask);
        if (inliers > local.inliers) {
          local.inliers = inliers;
          local.xi = xi_try;
          local.mask = std::move(mask);
        }
      }
      return local;
    },
    [](RansacBest a, const RansacBest & b) {
      if (b.inliers > a.inliers) {
        return b;
      }
      return a;
    });

  int best_inliers = best_state.inliers;
  XiVector best_xi = best_state.xi;
  std::vector<bool> best_mask = best_state.mask;

  std::vector<EdgePoint> inlier_points;
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (i < best_mask.size() && best_mask[i]) {
      inlier_points.push_back(candidates[i]);
    }
  }
  refineGaussNewton(best_xi, inlier_points);

  std::vector<bool> refined_mask;
  best_inliers = countInliers(best_xi, candidates, &refined_mask);
  inlier_points.clear();
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (i < refined_mask.size() && refined_mask[i]) {
      inlier_points.push_back(candidates[i]);
    }
  }
  if (inlier_points.size() >= static_cast<size_t>(params_.min_inliers)) {
    refineGaussNewton(best_xi, inlier_points);
    best_inliers = countInliers(best_xi, candidates, &refined_mask);
    inlier_points.clear();
    for (size_t i = 0; i < candidates.size(); ++i) {
      if (i < refined_mask.size() && refined_mask[i]) {
        inlier_points.push_back(candidates[i]);
      }
    }
  }

  best.xi = best_xi;
  best.inlier_ratio = candidates.empty() ? 0.0 :
    static_cast<double>(best_inliers) / static_cast<double>(candidates.size());
  best.score = seed.score * best.inlier_ratio;
  best.polyline = template_curve_->samplePolyline(best.xi);
  best.inlier_mask = refined_mask;
  best.supporting_edges = inlier_points;
  return best;
}

}  // namespace lie_lane_detection
