#include "lie_lane_detection/fitting/line_segment_ransac_gate.hpp"

#include <algorithm>
#include <random>

namespace lie_lane_detection
{

LineSegmentRansacGate::LineSegmentRansacGate(
  const PipelineParams & params, TemplateCurve * template_curve)
: params_(params),
  template_curve_(template_curve),
  association_(params, template_curve)
{
}

bool LineSegmentRansacGate::fitFromTwoLines(
  const LineSegment & a,
  const LineSegment & b,
  XiVector & xi_out) const
{
  if (template_curve_ == nullptr) {
    return false;
  }

  std::vector<EdgePoint> sample;
  sample.reserve(3);
  const auto add_mid = [&](const LineSegment & seg) {
      EdgePoint e;
      e.x = seg.mx;
      e.y = seg.my;
      e.magnitude = seg.length;
      e.orientation = seg.angle;
      sample.push_back(e);
    };
  add_mid(a);
  add_mid(b);
  EdgePoint mid;
  mid.x = 0.5 * (a.mx + b.mx);
  mid.y = 0.5 * (a.my + b.my);
  mid.magnitude = 0.5 * (a.length + b.length);
  mid.orientation = 0.5 * (a.angle + b.angle);
  sample.push_back(mid);

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
  tgt_mean /= static_cast<double>(target.size());

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
  const Vec2 rotated_src_mean(
    c * src_mean.x() - s * src_mean.y(), s * src_mean.x() + c * src_mean.y());
  const Vec2 trans = tgt_mean - rotated_src_mean;

  xi_out = XiVector::Zero();
  xi_out[0] = trans.x();
  xi_out[1] = trans.y();
  xi_out[2] = omega;
  return true;
}

int LineSegmentRansacGate::countInliers(
  const XiVector & xi,
  const std::vector<LineSegment> & candidates,
  std::vector<bool> * mask) const
{
  int count = 0;
  if (mask) {
    mask->assign(candidates.size(), false);
  }
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (association_.lineSupportsSeed(xi, candidates[i])) {
      ++count;
      if (mask) {
        (*mask)[i] = true;
      }
    }
  }
  return count;
}

RansacGateResult LineSegmentRansacGate::filterSeed(
  const LaneHypothesis & seed,
  const std::vector<LineSegment> & lines) const
{
  RansacGateResult result;
  result.xi = seed.xi;

  if (!params_.use_line_ransac_init_gate || lines.empty() || template_curve_ == nullptr) {
    result.inlier_lines = lines;
    result.valid = !lines.empty();
    result.inlier_ratio = 1.0;
    return result;
  }

  std::vector<LineSegment> candidates;
  candidates.reserve(lines.size());
  const double soft_gate = params_.inlier_threshold_px * 3.0;
  for (const auto & line : lines) {
    if (params_.use_longitudinal_line_filter &&
      !isLongitudinalSegment(line.angle, params_.longitudinal_max_deviation_rad))
    {
      continue;
    }
    if (template_curve_->segmentDistanceToCurve(seed.xi, line) < soft_gate) {
      candidates.push_back(line);
    }
  }

  if (candidates.size() < 2) {
    result.valid = false;
    return result;
  }

  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);

  XiVector best_xi = seed.xi;
  int best_inliers = 0;
  std::vector<bool> best_mask;

  for (int iter = 0; iter < params_.line_ransac_init_iterations; ++iter) {
    const size_t i0 = dist(rng);
    size_t i1 = dist(rng);
    if (i1 == i0) {
      i1 = (i0 + 1) % candidates.size();
    }

    XiVector trial = seed.xi;
    if (!fitFromTwoLines(candidates[i0], candidates[i1], trial)) {
      continue;
    }
    trial[3] = seed.xi[3];
    trial[4] = seed.xi[4];

    std::vector<bool> mask;
    const int inliers = countInliers(trial, candidates, &mask);
    if (inliers > best_inliers) {
      best_inliers = inliers;
      best_xi = trial;
      best_mask = std::move(mask);
    }
  }

  if (best_inliers < std::max(2, params_.min_inliers / 4)) {
    result.valid = false;
    return result;
  }

  result.xi = best_xi;
  result.inlier_lines.reserve(candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    if (best_mask[i]) {
      result.inlier_lines.push_back(candidates[i]);
    }
  }
  result.inlier_ratio = static_cast<double>(best_inliers) / static_cast<double>(candidates.size());
  result.valid = result.inlier_ratio >= params_.min_inlier_ratio * 0.5 &&
    static_cast<int>(result.inlier_lines.size()) >= std::max(2, params_.min_inliers / 4);
  return result;
}

}  // namespace lie_lane_detection
