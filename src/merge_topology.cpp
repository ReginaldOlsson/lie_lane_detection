#include "lie_lane_detection/merge_topology.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace lie_lane_detection
{

MergeTopology::MergeTopology(const PipelineParams & params, TemplateCurve * template_curve)
: params_(params), template_curve_(template_curve)
{
}

MergeTopologyType MergeTopology::classifyPair(
  const LaneHypothesis & a,
  const LaneHypothesis & b,
  Vec2 * merge_point) const
{
  constexpr int kSamples = 20;
  std::vector<double> deltas;
  deltas.reserve(kSamples);
  double y_near = std::numeric_limits<double>::max();
  double y_far = std::numeric_limits<double>::lowest();

  for (int i = 0; i < kSamples; ++i) {
    const double t = static_cast<double>(i) / static_cast<double>(kSamples - 1);
    const Vec2 pa = template_curve_->sample(a.xi, t);
    const Vec2 pb = template_curve_->sample(b.xi, t);
    deltas.push_back(std::abs(pa.x() - pb.x()));
    y_near = std::min(y_near, std::min(pa.y(), pb.y()));
    y_far = std::max(y_far, std::max(pa.y(), pb.y()));
  }

  const double delta_near = deltas.front();
  const double delta_far = deltas.back();

  bool monotonic_decrease = true;
  bool monotonic_increase = true;
  for (size_t i = 1; i < deltas.size(); ++i) {
    if (deltas[i] > deltas[i - 1] + params_.merge_lateral_rate_threshold) {
      monotonic_decrease = false;
    }
    if (deltas[i] + params_.merge_lateral_rate_threshold < deltas[i - 1]) {
      monotonic_increase = false;
    }
  }

  if (monotonic_decrease && delta_far < params_.merge_converged_threshold_m / 0.05) {
    if (merge_point) {
      const Vec2 pa = template_curve_->sample(a.xi, 1.0);
      const Vec2 pb = template_curve_->sample(b.xi, 1.0);
      *merge_point = 0.5 * (pa + pb);
    }
    return MergeTopologyType::MERGE;
  }

  if (monotonic_increase && delta_near < params_.merge_converged_threshold_m / 0.05) {
    if (merge_point) {
      const Vec2 pa = template_curve_->sample(a.xi, 0.0);
      const Vec2 pb = template_curve_->sample(b.xi, 0.0);
      *merge_point = 0.5 * (pa + pb);
    }
    return MergeTopologyType::DIVERGE;
  }

  if (merge_point) {
    merge_point->setZero();
  }
  return MergeTopologyType::PARALLEL;
}

void MergeTopology::completeGaps(LaneHypothesis & lane) const
{
  if (lane.polyline.size() >= 2) {
    return;
  }
  lane.polyline = template_curve_->samplePolyline(lane.xi);
}

std::vector<MergeEvent> MergeTopology::analyze(std::vector<LaneHypothesis> & lanes) const
{
  for (auto & lane : lanes) {
    completeGaps(lane);
    lane.topo = MergeTopologyType::PARALLEL;
  }

  std::vector<MergeEvent> events;
  for (size_t i = 0; i < lanes.size(); ++i) {
    for (size_t j = i + 1; j < lanes.size(); ++j) {
      Vec2 merge_pt = Vec2::Zero();
      const MergeTopologyType type = classifyPair(lanes[i], lanes[j], &merge_pt);
      if (type == MergeTopologyType::PARALLEL) {
        continue;
      }

      MergeEvent ev;
      ev.lane_a_id = lanes[i].lane_id;
      ev.lane_b_id = lanes[j].lane_id;
      ev.type = type;
      ev.merge_point = merge_pt;
      events.push_back(ev);

      lanes[i].topo = type;
      lanes[j].topo = type;
    }
  }
  return events;
}

}  // namespace lie_lane_detection
