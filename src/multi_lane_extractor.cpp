#include "lie_lane_detection/multi_lane_extractor.hpp"

#include <algorithm>
#include <set>

namespace lie_lane_detection
{

MultiLaneExtractor::MultiLaneExtractor(const PipelineParams & params)
: params_(params)
{
}

void MultiLaneExtractor::assignRoles(std::vector<LaneHypothesis> & lanes) const
{
  if (lanes.empty()) {
    return;
  }

  std::sort(lanes.begin(), lanes.end(), [](const LaneHypothesis & a, const LaneHypothesis & b) {
      return a.xi[0] < b.xi[0];
    });

  const int n = static_cast<int>(lanes.size());
  for (int i = 0; i < n; ++i) {
    if (n == 1) {
      lanes[i].role = LaneRole::CENTER;
    } else if (i == 0) {
      lanes[i].role = LaneRole::LEFT_ADJACENT;
    } else if (i == n - 1) {
      lanes[i].role = LaneRole::RIGHT_ADJACENT;
    } else if (i == n / 2 - 1 || (n % 2 == 0 && i == n / 2 - 1)) {
      lanes[i].role = LaneRole::LEFT_EGO;
    } else if (i == n / 2 || (n % 2 == 1 && i == n / 2)) {
      lanes[i].role = LaneRole::RIGHT_EGO;
    } else if (i < n / 2) {
      lanes[i].role = LaneRole::LEFT_ADJACENT;
    } else {
      lanes[i].role = LaneRole::RIGHT_ADJACENT;
    }
    lanes[i].lane_id = static_cast<uint32_t>(i);
  }
}

std::vector<LaneHypothesis> MultiLaneExtractor::extract(std::vector<LaneHypothesis> candidates) const
{
  std::sort(candidates.begin(), candidates.end(), [](const LaneHypothesis & a, const LaneHypothesis & b) {
      return a.score > b.score;
    });

  std::vector<LaneHypothesis> kept;
  for (auto & cand : candidates) {
    if (cand.inlier_ratio < params_.min_inlier_ratio) {
      continue;
    }
    if (cand.inlier_ratio <= 0.0 && cand.score <= 0.0) {
      continue;
    }

    bool duplicate = false;
    for (const auto & existing : kept) {
      if (existing.supporting_edges.empty() || cand.supporting_edges.empty()) {
        if (hypothesisDistance(cand.xi, existing.xi) < params_.nms_se2_min) {
          duplicate = true;
          break;
        }
        continue;
      }

      if (std::abs(cand.xi[0] - existing.xi[0]) < params_.min_lane_separation_px) {
        duplicate = true;
        break;
      }

      std::set<std::pair<int, int>> existing_pts;
      for (const auto & e : existing.supporting_edges) {
        existing_pts.insert({static_cast<int>(e.x), static_cast<int>(e.y)});
      }
      int shared = 0;
      for (const auto & e : cand.supporting_edges) {
        if (existing_pts.count({static_cast<int>(e.x), static_cast<int>(e.y)}) > 0) {
          ++shared;
        }
      }
      const double ratio = static_cast<double>(shared) /
        static_cast<double>(std::max(existing.supporting_edges.size(), cand.supporting_edges.size()));
      if (ratio > params_.inlier_dedup_ratio) {
        duplicate = true;
        break;
      }
    }

    if (!duplicate) {
      kept.push_back(cand);
    }
    if (static_cast<int>(kept.size()) >= params_.max_lane_hypotheses) {
      break;
    }
  }

  assignRoles(kept);
  return kept;
}

}  // namespace lie_lane_detection
