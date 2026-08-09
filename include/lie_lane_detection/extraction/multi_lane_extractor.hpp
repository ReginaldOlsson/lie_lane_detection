#pragma once

#include "lie_lane_detection/core/types.hpp"

#include <vector>

namespace lie_lane_detection
{

class MultiLaneExtractor
{
public:
  explicit MultiLaneExtractor(const PipelineParams & params);

  std::vector<LaneHypothesis> extract(std::vector<LaneHypothesis> candidates) const;

private:
  void assignRoles(std::vector<LaneHypothesis> & lanes) const;

  std::vector<LaneHypothesis> limitOutputLanes(std::vector<LaneHypothesis> lanes) const;

  PipelineParams params_;
};

}  // namespace lie_lane_detection
