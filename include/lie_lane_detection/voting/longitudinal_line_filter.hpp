#pragma once

#include "lie_lane_detection/core/types.hpp"

#include <vector>

namespace lie_lane_detection
{

/// Drop near-horizontal segments (crosswalks, stop bars) in BEV.
class LongitudinalLineFilter
{
public:
  explicit LongitudinalLineFilter(const PipelineParams & params);

  std::vector<LineSegment> filter(const std::vector<LineSegment> & lines) const;

  bool passes(const LineSegment & line) const;

private:
  PipelineParams params_;
};

}  // namespace lie_lane_detection
