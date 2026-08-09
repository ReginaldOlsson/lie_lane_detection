#pragma once

#include "lie_lane_detection/core/types.hpp"

#include <vector>

namespace lie_lane_detection
{

/// Pre-filter line segments that cannot belong to the same lane under curvature bounds.
class DualSpacePruner
{
public:
  explicit DualSpacePruner(const PipelineParams & params);

  std::vector<LineSegment> prune(const std::vector<LineSegment> & lines) const;

private:
  PipelineParams params_;
};

}  // namespace lie_lane_detection
