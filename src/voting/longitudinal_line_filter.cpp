#include "lie_lane_detection/voting/longitudinal_line_filter.hpp"

namespace lie_lane_detection
{

LongitudinalLineFilter::LongitudinalLineFilter(const PipelineParams & params)
: params_(params)
{
}

bool LongitudinalLineFilter::passes(const LineSegment & line) const
{
  return isLongitudinalSegment(line.angle, params_.longitudinal_max_deviation_rad);
}

std::vector<LineSegment> LongitudinalLineFilter::filter(const std::vector<LineSegment> & lines) const
{
  if (!params_.use_longitudinal_line_filter) {
    return lines;
  }
  std::vector<LineSegment> kept;
  kept.reserve(lines.size());
  for (const auto & line : lines) {
    if (passes(line)) {
      kept.push_back(line);
    }
  }
  return kept;
}

}  // namespace lie_lane_detection
