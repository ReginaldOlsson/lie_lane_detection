#include "lie_lane_detection/voting/longitudinal_line_filter.hpp"

#include <gtest/gtest.h>

#include <cmath>

TEST(LongitudinalLineFilter, dropsHorizontalCrosswalkSegments)
{
  lie_lane_detection::PipelineParams params;
  params.use_longitudinal_line_filter = true;
  params.longitudinal_max_deviation_rad = 0.52;

  lie_lane_detection::LineSegment lane;
  lane.x1 = 100.0;
  lane.y1 = 50.0;
  lane.x2 = 100.0;
  lane.y2 = 200.0;
  lane.length = 150.0;
  lane.angle = std::atan2(lane.y2 - lane.y1, lane.x2 - lane.x1);
  lane.mx = 0.5 * (lane.x1 + lane.x2);
  lane.my = 0.5 * (lane.y1 + lane.y2);

  lie_lane_detection::LineSegment crosswalk;
  crosswalk.x1 = 50.0;
  crosswalk.y1 = 120.0;
  crosswalk.x2 = 200.0;
  crosswalk.y2 = 120.0;
  crosswalk.length = 150.0;
  crosswalk.angle = std::atan2(crosswalk.y2 - crosswalk.y1, crosswalk.x2 - crosswalk.x1);
  crosswalk.mx = 0.5 * (crosswalk.x1 + crosswalk.x2);
  crosswalk.my = 0.5 * (crosswalk.y1 + crosswalk.y2);

  lie_lane_detection::LongitudinalLineFilter filter(params);
  const auto kept = filter.filter({lane, crosswalk});
  ASSERT_EQ(kept.size(), 1u);
  EXPECT_NEAR(kept[0].angle, lane.angle, 1e-6);
}

TEST(LongitudinalLineFilter, disabledPassesAll)
{
  lie_lane_detection::PipelineParams params;
  params.use_longitudinal_line_filter = false;

  lie_lane_detection::LineSegment crosswalk;
  crosswalk.angle = 0.0;
  crosswalk.length = 20.0;

  lie_lane_detection::LongitudinalLineFilter filter(params);
  const auto kept = filter.filter({crosswalk});
  EXPECT_EQ(kept.size(), 1u);
}
