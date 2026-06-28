#include <gtest/gtest.h>

#include "lie_lane_detection/tracking/lane_tracker.hpp"

TEST(LaneTracker, lateralStdGrowsWithDistance)
{
  lie_lane_detection::LaneTracker tracker;
  const double near_std = tracker.lateralStdAtY(50.0, 400.0);
  const double far_std = tracker.lateralStdAtY(350.0, 400.0);
  EXPECT_GT(far_std, near_std);
}

TEST(LaneTracker, narrowParamsShrinkVxBins)
{
  lie_lane_detection::PipelineParams params;
  params.se2_vx_min = 0.0;
  params.se2_vx_max = 240.0;
  params.se2_vx_bins = 30;

  lie_lane_detection::LaneTrack track;
  track.xi[0] = 120.0;
  lie_lane_detection::narrowParamsForTracks(params, {track}, 40.0);

  EXPECT_LT(params.se2_vx_max - params.se2_vx_min, 240.0);
  EXPECT_GE(params.se2_vx_bins, 9);
  EXPECT_LT(params.se2_vx_bins, 30);
}
