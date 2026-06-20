#include <gtest/gtest.h>

#include "lie_lane_detection/merge_topology.hpp"
#include "lie_lane_detection/test_helpers.hpp"

TEST(MergeTopologyTest, LabelsConvergingPairAsMerge)
{
  lie_lane_detection::PipelineParams params;
  params.merge_converged_threshold_m = 2.0;
  params.merge_lateral_rate_threshold = 0.01;

  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 100.0, -50.0, 50.0);

  lie_lane_detection::LaneHypothesis lane_a;
  lane_a.lane_id = 0;
  lane_a.xi = lie_lane_detection::XiVector::Zero();
  lane_a.xi[0] = -8.0;
  lane_a.xi[4] = 0.08;
  lane_a.polyline = curve.samplePolyline(lane_a.xi);

  lie_lane_detection::LaneHypothesis lane_b;
  lane_b.lane_id = 1;
  lane_b.xi = lie_lane_detection::XiVector::Zero();
  lane_b.xi[0] = 8.0;
  lane_b.xi[4] = -0.08;
  lane_b.polyline = curve.samplePolyline(lane_b.xi);

  std::vector<lie_lane_detection::LaneHypothesis> lanes = {lane_a, lane_b};
  lie_lane_detection::MergeTopology merge_topo(params, &curve);
  const auto events = merge_topo.analyze(lanes);

  ASSERT_FALSE(events.empty());
  EXPECT_EQ(events.front().type, lie_lane_detection::MergeTopologyType::MERGE);
}
