#include <gtest/gtest.h>

#include "lie_lane_detection/fitting/road_manifold_fitter.hpp"
#include "lie_lane_detection/testing/test_helpers.hpp"

TEST(RoadManifoldFitter, parallelLanesStaySeparated)
{
  lie_lane_detection::PipelineParams params;
  params.use_road_manifold_joint = true;
  params.min_lane_separation_px = 28.0;
  params.ceres_hard_gate_dist_px = 20.0;
  params.ceres_hard_gate_angle_rad = 0.5;
  params.min_inliers = 8;

  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 200.0, -100.0, 100.0);

  lie_lane_detection::XiVector xi_left = lie_lane_detection::XiVector::Zero();
  xi_left[0] = -40.0;
  lie_lane_detection::XiVector xi_right = lie_lane_detection::XiVector::Zero();
  xi_right[0] = 40.0;

  std::vector<lie_lane_detection::LineSegment> lines;
  const auto add_vertical_line = [&](double x, double y0, double y1) {
      lie_lane_detection::LineSegment seg;
      seg.x1 = x;
      seg.y1 = y0;
      seg.x2 = x;
      seg.y2 = y1;
      seg.length = std::abs(y1 - y0);
      seg.angle = std::atan2(seg.y2 - seg.y1, seg.x2 - seg.x1);
      seg.mx = 0.5 * (seg.x1 + seg.x2);
      seg.my = 0.5 * (seg.y1 + seg.y2);
      lines.push_back(seg);
    };
  for (int i = 0; i < 12; ++i) {
    add_vertical_line(-40.0, 20.0 + i * 10.0, 40.0 + i * 10.0);
    add_vertical_line(40.0, 20.0 + i * 10.0, 40.0 + i * 10.0);
  }

  std::vector<lie_lane_detection::LaneHypothesis> lanes(2);
  lanes[0].xi = xi_left;
  lanes[0].score = 100.0;
  lanes[1].xi = xi_right;
  lanes[1].score = 100.0;

  lie_lane_detection::RoadManifoldFitter fitter(params, &curve);
  ASSERT_TRUE(fitter.refine(lanes, lines));
  EXPECT_NEAR(lanes[0].xi[0], -40.0, 6.0);
  EXPECT_NEAR(lanes[1].xi[0], 40.0, 6.0);
  EXPECT_GT(lanes[1].xi[0] - lanes[0].xi[0], params.min_lane_separation_px);
}

TEST(RoadManifoldFitter, skipsSingleLane)
{
  lie_lane_detection::PipelineParams params;
  params.use_road_manifold_joint = true;

  lie_lane_detection::TemplateCurve curve(params);
  std::vector<lie_lane_detection::LaneHypothesis> lanes(1);
  lanes[0].score = 1.0;

  lie_lane_detection::RoadManifoldFitter fitter(params, &curve);
  EXPECT_FALSE(fitter.refine(lanes, std::vector<lie_lane_detection::LineSegment>{}));
}
