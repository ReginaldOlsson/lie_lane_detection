#include <gtest/gtest.h>

#include "lie_lane_detection/manifold_ransac.hpp"
#include "lie_lane_detection/test_helpers.hpp"

TEST(ManifoldRansacTest, RefinesStraightLane)
{
  lie_lane_detection::PipelineParams params;
  params.inlier_threshold_px = 5.0;
  params.ransac_iterations = 50;
  params.min_inliers = 10;

  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 200.0, -100.0, 100.0);

  lie_lane_detection::XiVector xi_true = lie_lane_detection::XiVector::Zero();
  xi_true[0] = -20.0;

  auto edges = lie_lane_detection::test_helpers::samplePointsOnCurve(curve, xi_true, 100, 2.0);

  lie_lane_detection::LaneHypothesis seed;
  seed.xi = lie_lane_detection::XiVector::Zero();
  seed.xi[0] = -10.0;
  seed.score = 100.0;

  lie_lane_detection::ManifoldRansac ransac(params, &curve);
  const auto result = ransac.fit(seed, edges);
  EXPECT_NEAR(result.xi[0], -20.0, 4.0);
  EXPECT_GT(result.inlier_ratio, 0.5);
}
