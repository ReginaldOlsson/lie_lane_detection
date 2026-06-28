#include <gtest/gtest.h>

#include "lie_lane_detection/fitting/ceres_lane_fitter.hpp"
#include "lie_lane_detection/testing/test_helpers.hpp"

TEST(CeresLaneFitter, refinesStraightLane)
{
  lie_lane_detection::PipelineParams params;
  params.use_ceres_fitter = true;
  params.soft_vote_sigma_px = 4.0;
  params.inlier_threshold_px = 8.0;
  params.ceres_hard_gate_dist_px = 20.0;
  params.ceres_hard_gate_angle_rad = 0.6;
  params.min_inliers = 10;

  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 200.0, -100.0, 100.0);

  lie_lane_detection::XiVector xi_true = lie_lane_detection::XiVector::Zero();
  xi_true[0] = 42.0;

  const auto edges = lie_lane_detection::test_helpers::samplePointsOnCurve(curve, xi_true, 80, 1.0);

  lie_lane_detection::LaneHypothesis seed;
  seed.xi[0] = 38.0;
  seed.score = 100.0;

  lie_lane_detection::CeresLaneFitter fitter(params, &curve);
  const auto refined = fitter.fitEdges(seed, edges);
  EXPECT_NEAR(refined.xi[0], 42.0, 2.0);
}
