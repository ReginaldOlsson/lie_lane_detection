#include "lie_lane_detection/testing/test_helpers.hpp"
#include "lie_lane_detection/voting/coarse_se2_voter.hpp"

#include <gtest/gtest.h>

TEST(CoarseSE2Voter, recoversStraightLane)
{
  lie_lane_detection::PipelineParams params;
  params.use_continuous_pipeline = true;
  params.use_coarse_pyramid_voter = true;
  params.use_soft_voting = true;
  params.use_ceres_fitter = true;
  params.pyramid_coarse_bins = 16;
  params.se2_vx_bins = 21;
  params.se2_vy_bins = 1;
  params.se2_omega_bins = 9;
  params.se2_vx_min = 0.0;
  params.se2_vx_max = 200.0;
  params.vote_threshold_px = 8.0;
  params.top_k_peaks = 3;

  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 200.0, 0.0, 200.0);

  lie_lane_detection::XiVector xi_true = lie_lane_detection::XiVector::Zero();
  xi_true[0] = 80.0;

  const auto edges = lie_lane_detection::test_helpers::samplePointsOnCurve(curve, xi_true, 60, 1.0);

  lie_lane_detection::CoarseSE2Voter voter(params, &curve);
  const auto hyps = voter.voteEdges(edges);
  ASSERT_FALSE(hyps.empty());
  EXPECT_NEAR(hyps.front().xi[0], 80.0, 18.0);
}
