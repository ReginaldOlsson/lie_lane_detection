#include <gtest/gtest.h>

#include "lie_lane_detection/voting/lie_hough_voter.hpp"
#include "lie_lane_detection/testing/test_helpers.hpp"

TEST(LieHoughTest, RecoversStraightLane)
{
  lie_lane_detection::PipelineParams params;
  params.se2_vx_bins = 21;
  params.se2_vy_bins = 5;
  params.se2_omega_bins = 11;
  params.se2_vx_min = -20.0;
  params.se2_vx_max = 20.0;
  params.vote_threshold_px = 6.0;
  params.top_k_peaks = 3;

  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 200.0, -100.0, 100.0);

  lie_lane_detection::XiVector xi_true = lie_lane_detection::XiVector::Zero();
  xi_true[0] = 15.0;

  const auto edges = lie_lane_detection::test_helpers::samplePointsOnCurve(curve, xi_true, 80, 1.0);

  lie_lane_detection::LieHoughVoter voter(params, &curve);
  const auto hyps = voter.vote(edges);
  ASSERT_FALSE(hyps.empty());
  EXPECT_NEAR(hyps.front().xi[0], 15.0, 3.0);
}

TEST(LieHoughTest, RecoversCurvedLaneKappa)
{
  lie_lane_detection::PipelineParams params;
  params.se2_vx_bins = 11;
  params.se2_vy_bins = 5;
  params.se2_omega_bins = 11;
  params.kappa_bins = 21;
  params.kappa_min = 0.0;
  params.kappa_max = 0.3;
  params.se2_omega_bins = 5;
  params.se2_omega_min = -0.05;
  params.se2_omega_max = 0.05;
  params.vote_threshold_px = 6.0;

  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 200.0, -100.0, 100.0);

  lie_lane_detection::XiVector xi_true = lie_lane_detection::XiVector::Zero();
  xi_true[3] = 0.12;

  const auto edges = lie_lane_detection::test_helpers::samplePointsOnCurve(curve, xi_true, 60, 1.0);

  lie_lane_detection::LieHoughVoter voter(params, &curve);
  const auto hyps = voter.vote(edges);
  ASSERT_FALSE(hyps.empty());
  EXPECT_NEAR(hyps.front().xi[3], 0.12, 0.1);
}
