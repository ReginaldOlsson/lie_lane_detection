#include <gtest/gtest.h>

#include "lie_lane_detection/voting/lie_hough_voter.hpp"
#include "lie_lane_detection/fitting/manifold_ransac.hpp"
#include "lie_lane_detection/testing/test_helpers.hpp"

TEST(CurvedLaneTest, KappaRecoveryWithinTolerance)
{
  lie_lane_detection::PipelineParams params;
  params.kappa_bins = 21;
  params.kappa_min = 0.0;
  params.kappa_max = 0.3;
  params.se2_omega_bins = 11;
  params.se2_omega_min = -0.2;
  params.se2_omega_max = 0.2;
  params.vote_threshold_px = 5.0;
  params.inlier_threshold_px = 6.0;

  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 200.0, -100.0, 100.0);

  lie_lane_detection::XiVector xi_true = lie_lane_detection::XiVector::Zero();
  xi_true[2] = 0.05;
  xi_true[3] = 0.1;

  const auto edges = lie_lane_detection::test_helpers::samplePointsOnCurve(curve, xi_true, 80, 1.5);

  lie_lane_detection::LieHoughVoter voter(params, &curve);
  auto seeds = voter.vote(edges);
  ASSERT_FALSE(seeds.empty());

  lie_lane_detection::ManifoldRansac ransac(params, &curve);
  const auto result = ransac.fit(seeds.front(), edges);

  EXPECT_NEAR(result.xi[3], 0.1, 0.1 * 0.15 + 0.03);
}
