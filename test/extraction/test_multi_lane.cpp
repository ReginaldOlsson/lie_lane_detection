#include "lie_lane_detection/extraction/multi_lane_extractor.hpp"
#include "lie_lane_detection/fitting/manifold_ransac.hpp"
#include "lie_lane_detection/testing/test_helpers.hpp"
#include "lie_lane_detection/voting/lie_hough_voter.hpp"

#include <gtest/gtest.h>

#include <array>

TEST(MultiLaneTest, DetectsThreeParallelLanes)
{
  lie_lane_detection::PipelineParams params;
  params.se2_vx_bins = 31;
  params.se2_vx_min = -40.0;
  params.se2_vx_max = 40.0;
  params.vote_threshold_px = 6.0;
  params.top_k_peaks = 6;
  params.nms_se2_min = 0.5;
  params.min_lane_separation_px = 15.0;
  params.min_inlier_ratio = 0.25;
  params.inlier_dedup_ratio = 0.45;
  params.use_ceres_fitter = false;
  params.se2_omega_bins = 7;
  params.kappa_bins = 7;

  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 200.0, -100.0, 100.0);

  std::vector<lie_lane_detection::EdgePoint> all_edges;
  const std::array<double, 3> offsets = {-25.0, 0.0, 25.0};
  for (size_t i = 0; i < offsets.size(); ++i) {
    lie_lane_detection::XiVector xi = lie_lane_detection::XiVector::Zero();
    xi[0] = offsets[i];
    auto edges = lie_lane_detection::test_helpers::samplePointsOnCurve(
      curve, xi, 50, 1.5, static_cast<unsigned>(42 + i));
    all_edges.insert(all_edges.end(), edges.begin(), edges.end());
  }

  lie_lane_detection::LieHoughVoter voter(params, &curve);
  auto seeds = voter.vote(all_edges);
  ASSERT_GE(seeds.size(), 2u);

  lie_lane_detection::ManifoldRansac ransac(params, &curve);
  std::vector<lie_lane_detection::LaneHypothesis> refined;
  for (const auto & s : seeds) {
    refined.push_back(ransac.fit(s, all_edges));
  }

  lie_lane_detection::MultiLaneExtractor extractor(params);
  const auto lanes = extractor.extract(refined);
  EXPECT_GE(lanes.size(), 2u);
}

TEST(MultiLaneTest, LateralDedupWithoutSupportingEdges)
{
  lie_lane_detection::PipelineParams params;
  params.min_lane_separation_px = 20.0;
  params.min_inlier_ratio = 0.1;
  params.max_output_lanes = 0;

  lie_lane_detection::LaneHypothesis a;
  a.xi[0] = 50.0;
  a.score = 10.0;
  a.inlier_ratio = 0.5;
  lie_lane_detection::LaneHypothesis b;
  b.xi[0] = 55.0;
  b.score = 8.0;
  b.inlier_ratio = 0.5;

  lie_lane_detection::MultiLaneExtractor extractor(params);
  const auto lanes = extractor.extract({a, b});
  EXPECT_EQ(lanes.size(), 1u);
}

TEST(MultiLaneTest, MaxOutputLanesKeepsEgoPair)
{
  lie_lane_detection::PipelineParams params;
  params.min_lane_separation_px = 15.0;
  params.min_inlier_ratio = 0.1;
  params.max_output_lanes = 2;
  params.se2_vx_min = 0.0;
  params.se2_vx_max = 200.0;

  lie_lane_detection::LaneHypothesis left;
  left.xi[0] = 60.0;
  left.score = 5.0;
  left.inlier_ratio = 0.5;
  lie_lane_detection::LaneHypothesis right;
  right.xi[0] = 140.0;
  right.score = 4.0;
  right.inlier_ratio = 0.5;
  lie_lane_detection::LaneHypothesis far_left;
  far_left.xi[0] = 20.0;
  far_left.score = 100.0;
  far_left.inlier_ratio = 0.9;

  lie_lane_detection::MultiLaneExtractor extractor(params);
  const auto lanes = extractor.extract({far_left, left, right});
  ASSERT_EQ(lanes.size(), 2u);
  EXPECT_LT(lanes[0].xi[0], 100.0);
  EXPECT_GT(lanes[1].xi[0], 100.0);
}
