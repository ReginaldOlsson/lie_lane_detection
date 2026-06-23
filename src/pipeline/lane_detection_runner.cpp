#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"

#include <algorithm>
#include <chrono>

#include <opencv2/imgproc.hpp>

#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/preprocessing/edge_extractor.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"
#include "lie_lane_detection/voting/lie_hough_voter.hpp"
#include "lie_lane_detection/fitting/manifold_ransac.hpp"
#include "lie_lane_detection/extraction/merge_topology.hpp"
#include "lie_lane_detection/extraction/multi_lane_extractor.hpp"
#include "lie_lane_detection/common/parallel.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"

namespace lie_lane_detection
{

void configureParamsForBev(PipelineParams & params, int cols, int rows)
{
  params.se2_vx_min = 0.0;
  params.se2_vx_max = static_cast<double>(cols);
  params.se2_vy_min = -static_cast<double>(rows) * 0.1;
  params.se2_vy_max = static_cast<double>(rows) * 0.1;
  params.se2_vx_bins = std::clamp(cols / 8, 25, 41);
  params.se2_vy_bins = 1;
  params.se2_omega_bins = 9;
  params.se2_omega_min = -0.2;
  params.se2_omega_max = 0.2;
  params.kappa_bins = 11;
  params.sigma_bins = 9;
  params.min_lane_separation_px = std::max(22.0, static_cast<double>(cols) / 10.0);
  params.peel_edge_margin_px = std::max(8.0, params.inlier_threshold_px * 1.2);
}

void enhanceParamsForCurvature(PipelineParams & params)
{
  params.se2_omega_bins = std::max(params.se2_omega_bins, 11);
  params.kappa_bins = std::max(params.kappa_bins, 15);
  params.sigma_bins = std::max(params.sigma_bins, 11);
  params.ransac_iterations = std::max(params.ransac_iterations, 150);
  params.nms_def_min = 0.03;
}

cv::Mat prepareBevImage(const cv::Mat & bev_bgr)
{
  cv::Mat bev = bev_bgr.clone();
  cv::Mat gray;
  cv::cvtColor(bev, gray, cv::COLOR_BGR2GRAY);
  cv::Mat road_mask = gray > 25;
  bev.setTo(cv::Scalar(0, 0, 0), ~road_mask);
  return bev;
}

void setDefaultHighwayIpmRoi(PipelineParams & params, int cols, int rows)
{
  const double w = static_cast<double>(cols);
  const double h = static_cast<double>(rows);
  params.bev_width_m = 12.0;
  params.bev_length_m = 40.0;
  params.bev_resolution_m_per_px = 0.05;
  params.ipm_dst_points = {-6.0, 0.0, 6.0, 0.0, 6.0, 40.0, -6.0, 40.0};
  params.ipm_src_points = {
    0.12 * w, 0.96 * h,
    0.88 * w, 0.96 * h,
    0.58 * w, 0.48 * h,
    0.42 * w, 0.48 * h,
  };
}

void configureParamsForPerspectiveIpm(PipelineParams & params)
{
  params.edge_low_threshold = 35.0;
  params.edge_high_threshold = 95.0;
  params.vote_threshold_px = 9.0;
  params.inlier_threshold_px = 11.0;
  params.min_inlier_ratio = 0.32;
  params.min_inliers = 20;
  params.min_lane_separation_px = 20.0;
  params.edge_border_margin_ratio = 0.05;
  params.top_k_peaks = 10;
  params.max_lane_hypotheses = 10;
}

void configureParamsForFrontalImage(PipelineParams & params, int cols, int rows)
{
  configureParamsForBev(params, cols, rows);
  params.se2_omega_min = -0.8;
  params.se2_omega_max = 0.8;
  params.se2_omega_bins = 15;
  params.se2_vy_bins = 3;
  params.se2_vy_min = -static_cast<double>(rows) * 0.12;
  params.se2_vy_max = static_cast<double>(rows) * 0.12;
  params.sigma_min = -0.55;
  params.sigma_max = 0.55;
  params.sigma_bins = 13;
  params.kappa_min = -0.28;
  params.kappa_max = 0.28;
  params.kappa_bins = 13;
  params.min_inlier_y_coverage = 0.22;
  params.edge_border_margin_ratio = 0.03;
  params.vote_threshold_px = 11.0;
  params.inlier_threshold_px = 14.0;
  params.min_inlier_ratio = 0.26;
  params.min_inliers = 18;
  params.min_lane_separation_px = std::max(45.0, static_cast<double>(cols) * 0.07);
  params.top_k_peaks = 10;
  params.max_lane_hypotheses = 10;
  params.line_angle_threshold_rad = 0.75;
  params.line_hough_threshold = 28;
  params.line_min_length_px = 22.0;
}

cv::Mat prepareFrontalImage(const cv::Mat & image_bgr)
{
  cv::Mat out = image_bgr.clone();
  const int sky_rows = static_cast<int>(out.rows * 0.30);
  if (sky_rows > 0) {
    out.rowRange(0, sky_rows).setTo(cv::Scalar(0, 0, 0));
  }
  return out;
}

cv::Mat warpPerspectiveToBev(const cv::Mat & image_bgr, const PipelineParams & params)
{
  IPMTransformer ipm(params);
  if (!ipm.computeHomography(nullptr)) {
    return cv::Mat();
  }
  return ipm.warpToBev(image_bgr);
}

BevDetectionResult detectLanesInBev(const cv::Mat & bev_bgr, PipelineParams params)
{
  const auto t0 = std::chrono::steady_clock::now();
  BevDetectionResult result;
  if (bev_bgr.empty()) {
    return result;
  }

  configureParamsForBev(params, bev_bgr.cols, bev_bgr.rows);

  TemplateCurve template_curve(params);
  template_curve.setBevExtents(
    0.0, static_cast<double>(bev_bgr.rows),
    0.0, static_cast<double>(bev_bgr.cols));

  EdgeExtractor edge_extractor(params);
  cv::Mat edges_img;
  const auto all_edges = edge_extractor.extract(bev_bgr, &edges_img);
  result.edge_point_count = all_edges.size();

  const double border_margin = static_cast<double>(bev_bgr.cols) * params.edge_border_margin_ratio;
  auto working_edges = filterBorderEdges(
    all_edges,
    border_margin,
    static_cast<double>(bev_bgr.cols) - border_margin);

  constexpr size_t kMaxVoteEdges = 18000;
  working_edges = subsampleEdges(working_edges, kMaxVoteEdges);

  LieHoughVoter voter(params, &template_curve);
  ManifoldRansac ransac(params, &template_curve);
  MultiLaneExtractor multi_lane(params);
  MergeTopology merge_topology(params, &template_curve);

  std::vector<LaneHypothesis> refined;
  if (params.use_iterative_peeling) {
    std::vector<EdgePoint> remaining = working_edges;
    for (int iter = 0; iter < params.max_lane_hypotheses; ++iter) {
      if (static_cast<int>(remaining.size()) < params.min_inliers) {
        break;
      }

      cv::Mat * hough_ptr = (iter == 0) ? &result.hough_slice : nullptr;
      const auto seeds = voter.vote(remaining, hough_ptr);
      if (seeds.empty()) {
        break;
      }

      LaneHypothesis best = pickBestSeed(seeds, remaining, ransac, refined, params, bev_bgr.rows);
      if (best.score <= 0.0 || !passesQualityGate(best, params, bev_bgr.rows)) {
        break;
      }

      refined.push_back(best);
      remaining = peelEdgesNearCurve(
        remaining, template_curve, best.xi, params.peel_edge_margin_px);
    }
  } else {
    const auto seeds = voter.vote(working_edges, &result.hough_slice);
    refined.resize(seeds.size());
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, seeds.size()),
      [&](const tbb::blocked_range<size_t> & range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          refined[i] = ransac.fit(seeds[i], working_edges);
        }
      });
  }

  result.lanes = multi_lane.extract(refined);
  result.lanes.erase(
    std::remove_if(
      result.lanes.begin(), result.lanes.end(),
      [&params, &bev_bgr](const LaneHypothesis & lane) {
        if (lane.xi[0] < params.se2_vx_min ||
            lane.xi[0] > params.se2_vx_max ||
            lane.inlier_ratio < params.min_inlier_ratio ||
            isBorderLane(lane, params))
        {
          return true;
        }
        return !passesQualityGate(lane, params, static_cast<double>(bev_bgr.rows));
      }),
    result.lanes.end());

  result.merges = merge_topology.analyze(result.lanes);
  result.edges = edges_img;
  result.overlay = drawOverlay(bev_bgr, result.lanes, result.merges);
  const auto t1 = std::chrono::steady_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return result;
}

}  // namespace lie_lane_detection