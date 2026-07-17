#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"

#include <algorithm>
#include <chrono>

#include <opencv2/imgproc.hpp>

#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/preprocessing/edge_extractor.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"
#include "lie_lane_detection/preprocessing/road_feature_segmenter.hpp"
#include "lie_lane_detection/voting/lie_hough_voter.hpp"
#include "lie_lane_detection/voting/coarse_se2_voter.hpp"
#include "lie_lane_detection/fitting/manifold_ransac.hpp"
#include "lie_lane_detection/fitting/road_manifold_fitter.hpp"
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
  params.min_lane_separation_px = std::max(
    params.min_lane_separation_px,
    std::max(35.0, static_cast<double>(cols) / 8.0));
  params.peel_edge_margin_px = std::max(
    params.peel_edge_margin_px,
    std::max(12.0, params.inlier_threshold_px * 1.5));

  // Hood crop defaults target ~800px highway BEV; clamp for small metric IPM footprints.
  if (rows > 0) {
    const double max_exclude = 0.30 * static_cast<double>(rows);
    if (params.bev_bottom_exclude_px > max_exclude) {
      params.bev_bottom_exclude_px = max_exclude;
    }
    const double max_margin = 0.12 * static_cast<double>(rows);
    if (params.bev_bottom_edge_margin_px > max_margin) {
      params.bev_bottom_edge_margin_px = max_margin;
    }
  }

  const int area = std::max(1, cols * rows);
  params.min_inliers = std::min(params.min_inliers, std::max(5, area / 120));
  params.min_lane_separation_px = std::min(
    params.min_lane_separation_px,
    std::max(3.0, static_cast<double>(cols) / 6.0));
  if (params.min_lane_inlier_y_span_px <= 0.0) {
    params.min_lane_inlier_y_span_px = 0.22 * static_cast<double>(rows);
  }
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

void updateIpmDstFromBevExtent(PipelineParams & params)
{
  const double half_w = 0.5 * params.bev_width_m;
  const double length = params.bev_length_m;
  params.ipm_dst_points = {
    -half_w, 0.0,
    half_w, 0.0,
    half_w, length,
    -half_w, length,
  };
}

void setDefaultHighwayIpmRoi(PipelineParams & params, int cols, int rows)
{
  const double w = static_cast<double>(cols);
  const double h = static_cast<double>(rows);
  updateIpmDstFromBevExtent(params);
  params.ipm_src_points = {
    params.ipm_bottom_x_min_ratio * w, params.ipm_bottom_y_ratio * h,
    params.ipm_bottom_x_max_ratio * w, params.ipm_bottom_y_ratio * h,
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
  params.min_inlier_ratio = 0.36;
  params.min_inliers = 16;
  params.min_lane_separation_px = 22.0;
  params.edge_border_margin_ratio = 0.05;
  params.top_k_peaks = 10;
  params.max_lane_hypotheses = 8;
  params.min_inlier_y_coverage = 0.32;
  params.min_lane_relative_score = 0.40;
  params.max_lane_omega_deviation_rad = 0.16;
  params.max_output_lanes = 6;
  params.min_lane_edge_orientation_ratio = 0.44;
  params.max_lane_inlier_gap_px = 55.0;
  params.max_lane_inlier_gap_ratio = 0.32;
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

BevDetectionResult detectLanesInBev(
  const cv::Mat & bev_bgr,
  PipelineParams params,
  bool configure_params,
  RoadFeatureSegmenter * road_segmenter)
{
  const auto t0 = std::chrono::steady_clock::now();
  BevDetectionResult result;
  if (bev_bgr.empty()) {
    return result;
  }

  if (configure_params) {
    configureParamsForBev(params, bev_bgr.cols, bev_bgr.rows);
  }

  const cv::Mat work_bev = prepareBevForDetection(bev_bgr, params);
  const double y_max = bevEffectiveYMax(bev_bgr.rows, params);

  cv::Mat segmenter_mask;
  if (params.use_road_feature_segmenter && road_segmenter != nullptr && road_segmenter->isReady()) {
    road_segmenter->updateParams(params);
    const auto seg = road_segmenter->infer(work_bev);
    segmenter_mask = seg.drivable_mask;
    result.road_feature_mask = seg.drivable_mask.clone();
    result.road_feature_debug = seg.debug_bgr.clone();
  }

  TemplateCurve template_curve(params);
  template_curve.setBevExtents(
    0.0, y_max,
    0.0, static_cast<double>(bev_bgr.cols));

  EdgeExtractor edge_extractor(params);
  cv::Mat edges_img;
  cv::Mat detect_bev = work_bev;
  if (!segmenter_mask.empty()) {
    detect_bev = work_bev.clone();
    applyRoadFeatureMask(detect_bev, segmenter_mask);
  }
  const auto t_edge0 = std::chrono::steady_clock::now();
  const auto all_edges = edge_extractor.extract(detect_bev, &edges_img);
  result.edge_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_edge0).count();
  result.edge_point_count = all_edges.size();

  if (!edges_img.empty()) {
    const int mask_rows = bevBottomMaskRows(params);
    if (mask_rows > 0) {
      const int y0 = std::max(0, edges_img.rows - mask_rows);
      edges_img.rowRange(y0, edges_img.rows).setTo(0);
    }
    if (!segmenter_mask.empty()) {
      edges_img = maskEdgesWithRoadFeatures(edges_img, segmenter_mask);
    }
  }

  const double border_margin = static_cast<double>(bev_bgr.cols) * params.edge_border_margin_ratio;
  auto working_edges = filterBorderEdges(
    all_edges,
    border_margin,
    static_cast<double>(bev_bgr.cols) - border_margin);
  working_edges = filterBevYMaxEdges(working_edges, y_max);

  constexpr size_t kMaxVoteEdges = 18000;
  working_edges = subsampleEdges(working_edges, kMaxVoteEdges);

  if (params.use_continuous_pipeline && params.use_coarse_pyramid_voter) {
    CoarseSE2Voter coarse_voter(params, &template_curve);
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
        const auto iter_seeds = coarse_voter.voteEdges(remaining, hough_ptr);
        if (iter_seeds.empty()) {
          break;
        }
        LaneHypothesis best = pickBestSeed(iter_seeds, remaining, ransac, refined, params, y_max, &template_curve);
        if (best.score <= 0.0 || !passesQualityGate(best, params, y_max, &template_curve)) {
          break;
        }
        refined.push_back(best);
        remaining = peelEdgesNearCurve(
          remaining, template_curve, best.xi, params.peel_edge_margin_px);
      }
      if (params.use_road_manifold_joint && refined.size() >= 2) {
        RoadManifoldFitter road_fitter(params, &template_curve);
        road_fitter.refine(refined, working_edges);
      }
    } else {
      const auto iter_seeds = coarse_voter.voteEdges(working_edges, &result.hough_slice);
      refined.resize(iter_seeds.size());
      tbb::parallel_for(
        tbb::blocked_range<size_t>(0, iter_seeds.size()),
        [&](const tbb::blocked_range<size_t> & range) {
          for (size_t i = range.begin(); i != range.end(); ++i) {
            refined[i] = ransac.fit(iter_seeds[i], working_edges);
          }
        });
    }

    result.lanes = multi_lane.extract(refined);
    result.lanes = pruneNoiseLaneHypotheses(std::move(result.lanes), params);
    result.lanes.erase(
      std::remove_if(
        result.lanes.begin(), result.lanes.end(),
        [&params, &y_max, &template_curve](const LaneHypothesis & lane) {
          if (lane.xi[0] < params.se2_vx_min ||
            lane.xi[0] > params.se2_vx_max ||
            lane.inlier_ratio < params.min_inlier_ratio ||
            isBorderLane(lane, params))
          {
            return true;
          }
          return !passesQualityGate(lane, params, y_max, &template_curve);
        }),
      result.lanes.end());

    result.merges = merge_topology.analyze(result.lanes);
    result.edges = edges_img;
    result.overlay = drawOverlay(work_bev, result.lanes, result.merges);
    const auto t1 = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return result;
  }

  LieHoughVoter voter(params, &template_curve);
  ManifoldRansac ransac(params, &template_curve);
  MultiLaneExtractor multi_lane(params);
  MergeTopology merge_topology(params, &template_curve);

  const auto t_votefit0 = std::chrono::steady_clock::now();
  std::vector<LaneHypothesis> refined;
  if (params.use_iterative_peeling) {
    std::vector<EdgePoint> remaining = working_edges;
    for (int iter = 0; iter < params.max_lane_hypotheses; ++iter) {
      if (static_cast<int>(remaining.size()) < params.min_inliers) {
        break;
      }

      cv::Mat * hough_ptr = (iter == 0) ? &result.hough_slice : nullptr;
      const auto t_vote0 = std::chrono::steady_clock::now();
      const auto seeds = voter.vote(remaining, hough_ptr);
      result.vote_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_vote0)
        .count();
      if (seeds.empty()) {
        break;
      }

      const auto t_fit0 = std::chrono::steady_clock::now();
      LaneHypothesis best = pickBestSeed(seeds, remaining, ransac, refined, params, y_max, &template_curve);
      result.fit_ms +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_fit0)
        .count();
      if (best.score <= 0.0 || !passesQualityGate(best, params, y_max, &template_curve)) {
        break;
      }

      refined.push_back(best);
      remaining = peelEdgesNearCurve(
        remaining, template_curve, best.xi, params.peel_edge_margin_px);
    }
  } else {
    const auto t_vote0 = std::chrono::steady_clock::now();
    const auto seeds = voter.vote(working_edges, &result.hough_slice);
    result.vote_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_vote0).count();
    refined.resize(seeds.size());
    const auto t_fit0 = std::chrono::steady_clock::now();
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, seeds.size()),
      [&](const tbb::blocked_range<size_t> & range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          refined[i] = ransac.fit(seeds[i], working_edges);
        }
      });
    result.fit_ms +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_fit0).count();
  }
  result.lie_vote_ms = result.vote_ms;
  (void)t_votefit0;

  const auto t_post0 = std::chrono::steady_clock::now();
  result.lanes = multi_lane.extract(refined);
  result.lanes = pruneNoiseLaneHypotheses(std::move(result.lanes), params);
  result.lanes.erase(
    std::remove_if(
      result.lanes.begin(), result.lanes.end(),
      [&params, &y_max, &template_curve](const LaneHypothesis & lane) {
        if (lane.xi[0] < params.se2_vx_min ||
            lane.xi[0] > params.se2_vx_max ||
            lane.inlier_ratio < params.min_inlier_ratio ||
            isBorderLane(lane, params))
        {
          return true;
        }
        return !passesQualityGate(lane, params, y_max, &template_curve);
      }),
    result.lanes.end());

  result.merges = merge_topology.analyze(result.lanes);
  result.edges = edges_img;
  result.overlay = drawOverlay(work_bev, result.lanes, result.merges);
  result.post_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_post0).count();
  const auto t1 = std::chrono::steady_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return result;
}

}  // namespace lie_lane_detection