#include "lie_lane_detection/lane_detection_runner.hpp"

#include <algorithm>
#include <chrono>

#include <opencv2/imgproc.hpp>

#include "lie_lane_detection/edge_extractor.hpp"
#include "lie_lane_detection/ipm_transformer.hpp"
#include "lie_lane_detection/parallel.hpp"
#include "lie_lane_detection/lie_hough_voter.hpp"
#include "lie_lane_detection/manifold_ransac.hpp"
#include "lie_lane_detection/merge_topology.hpp"
#include "lie_lane_detection/multi_lane_extractor.hpp"
#include "lie_lane_detection/template_curve.hpp"
#include "lie_lane_detection/visualization.hpp"

namespace lie_lane_detection
{

namespace
{

std::vector<EdgePoint> filterBorderEdges(
  const std::vector<EdgePoint> & edges,
  double x_min,
  double x_max)
{
  std::vector<EdgePoint> filtered;
  filtered.reserve(edges.size());
  for (const auto & e : edges) {
    if (e.x >= x_min && e.x <= x_max) {
      filtered.push_back(e);
    }
  }
  return filtered;
}

std::vector<EdgePoint> peelEdgesNearCurve(
  const std::vector<EdgePoint> & edges,
  const TemplateCurve & curve,
  const XiVector & xi,
  double margin_px)
{
  std::vector<EdgePoint> remaining;
  remaining.reserve(edges.size());
  for (const auto & e : edges) {
    const Vec2 p(e.x, e.y);
    if (curve.distanceToCurve(xi, p) > margin_px) {
      remaining.push_back(e);
    }
  }
  return remaining;
}

bool isBorderLane(const LaneHypothesis & lane, const PipelineParams & params)
{
  const double span = params.se2_vx_max - params.se2_vx_min;
  const double margin = span * params.edge_border_margin_ratio;
  return lane.xi[0] < params.se2_vx_min + margin ||
         lane.xi[0] > params.se2_vx_max - margin;
}

bool isTooCloseToExisting(
  const LaneHypothesis & lane,
  const std::vector<LaneHypothesis> & existing,
  double min_sep_px)
{
  for (const auto & other : existing) {
    if (std::abs(lane.xi[0] - other.xi[0]) < min_sep_px) {
      return true;
    }
  }
  return false;
}

std::vector<EdgePoint> subsampleEdges(const std::vector<EdgePoint> & edges, size_t max_count)
{
  if (edges.size() <= max_count) {
    return edges;
  }
  std::vector<EdgePoint> sampled;
  sampled.reserve(max_count);
  const double stride = static_cast<double>(edges.size()) / static_cast<double>(max_count);
  for (size_t i = 0; i < max_count; ++i) {
    sampled.push_back(edges[static_cast<size_t>(i * stride)]);
  }
  return sampled;
}

bool passesQualityGate(const LaneHypothesis & lane, const PipelineParams & params)
{
  if (lane.inlier_ratio < params.min_inlier_ratio) {
    return false;
  }
  if (static_cast<int>(lane.supporting_edges.size()) < params.min_inliers) {
    return false;
  }
  if (isBorderLane(lane, params)) {
    return false;
  }
  return true;
}

LaneHypothesis pickBestSeed(
  const std::vector<LaneHypothesis> & seeds,
  const std::vector<EdgePoint> & edges,
  ManifoldRansac & ransac,
  const std::vector<LaneHypothesis> & accepted,
  const PipelineParams & params)
{
  std::vector<LaneHypothesis> fitted(seeds.size());
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, seeds.size()),
    [&](const tbb::blocked_range<size_t> & range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        fitted[i] = ransac.fit(seeds[i], edges);
      }
    });

  LaneHypothesis best;
  double best_metric = -1.0;
  for (const auto & hyp : fitted) {
    if (!passesQualityGate(hyp, params)) {
      continue;
    }
    if (isTooCloseToExisting(hyp, accepted, params.min_lane_separation_px)) {
      continue;
    }
    const double metric = hyp.score * hyp.inlier_ratio;
    if (metric > best_metric) {
      best_metric = metric;
      best = hyp;
    }
  }
  best.score = best_metric;
  return best;
}

}  // namespace

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

      LaneHypothesis best = pickBestSeed(seeds, remaining, ransac, refined, params);
      if (best.score <= 0.0 || !passesQualityGate(best, params)) {
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
      [&params](const LaneHypothesis & lane) {
        return lane.xi[0] < params.se2_vx_min ||
               lane.xi[0] > params.se2_vx_max ||
               lane.inlier_ratio < params.min_inlier_ratio ||
               isBorderLane(lane, params);
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
