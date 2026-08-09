#include "lie_lane_detection/pipeline/line_lane_detection_runner.hpp"

#include "lie_lane_detection/common/parallel.hpp"
#include "lie_lane_detection/extraction/merge_topology.hpp"
#include "lie_lane_detection/extraction/multi_lane_extractor.hpp"
#include "lie_lane_detection/fitting/manifold_ransac.hpp"
#include "lie_lane_detection/fitting/road_manifold_fitter.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/preprocessing/edge_extractor.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"
#include "lie_lane_detection/voting/coarse_se2_voter.hpp"
#include "lie_lane_detection/voting/dual_space_pruner.hpp"
#include "lie_lane_detection/voting/line_hough_extractor.hpp"
#include "lie_lane_detection/voting/line_kdtree.hpp"
#include "lie_lane_detection/voting/line_lie_hough_voter.hpp"
#include "lie_lane_detection/voting/longitudinal_line_filter.hpp"

#include <algorithm>
#include <chrono>

namespace lie_lane_detection
{

BevDetectionResult detectLanesInBevFromLines(const cv::Mat & bev_bgr, PipelineParams params)
{
  const auto t0 = std::chrono::steady_clock::now();
  BevDetectionResult result;
  if (bev_bgr.empty()) {
    return result;
  }

  configureParamsForBev(params, bev_bgr.cols, bev_bgr.rows);

  const cv::Mat work_bev = prepareBevForDetection(bev_bgr, params);
  const double y_max = bevEffectiveYMax(bev_bgr.rows, params);

  TemplateCurve template_curve(params);
  template_curve.setBevExtents(0.0, y_max, 0.0, static_cast<double>(bev_bgr.cols));

  EdgeExtractor edge_extractor(params);
  cv::Mat edges_img;
  const auto all_edges = edge_extractor.extract(work_bev, &edges_img);
  result.edge_point_count = all_edges.size();

  const auto t_line0 = std::chrono::steady_clock::now();
  LineHoughExtractor line_extractor(params);
  auto line_segments = line_extractor.extract(edges_img);
  const auto t_line1 = std::chrono::steady_clock::now();
  result.line_hough_ms = std::chrono::duration<double, std::milli>(t_line1 - t_line0).count();
  result.line_segment_count = line_segments.size();

  const double border_margin = static_cast<double>(bev_bgr.cols) * params.edge_border_margin_ratio;
  line_segments = filterBorderLines(
    line_segments, border_margin, static_cast<double>(bev_bgr.cols) - border_margin);
  line_segments = filterBevYMaxLines(line_segments, y_max);

  if (params.use_longitudinal_line_filter) {
    LongitudinalLineFilter long_filter(params);
    line_segments = long_filter.filter(line_segments);
  }

  if (params.use_dual_space_prune) {
    DualSpacePruner pruner(params);
    line_segments = pruner.prune(line_segments);
  }

  LineKDTree line_kdtree;
  if (params.use_line_kdtree) {
    line_kdtree.build(line_segments);
  }
  (void)line_kdtree;

  const double border_margin_edges = border_margin;
  auto working_edges = filterBorderEdges(
    all_edges, border_margin_edges, static_cast<double>(bev_bgr.cols) - border_margin_edges);
  working_edges = filterBevYMaxEdges(working_edges, y_max);
  constexpr size_t kMaxVoteEdges = 18000;
  working_edges = subsampleEdges(working_edges, kMaxVoteEdges);

  LineLieHoughVoter voter(params, &template_curve);
  ManifoldRansac ransac(params, &template_curve);
  MultiLaneExtractor multi_lane(params);
  MergeTopology merge_topology(params, &template_curve);

  const auto t_vote0 = std::chrono::steady_clock::now();
  std::vector<LaneHypothesis> refined;
  if (params.use_continuous_pipeline && params.use_coarse_pyramid_voter) {
    CoarseSE2Voter coarse_voter(params, &template_curve);
    if (params.use_iterative_peeling) {
      std::vector<LineSegment> remaining_lines = line_segments;
      for (int iter = 0; iter < params.max_lane_hypotheses; ++iter) {
        if (static_cast<int>(remaining_lines.size()) < 2) {
          break;
        }
        cv::Mat * hough_ptr = (iter == 0) ? &result.hough_slice : nullptr;
        const auto seeds = coarse_voter.voteLines(remaining_lines, hough_ptr);
        if (seeds.empty()) {
          break;
        }
        LaneHypothesis best = pickBestSeed(seeds, working_edges, ransac, refined, params, y_max);
        if (best.score <= 0.0 || !passesQualityGate(best, params, y_max)) {
          break;
        }
        refined.push_back(best);
        remaining_lines =
          peelLinesNearCurve(remaining_lines, template_curve, best.xi, params.peel_edge_margin_px);
      }
    } else {
      const auto seeds = coarse_voter.voteLines(line_segments, &result.hough_slice);
      refined.resize(seeds.size());
      tbb::parallel_for(
        tbb::blocked_range<size_t>(0, seeds.size()), [&](const tbb::blocked_range<size_t> & range) {
          for (size_t i = range.begin(); i != range.end(); ++i) {
            refined[i] = ransac.fit(seeds[i], working_edges);
          }
        });
    }
  } else if (params.use_iterative_peeling) {
    std::vector<LineSegment> remaining_lines = line_segments;
    for (int iter = 0; iter < params.max_lane_hypotheses; ++iter) {
      if (static_cast<int>(remaining_lines.size()) < 2) {
        break;
      }

      cv::Mat * hough_ptr = (iter == 0) ? &result.hough_slice : nullptr;
      const auto seeds = voter.vote(remaining_lines, hough_ptr);
      if (seeds.empty()) {
        break;
      }

      LaneHypothesis best = pickBestSeed(seeds, working_edges, ransac, refined, params, y_max);
      if (best.score <= 0.0 || !passesQualityGate(best, params, y_max)) {
        break;
      }

      refined.push_back(best);
      remaining_lines =
        peelLinesNearCurve(remaining_lines, template_curve, best.xi, params.peel_edge_margin_px);
    }
  } else {
    const auto seeds = voter.vote(line_segments, &result.hough_slice);
    refined.resize(seeds.size());
    tbb::parallel_for(
      tbb::blocked_range<size_t>(0, seeds.size()), [&](const tbb::blocked_range<size_t> & range) {
        for (size_t i = range.begin(); i != range.end(); ++i) {
          refined[i] = ransac.fit(seeds[i], working_edges);
        }
      });
  }
  const auto t_vote1 = std::chrono::steady_clock::now();
  result.lie_vote_ms = std::chrono::duration<double, std::milli>(t_vote1 - t_vote0).count();

  if (params.use_road_manifold_joint && refined.size() >= 2) {
    RoadManifoldFitter road_fitter(params, &template_curve);
    road_fitter.refine(refined, line_segments);
  }

  result.lanes = multi_lane.extract(refined);
  result.lanes.erase(
    std::remove_if(
      result.lanes.begin(), result.lanes.end(),
      [&params, &y_max](const LaneHypothesis & lane) {
        if (
          lane.xi[0] < params.se2_vx_min || lane.xi[0] > params.se2_vx_max ||
          lane.inlier_ratio < params.min_inlier_ratio || isBorderLane(lane, params)) {
          return true;
        }
        return !passesQualityGate(lane, params, y_max);
      }),
    result.lanes.end());

  result.merges = merge_topology.analyze(result.lanes);
  result.edges = LineHoughExtractor::drawSegments(work_bev, line_segments);
  result.overlay = drawOverlay(work_bev, result.lanes, result.merges);
  const auto t1 = std::chrono::steady_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return result;
}

}  // namespace lie_lane_detection
