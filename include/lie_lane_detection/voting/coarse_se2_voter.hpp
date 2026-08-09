#pragma once

#include "lie_lane_detection/core/types.hpp"
#include "lie_lane_detection/fitting/ceres_lane_fitter.hpp"
#include "lie_lane_detection/fitting/line_segment_ransac_gate.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/voting/peak_refinement.hpp"

#include <opencv2/core.hpp>

#include <vector>

namespace lie_lane_detection
{

/// Coarse-to-fine SE(2) voter with soft Gaussian voting; defers (κ,σ) to Ceres.
class CoarseSE2Voter
{
public:
  struct GridConfig
  {
    int vx_bins{16};
    int vy_bins{1};
    int omega_bins{16};
    double vx_min{0.0};
    double vx_max{1.0};
    double vy_min{0.0};
    double vy_max{0.0};
    double omega_min{-0.2};
    double omega_max{0.2};
  };

  CoarseSE2Voter(const PipelineParams & params, TemplateCurve * template_curve);

  void setTemplateCurve(TemplateCurve * template_curve) { template_curve_ = template_curve; }

  std::vector<LaneHypothesis> voteEdges(
    const std::vector<EdgePoint> & edges, cv::Mat * hough_debug_slice = nullptr);

  std::vector<LaneHypothesis> voteLines(
    const std::vector<LineSegment> & lines, cv::Mat * hough_debug_slice = nullptr);

private:
  void voteEdgesIntoAccum(
    const std::vector<EdgePoint> & edges, const GridConfig & grid,
    std::vector<double> & accum) const;

  void voteLinesIntoAccum(
    const std::vector<LineSegment> & lines, const GridConfig & grid,
    std::vector<double> & accum) const;

  std::vector<SE2PeakCandidate> extractPeaks(
    const std::vector<double> & accum, const GridConfig & grid) const;

  GridConfig makeCoarseGrid() const;
  GridConfig makeRefineGrid(const SE2PeakCandidate & peak) const;

  double softVoteWeight(double dist_sq, double feature_weight, double angle_err_rad) const;

  PipelineParams params_;
  TemplateCurve * template_curve_{nullptr};
  CeresLaneFitter ceres_fitter_;
  LineSegmentRansacGate ransac_gate_;
};

}  // namespace lie_lane_detection
