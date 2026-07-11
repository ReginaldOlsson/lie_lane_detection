#pragma once

#include <opencv2/core.hpp>

#include "lie_lane_detection/fitting/manifold_ransac.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

bool isBorderLane(const LaneHypothesis & lane, const PipelineParams & params);

bool isTooCloseToExisting(
  const LaneHypothesis & lane,
  const std::vector<LaneHypothesis> & existing,
  double min_sep_px);

bool passesQualityGate(
  const LaneHypothesis & lane,
  const PipelineParams & params,
  double image_height);

/// Highest valid forward row in BEV (excludes bottom hood crop + edge margin).
double bevEffectiveYMax(int bev_rows, const PipelineParams & params);

/// Total bottom rows to black out before edge detection (exclude + edge margin).
int bevBottomMaskRows(const PipelineParams & params);

/// Black out bottom `bev_bottom_exclude_px` rows (truck body / hood).
void maskBevBottomExclude(cv::Mat & bev_bgr, const PipelineParams & params);

/// Black out bottom hood rows plus edge margin (detection preprocess only).
void maskBevBottomForDetection(cv::Mat & bev_bgr, const PipelineParams & params);

/// Clone BEV and mask bottom exclude region when enabled.
cv::Mat prepareBevForDetection(const cv::Mat & bev_bgr, const PipelineParams & params);

/// Filter BEV edge debug image: threshold, road mask, exclude IPM trapezoid + bottom cut.
cv::Mat filterBevEdgeArtifacts(
  const cv::Mat & edges_gray,
  const cv::Mat & bev_bgr,
  const PipelineParams & params,
  const cv::Mat & H_img2bev = cv::Mat());

struct BevTrackingPrep
{
  cv::Mat display_bgr;
  cv::Mat gray;
};

/// Grayscale + road mask for lane tracking (edges, stripe, main detect); BGR kept for overlay.
BevTrackingPrep prepareBevGrayForTracking(
  const cv::Mat & bev_bgr,
  const PipelineParams & params);

struct BevPreprocessResult
{
  cv::Mat display_bgr;
  cv::Mat detect_image;
  cv::Mat filtered_debug;
  double otsu_threshold{-1.0};
};

/// Grayscale road-masked preprocess for detection; optional Otsu on filtered debug topic.
BevPreprocessResult preprocessBevForLaneDetection(
  const cv::Mat & bev_bgr,
  const PipelineParams & params);

std::vector<EdgePoint> filterBorderEdges(
  const std::vector<EdgePoint> & edges,
  double x_min,
  double x_max);

std::vector<EdgePoint> filterBevYMaxEdges(
  const std::vector<EdgePoint> & edges,
  double y_max);

std::vector<EdgePoint> peelEdgesNearCurve(
  const std::vector<EdgePoint> & edges,
  const TemplateCurve & curve,
  const XiVector & xi,
  double margin_px);

std::vector<EdgePoint> subsampleEdges(const std::vector<EdgePoint> & edges, size_t max_count);

std::vector<LineSegment> filterBorderLines(
  const std::vector<LineSegment> & lines,
  double x_min,
  double x_max);

std::vector<LineSegment> filterBevYMaxLines(
  const std::vector<LineSegment> & lines,
  double y_max);

std::vector<LineSegment> peelLinesNearCurve(
  const std::vector<LineSegment> & lines,
  const TemplateCurve & curve,
  const XiVector & xi,
  double margin_px);

/// Sample line midpoints (and endpoints for long segments) as pseudo-edges for RANSAC.
std::vector<EdgePoint> linesToRefineEdges(const std::vector<LineSegment> & lines);

LaneHypothesis pickBestSeed(
  const std::vector<LaneHypothesis> & seeds,
  const std::vector<EdgePoint> & edges,
  ManifoldRansac & ransac,
  const std::vector<LaneHypothesis> & accepted,
  const PipelineParams & params,
  double image_height);

}  // namespace lie_lane_detection
