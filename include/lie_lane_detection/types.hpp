#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <sophus/se2.hpp>

namespace lie_lane_detection
{

using Vec2 = Eigen::Vector2d;
using XiVector = Eigen::Matrix<double, 5, 1>;

enum class LaneRole
{
  UNKNOWN,
  LEFT_ADJACENT,
  LEFT_EGO,
  CENTER,
  RIGHT_EGO,
  RIGHT_ADJACENT
};

enum class MergeTopologyType
{
  PARALLEL,
  MERGE,
  DIVERGE
};

struct EdgePoint
{
  double x{0.0};
  double y{0.0};
  double magnitude{0.0};
  double orientation{0.0};
};

struct PipelineParams
{
  // IPM / BEV
  double bev_width_m{12.0};
  double bev_length_m{40.0};
  double bev_resolution_m_per_px{0.05};
  std::vector<double> ipm_src_points;  // 8 values: x0,y0,...
  std::vector<double> ipm_dst_points;  // 8 values in meters

  // Edge extraction
  bool use_steerable_filter{true};
  bool connect_dashed_edges{true};
  double edge_low_threshold{30.0};
  double edge_high_threshold{90.0};
  double edge_border_margin_ratio{0.07};

  // Lie-Hough Stage A (SE2)
  int se2_vx_bins{41};
  int se2_vy_bins{11};
  int se2_omega_bins{21};
  double se2_vx_min{-5.0};
  double se2_vx_max{5.0};
  double se2_vy_min{-2.0};
  double se2_vy_max{2.0};
  double se2_omega_min{-0.5};
  double se2_omega_max{0.5};
  double vote_threshold_px{4.0};

  // Lie-Hough Stage B (kappa, sigma)
  int kappa_bins{11};
  int sigma_bins{11};
  double kappa_min{-0.15};
  double kappa_max{0.15};
  double sigma_min{-0.3};
  double sigma_max{0.3};
  int top_k_peaks{8};
  int max_lane_hypotheses{8};
  double nms_se2_min{0.3};
  double nms_def_min{0.05};
  bool use_iterative_peeling{true};
  double peel_edge_margin_px{10.0};
  double min_lane_separation_px{28.0};

  // RANSAC
  int ransac_iterations{120};
  double inlier_threshold_px{5.0};
  double inlier_dedup_ratio{0.3};
  double min_inlier_ratio{0.42};
  int min_inliers{15};

  // Merge topology
  double merge_converged_threshold_m{1.5};
  double merge_lateral_rate_threshold{0.02};

  // Template sampling
  int template_samples{50};
};

struct LaneHypothesis
{
  uint32_t lane_id{0};
  XiVector xi{XiVector::Zero()};
  double vote_count{0.0};
  double score{0.0};
  double inlier_ratio{0.0};
  LaneRole role{LaneRole::UNKNOWN};
  MergeTopologyType topo{MergeTopologyType::PARALLEL};
  std::vector<Vec2> polyline;
  std::vector<bool> inlier_mask;
  std::vector<EdgePoint> supporting_edges;
};

struct MergeEvent
{
  uint32_t lane_a_id{0};
  uint32_t lane_b_id{0};
  MergeTopologyType type{MergeTopologyType::PARALLEL};
  Vec2 merge_point{Vec2::Zero()};
};

struct DebugImages
{
  cv::Mat bev;
  cv::Mat edges;
  cv::Mat overlay;
  cv::Mat hough_slice;
};

struct LaneDetectionResult
{
  std::vector<LaneHypothesis> lanes;
  std::vector<MergeEvent> merges;
  DebugImages debug;
  cv::Mat H_img2bev;
};

inline Sophus::SE2d xiToSE2(const XiVector & xi)
{
  Sophus::SE2d::Tangent se2_tangent;
  se2_tangent << xi[0], xi[1], xi[2];
  return Sophus::SE2d::exp(se2_tangent);
}

inline double hypothesisDistance(const XiVector & a, const XiVector & b, double lambda_k = 1.0, double lambda_s = 1.0)
{
  const Sophus::SE2d ga = xiToSE2(a);
  const Sophus::SE2d gb = xiToSE2(b);
  const double d_se2 = (ga.inverse() * gb).log().norm();
  return d_se2 + lambda_k * std::abs(a[3] - b[3]) + lambda_s * std::abs(a[4] - b[4]);
}

}  // namespace lie_lane_detection
