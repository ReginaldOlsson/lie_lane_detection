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

struct LineSegment
{
  double x1{0.0};
  double y1{0.0};
  double x2{0.0};
  double y2{0.0};
  double length{0.0};
  double angle{0.0};  // radians, atan2(dy, dx)
  double mx{0.0};
  double my{0.0};
};

struct PipelineParams
{
  // IPM / BEV
  double bev_width_m{12.0};
  double bev_length_m{40.0};
  double bev_resolution_m_per_px{0.05};
  std::vector<double> ipm_src_points;  // 8 values: x0,y0,...
  std::vector<double> ipm_dst_points;  // 8 values in meters
  /// Auto-IPM source trapezoid (fractions of image width/height).
  double ipm_bottom_x_min_ratio{0.06};
  double ipm_bottom_x_max_ratio{0.94};
  double ipm_bottom_y_ratio{0.97};
  double ipm_top_y_offset_ratio{0.12};
  double ipm_top_y_min_ratio{0.42};
  double ipm_top_y_max_ratio{0.72};
  bool use_manual_ipm{false};

  // Edge extraction
  bool use_steerable_filter{true};
  bool connect_dashed_edges{true};
  bool edge_anisotropic_blur{true};
  bool edge_thin{true};
  double edge_low_threshold{30.0};
  double edge_high_threshold{90.0};
  double edge_border_margin_ratio{0.07};
  /// Pixels masked at bottom of BEV (hood / truck body). 0 = disabled.
  double bev_bottom_exclude_px{0.0};
  /// BEV noise filter / preprocessing before lane detection.
  bool bev_use_sharpen{false};
  /// Compute Otsu binary for /lanes/detect/filtered debug topic.
  bool bev_use_otsu{false};
  /// When true, run detection on Otsu binary (aggressive; usually worse for lanes).
  bool bev_otsu_for_detection{false};
  int bev_gaussian_blur_ksize{0};
  int bev_morph_open_px{0};
  int bev_min_road_gray{25};

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
  /// Final cap after extraction; 0 = unlimited. Use 2 for ego left/right boundaries.
  int max_output_lanes{0};
  double nms_se2_min{0.3};
  double nms_def_min{0.05};
  bool use_iterative_peeling{true};
  double peel_edge_margin_px{10.0};
  double min_lane_separation_px{28.0};
  double hough_hypothesis_merge_ratio{0.85};
  double min_inlier_y_coverage{0.35};

  // Classical line Hough (stage 1 of line-first pipeline)
  int line_hough_threshold{25};
  double line_min_length_px{18.0};
  double line_max_gap_px{12.0};
  double line_angle_threshold_rad{0.45};

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

  // Continuous production pipeline (coarse SE2 Hough + Ceres LM)
  bool use_continuous_pipeline{false};
  bool use_coarse_pyramid_voter{false};
  bool use_soft_voting{true};
  bool use_ceres_fitter{true};
  bool use_dual_space_prune{true};
  bool use_line_kdtree{true};
  bool use_peak_mean_shift{true};
  double soft_vote_sigma_px{4.0};
  int pyramid_coarse_bins{16};
  int pyramid_refine_top_k{20};
  int pyramid_refine_factor{2};
  double ceres_huber_delta_px{5.0};
  int ceres_max_iterations{25};
  bool use_ekf_temporal_prior{true};
  double ekf_hough_gate_sigma{2.5};

  // Crosswalk / stop-bar rejection (absolute BEV longitudinal mask)
  bool use_longitudinal_line_filter{true};
  double longitudinal_max_deviation_rad{0.52};

  // Per-peak line RANSAC init gate (before Ceres)
  bool use_line_ransac_init_gate{true};
  int line_ransac_init_iterations{80};
  double line_ransac_init_angle_rad{0.35};

  // Ceres hard association + heading residuals
  double ceres_hard_gate_dist_px{12.0};
  double ceres_hard_gate_angle_rad{0.35};
  double ceres_heading_weight{1.0};
  bool ceres_reject_corridor_center{false};
  double ceres_corridor_half_width_px{18.0};
  double ceres_length_weight_floor_px{8.0};

  // Joint road manifold (shared T_ego, kappa, sigma, w_lane)
  bool use_road_manifold_joint{true};
};

/// BEV forward axis = +y; lane segments ≈ vertical (angle ≈ π/2).
inline double longitudinalDeviationRad(double segment_angle)
{
  double d = std::abs(segment_angle - CV_PI / 2.0);
  d = std::min(d, CV_PI - d);
  return d;
}

inline bool isLongitudinalSegment(double segment_angle, double max_deviation_rad)
{
  return longitudinalDeviationRad(segment_angle) <= max_deviation_rad;
}

inline double angleDiffRad(double a, double b)
{
  double d = std::abs(a - b);
  while (d > CV_PI) {
    d -= CV_PI;
  }
  return std::min(d, CV_PI - d);
}

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
