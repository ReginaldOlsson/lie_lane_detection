#include "lie_lane_detection/nodes/node_params.hpp"

namespace lie_lane_detection
{

PipelineParams loadIpmParams(rclcpp::Node & node)
{
  PipelineParams p;
  p.bev_width_m = node.declare_parameter<double>("bev_width_m", 20.0);
  p.bev_length_m = node.declare_parameter<double>("bev_length_m", 60.0);
  p.bev_resolution_m_per_px = node.declare_parameter<double>("bev_resolution_m_per_px", 0.05);
  p.use_manual_ipm = node.declare_parameter<bool>("use_manual_ipm", false);
  const std::vector<double> empty_points;
  p.ipm_src_points = node.declare_parameter<std::vector<double>>("ipm_src_points", empty_points);
  p.ipm_dst_points = node.declare_parameter<std::vector<double>>("ipm_dst_points", empty_points);
  p.ipm_bottom_x_min_ratio = node.declare_parameter<double>("ipm_bottom_x_min_ratio", 0.02);
  p.ipm_bottom_x_max_ratio = node.declare_parameter<double>("ipm_bottom_x_max_ratio", 0.98);
  p.ipm_bottom_y_ratio = node.declare_parameter<double>("ipm_bottom_y_ratio", 0.97);
  p.ipm_top_y_offset_ratio = node.declare_parameter<double>("ipm_top_y_offset_ratio", 0.12);
  p.ipm_top_y_min_ratio = node.declare_parameter<double>("ipm_top_y_min_ratio", 0.32);
  p.ipm_top_y_max_ratio = node.declare_parameter<double>("ipm_top_y_max_ratio", 0.72);
  p.bev_bottom_exclude_px = node.declare_parameter<double>("bev_bottom_exclude_px", 200.0);
  return p;
}

PipelineParams loadDetectionParams(rclcpp::Node & node)
{
  PipelineParams p;
  p.use_steerable_filter = node.declare_parameter<bool>("use_steerable_filter", true);
  p.connect_dashed_edges = node.declare_parameter<bool>("connect_dashed_edges", true);
  p.edge_anisotropic_blur = node.declare_parameter<bool>("edge_anisotropic_blur", true);
  p.edge_thin = node.declare_parameter<bool>("edge_thin", true);
  p.edge_low_threshold = node.declare_parameter<double>("edge_low_threshold", 25.0);
  p.edge_high_threshold = node.declare_parameter<double>("edge_high_threshold", 70.0);
  p.bev_bottom_exclude_px = node.declare_parameter<double>("bev_bottom_exclude_px", 0.0);
  p.top_k_peaks = node.declare_parameter<int>("top_k_peaks", 10);
  p.max_lane_hypotheses = node.declare_parameter<int>("max_lane_hypotheses", 8);
  p.max_output_lanes = node.declare_parameter<int>("max_output_lanes", 0);
  p.use_iterative_peeling = node.declare_parameter<bool>("use_iterative_peeling", true);
  p.vote_threshold_px = node.declare_parameter<double>("vote_threshold_px", 8.0);
  p.inlier_threshold_px = node.declare_parameter<double>("inlier_threshold_px", 10.0);
  p.min_inlier_ratio = node.declare_parameter<double>("min_inlier_ratio", 0.30);
  p.min_inliers = node.declare_parameter<int>("min_inliers", 15);
  p.min_lane_separation_px = node.declare_parameter<double>("min_lane_separation_px", 28.0);
  p.peel_edge_margin_px = node.declare_parameter<double>("peel_edge_margin_px", 10.0);
  p.inlier_dedup_ratio = node.declare_parameter<double>("inlier_dedup_ratio", 0.30);
  p.kappa_min = node.declare_parameter<double>("kappa_min", -0.28);
  p.kappa_max = node.declare_parameter<double>("kappa_max", 0.28);
  p.sigma_min = node.declare_parameter<double>("sigma_min", -0.55);
  p.sigma_max = node.declare_parameter<double>("sigma_max", 0.55);
  p.hough_hypothesis_merge_ratio =
    node.declare_parameter<double>("hough_hypothesis_merge_ratio", 0.85);
  p.min_inlier_y_coverage = node.declare_parameter<double>("min_inlier_y_coverage", 0.30);
  p.use_longitudinal_line_filter =
    node.declare_parameter<bool>("use_longitudinal_line_filter", true);
  p.longitudinal_max_deviation_rad =
    node.declare_parameter<double>("longitudinal_max_deviation_rad", 0.52);
  p.use_line_ransac_init_gate =
    node.declare_parameter<bool>("use_line_ransac_init_gate", true);
  p.ceres_hard_gate_dist_px =
    node.declare_parameter<double>("ceres_hard_gate_dist_px", 12.0);
  p.ceres_hard_gate_angle_rad =
    node.declare_parameter<double>("ceres_hard_gate_angle_rad", 0.35);
  p.ceres_heading_weight = node.declare_parameter<double>("ceres_heading_weight", 1.0);
  p.ceres_reject_corridor_center =
    node.declare_parameter<bool>("ceres_reject_corridor_center", false);
  p.ceres_corridor_half_width_px =
    node.declare_parameter<double>("ceres_corridor_half_width_px", 18.0);
  p.line_hough_threshold = node.declare_parameter<int>("line_hough_threshold", 25);
  p.line_min_length_px = node.declare_parameter<double>("line_min_length_px", 18.0);
  p.line_max_gap_px = node.declare_parameter<double>("line_max_gap_px", 12.0);
  p.use_road_manifold_joint =
    node.declare_parameter<bool>("use_road_manifold_joint", true);
  return p;
}

}  // namespace lie_lane_detection
