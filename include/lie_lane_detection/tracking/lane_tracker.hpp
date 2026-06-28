#pragma once

#include <cstdint>
#include <vector>

#include <opencv2/core.hpp>

#include "lie_lane_detection/core/types.hpp"
#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"

namespace lie_lane_detection
{

struct LaneTrackerParams
{
  double sigma_0_px{6.0};
  double sigma_alpha_px{28.0};
  double sigma_power{1.4};
  double corridor_k{2.0};
  int stripe_rows{10};
  double velocity_beta{0.1};
  double q_vx{1.0};
  double q_omega{0.005};
  double q_kappa{0.002};
  double r_stripe_vx{12.0};
  double r_main_base{4.0};
  int main_detect_interval{3};
  int max_misses{6};
  double assoc_max_dist_px{40.0};
  double narrow_hough_margin_px{70.0};
  int max_tracks{6};
  int min_stripe_hits{4};
  /// Fixed dashcam: lanes are static in BEV — do not CV-predict lateral motion.
  bool fixed_camera{true};
  /// On main-detect frames, snap track state to measurement (not soft Kalman).
  bool snap_on_main{true};
  double max_stripe_lateral_delta_px{5.0};
  double max_stripe_omega_delta{0.04};
  double stripe_y_min_ratio{0.50};
  double min_spawn_inlier_ratio{0.45};
  bool use_corridor_filter_on_main{false};
  bool stripe_use_gn_refine{true};
};

struct LaneTrack
{
  uint32_t id{0};
  XiVector xi{XiVector::Zero()};
  XiVector xi_dot{XiVector::Zero()};
  Eigen::Matrix<double, 5, 1> p_diag{Eigen::Matrix<double, 5, 1>::Constant(100.0)};
  int age{0};
  int misses{0};
  LaneRole role{LaneRole::UNKNOWN};
};

struct TrackedFrameResult
{
  std::vector<LaneHypothesis> lanes;
  std::vector<MergeEvent> merges;
  cv::Mat overlay;
  cv::Mat edges;
  bool ran_main_detector{false};
  bool ran_stripe_update{false};
  double predict_ms{0.0};
  double stripe_ms{0.0};
  double main_ms{0.0};
  double total_ms{0.0};
  size_t track_count{0};
};

/// Narrow Lie-Hough vx search window around tracked lanes (Phase 1).
void narrowParamsForTracks(
  PipelineParams & params,
  const std::vector<LaneTrack> & tracks,
  double margin_px,
  double ekf_gate_sigma = 2.5);

/// Keep edges whose (x,y) falls inside any track corridor.
std::vector<EdgePoint> filterEdgesInTrackCorridors(
  const std::vector<EdgePoint> & edges,
  const std::vector<LaneTrack> & tracks,
  const LaneTrackerParams & tracker_params,
  const TemplateCurve & curve,
  double y_max);

class LaneTracker
{
public:
  explicit LaneTracker(LaneTrackerParams params = LaneTrackerParams{});

  void reset();
  void setParams(const LaneTrackerParams & params) {params_ = params;}

  /// Compensate estimated ego lateral motion in BEV (call before predict/process).
  void compensateEgoMotion(double bev_lateral_delta);

  const std::vector<LaneTrack> & tracks() const {return tracks_;}

  /// Pass \p video_frame_index for timestamp; \p video_fps for predict dt.
  TrackedFrameResult processFrame(
    const cv::Mat & bev_bgr,
    PipelineParams detect_params,
    int video_frame_index = -1,
    double video_fps = 30.0);

  double lateralStdAtY(double y, double y_max) const;

private:
  void predict(double dt);
  void stripeUpdate(
    const std::vector<EdgePoint> & edges,
    const TemplateCurve & curve,
    double bev_cols,
    double bev_rows,
    const PipelineParams & detect_params);
  void fuseMeasurements(
    const std::vector<LaneHypothesis> & detections,
    double bev_cols);
  void clampTrackToBev(LaneTrack & track, double bev_cols, double bev_rows) const;
  void pruneDuplicateTracks(double min_sep_px);
  std::vector<LaneHypothesis> tracksToHypotheses(const TemplateCurve & curve) const;
  uint32_t nextTrackId();

  LaneTrackerParams params_;
  std::vector<LaneTrack> tracks_;
  int processed_frames_{0};
  int last_video_frame_{-1};
  uint32_t next_id_{1};
};

}  // namespace lie_lane_detection
