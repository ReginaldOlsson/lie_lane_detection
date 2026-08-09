#include "lie_lane_detection/tracking/lane_tracker.hpp"

#include "lie_lane_detection/extraction/merge_topology.hpp"
#include "lie_lane_detection/extraction/multi_lane_extractor.hpp"
#include "lie_lane_detection/fitting/manifold_ransac.hpp"
#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/preprocessing/edge_extractor.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace lie_lane_detection
{

namespace
{

double clampDouble(double v, double lo, double hi)
{
  return std::max(lo, std::min(hi, v));
}

void kalmanUpdateScalar(
  double & state, double & variance, double measurement, double measurement_var)
{
  const double k = variance / (variance + measurement_var);
  state += k * (measurement - state);
  variance = (1.0 - k) * variance;
}

void kalmanFuseXi(
  LaneTrack & track, const XiVector & measurement, double inlier_ratio,
  const LaneTrackerParams & params)
{
  const double conf = std::max(0.15, std::min(1.0, inlier_ratio));
  for (int i = 0; i < 5; ++i) {
    double r = params.r_main_base;
    if (i == 0) {
      r = params.r_main_base / conf;
    } else if (i == 2) {
      r = params.r_main_base * 2.0 / conf;
    } else {
      r = params.r_main_base * 4.0 / conf;
    }
    kalmanUpdateScalar(track.xi[i], track.p_diag[i], measurement[i], r);
  }
}

}  // namespace

void narrowParamsForTracks(
  PipelineParams & params, const std::vector<LaneTrack> & tracks, double margin_px,
  double ekf_gate_sigma)
{
  if (tracks.empty()) {
    return;
  }

  double vx_lo = std::numeric_limits<double>::max();
  double vx_hi = std::numeric_limits<double>::lowest();
  for (const auto & track : tracks) {
    const double sigma_x = std::sqrt(std::max(1.0, track.p_diag[0]));
    const double gate =
      params.use_ekf_temporal_prior ? std::max(margin_px, ekf_gate_sigma * sigma_x) : margin_px;
    vx_lo = std::min(vx_lo, track.xi[0] - gate);
    vx_hi = std::max(vx_hi, track.xi[0] + gate);
  }

  vx_lo = std::max(params.se2_vx_min, vx_lo);
  vx_hi = std::min(params.se2_vx_max, vx_hi);
  if (vx_hi <= vx_lo + 20.0) {
    return;
  }

  const double full_span = params.se2_vx_max - params.se2_vx_min;
  const double narrow_span = vx_hi - vx_lo;
  params.se2_vx_min = vx_lo;
  params.se2_vx_max = vx_hi;
  params.se2_vx_bins = std::max(
    9, std::clamp(
         static_cast<int>(std::lround(params.se2_vx_bins * narrow_span / std::max(full_span, 1.0))),
         9, params.se2_vx_bins));
}

std::vector<EdgePoint> filterEdgesInTrackCorridors(
  const std::vector<EdgePoint> & edges, const std::vector<LaneTrack> & tracks,
  const LaneTrackerParams & tracker_params, const TemplateCurve & curve, double y_max)
{
  if (tracks.empty()) {
    return edges;
  }

  std::vector<EdgePoint> filtered;
  filtered.reserve(edges.size() / 4 + 1);
  for (const auto & e : edges) {
    const Vec2 p(e.x, e.y);
    for (const auto & track : tracks) {
      const double yn = clampDouble(e.y / std::max(y_max, 1.0), 0.0, 1.0);
      const double sigma = tracker_params.sigma_0_px +
                           tracker_params.sigma_alpha_px * std::pow(yn, tracker_params.sigma_power);
      const double half_w = tracker_params.corridor_k * sigma;
      const Vec2 pred = curve.sample(track.xi, yn);
      if (
        std::abs(p.x() - pred.x()) <= half_w &&
        curve.distanceToCurve(track.xi, p) <= half_w * 1.5) {
        filtered.push_back(e);
        break;
      }
    }
  }
  return filtered;
}

LaneTracker::LaneTracker(LaneTrackerParams params) : params_(std::move(params))
{
}

void LaneTracker::reset()
{
  tracks_.clear();
  processed_frames_ = 0;
  last_video_frame_ = -1;
  next_id_ = 1;
}

double LaneTracker::lateralStdAtY(double y, double y_max) const
{
  const double yn = clampDouble(y / std::max(y_max, 1.0), 0.0, 1.0);
  return params_.sigma_0_px + params_.sigma_alpha_px * std::pow(yn, params_.sigma_power);
}

uint32_t LaneTracker::nextTrackId()
{
  return next_id_++;
}

void LaneTracker::compensateEgoMotion(double bev_lateral_delta)
{
  if (std::abs(bev_lateral_delta) < 0.05) {
    return;
  }
  for (auto & track : tracks_) {
    track.xi[0] -= bev_lateral_delta;
  }
}

void LaneTracker::predict(double dt)
{
  if (dt <= 0.0) {
    dt = 1.0 / 30.0;
  }
  for (auto & track : tracks_) {
    const XiVector xi_prev = track.xi;
    if (!params_.fixed_camera) {
      track.xi += track.xi_dot * dt;
    } else {
      track.xi_dot *= std::max(0.0, 1.0 - 5.0 * dt);
    }
    if (dt > 1e-6) {
      const XiVector delta = track.xi - xi_prev;
      track.xi_dot = track.xi_dot * 0.7 + delta * (0.3 / dt);
    }
    track.p_diag[0] += params_.q_vx * dt;
    track.p_diag[2] += params_.q_omega * dt;
    track.p_diag[3] += params_.q_kappa * dt;
    track.age++;
  }
}

void LaneTracker::clampTrackToBev(LaneTrack & track, double bev_cols, double bev_rows) const
{
  const double margin = std::max(8.0, bev_cols * 0.04);
  track.xi[0] = clampDouble(track.xi[0], margin, bev_cols - margin);
  track.xi[1] = clampDouble(track.xi[1], -bev_rows * 0.15, bev_rows * 0.15);
  track.xi[2] = clampDouble(track.xi[2], -0.35, 0.35);
  track.xi[3] = clampDouble(track.xi[3], -0.35, 0.35);
  track.xi[4] = clampDouble(track.xi[4], -0.6, 0.6);
}

void LaneTracker::pruneDuplicateTracks(double min_sep_px)
{
  std::vector<bool> drop(tracks_.size(), false);
  for (size_t i = 0; i < tracks_.size(); ++i) {
    if (drop[i]) {
      continue;
    }
    for (size_t j = i + 1; j < tracks_.size(); ++j) {
      if (drop[j]) {
        continue;
      }
      if (std::abs(tracks_[i].xi[0] - tracks_[j].xi[0]) < min_sep_px) {
        const bool drop_j =
          tracks_[j].misses > tracks_[i].misses || tracks_[j].p_diag[0] > tracks_[i].p_diag[0];
        drop[drop_j ? j : i] = true;
      }
    }
  }
  std::vector<LaneTrack> kept;
  kept.reserve(tracks_.size());
  for (size_t i = 0; i < tracks_.size(); ++i) {
    if (!drop[i]) {
      kept.push_back(tracks_[i]);
    }
  }
  tracks_ = std::move(kept);
}

void LaneTracker::stripeUpdate(
  const std::vector<EdgePoint> & edges, const TemplateCurve & curve, double bev_cols,
  double bev_rows, const PipelineParams & detect_params)
{
  const double y_max = bev_rows;
  const double y_lo = y_max * params_.stripe_y_min_ratio;
  const double y_hi = y_max * 0.96;
  ManifoldRansac ransac(detect_params, const_cast<TemplateCurve *>(&curve));

  for (auto & track : tracks_) {
    std::vector<EdgePoint> stripe_hits;
    stripe_hits.reserve(static_cast<size_t>(params_.stripe_rows));

    for (int ri = 0; ri < params_.stripe_rows; ++ri) {
      const double y = y_lo + (y_hi - y_lo) * static_cast<double>(ri) /
                                static_cast<double>(std::max(params_.stripe_rows - 1, 1));
      const double t = clampDouble(y / std::max(y_max, 1.0), 0.0, 1.0);
      const Vec2 pred = curve.sample(track.xi, t);
      const double half_w = params_.corridor_k * lateralStdAtY(y, y_max);

      const EdgePoint * best = nullptr;
      double best_mag = 0.0;
      for (const auto & e : edges) {
        if (std::abs(e.y - y) > 5.0) {
          continue;
        }
        if (std::abs(e.x - pred.x()) > half_w) {
          continue;
        }
        if (e.magnitude > best_mag) {
          best_mag = e.magnitude;
          best = &e;
        }
      }
      if (best != nullptr) {
        stripe_hits.push_back(*best);
      }
    }

    if (static_cast<int>(stripe_hits.size()) < params_.min_stripe_hits) {
      track.misses++;
      continue;
    }

    XiVector xi_new = track.xi;
    if (params_.stripe_use_gn_refine) {
      ransac.refineGaussNewton(xi_new, stripe_hits);
    } else {
      double sum_w = 0.0;
      double sum_vx = 0.0;
      for (const auto & hit : stripe_hits) {
        const double t = clampDouble(hit.y / std::max(y_max, 1.0), 0.0, 1.0);
        const Vec2 pred = curve.sample(track.xi, t);
        const double w = std::max(hit.magnitude, 1.0);
        sum_w += w;
        sum_vx += w * (hit.x - pred.x() + track.xi[0]);
      }
      xi_new[0] = sum_vx / sum_w;
    }

    XiVector delta = xi_new - track.xi;
    delta[0] = clampDouble(
      delta[0], -params_.max_stripe_lateral_delta_px, params_.max_stripe_lateral_delta_px);
    delta[2] =
      clampDouble(delta[2], -params_.max_stripe_omega_delta, params_.max_stripe_omega_delta);
    delta[1] = 0.0;
    delta[3] *= 0.35;
    delta[4] *= 0.35;

    track.xi += delta;
    track.misses = 0;
    clampTrackToBev(track, bev_cols, bev_rows);
  }
}

void LaneTracker::fuseMeasurements(const std::vector<LaneHypothesis> & detections, double bev_cols)
{
  std::vector<bool> det_used(detections.size(), false);
  std::vector<bool> track_used(tracks_.size(), false);

  struct Assoc
  {
    size_t track_idx{0};
    size_t det_idx{0};
    double cost{0.0};
  };
  std::vector<Assoc> pairs;
  pairs.reserve(tracks_.size() * detections.size());

  for (size_t ti = 0; ti < tracks_.size(); ++ti) {
    for (size_t di = 0; di < detections.size(); ++di) {
      if (detections[di].inlier_ratio < params_.min_spawn_inlier_ratio) {
        continue;
      }
      const double d = hypothesisDistance(tracks_[ti].xi, detections[di].xi);
      if (d < params_.assoc_max_dist_px) {
        pairs.push_back({ti, di, d});
      }
    }
  }
  std::sort(
    pairs.begin(), pairs.end(), [](const Assoc & a, const Assoc & b) { return a.cost < b.cost; });

  for (const auto & pair : pairs) {
    if (track_used[pair.track_idx] || det_used[pair.det_idx]) {
      continue;
    }
    auto & track = tracks_[pair.track_idx];
    const auto & det = detections[pair.det_idx];

    if (params_.snap_on_main) {
      track.xi = det.xi;
      track.p_diag = Eigen::Matrix<double, 5, 1>::Constant(25.0);
      track.xi_dot.setZero();
    } else {
      const XiVector prev_xi = track.xi;
      kalmanFuseXi(track, det.xi, det.inlier_ratio, params_);
      const double dt = 1.0 / 30.0;
      const XiVector delta = track.xi - prev_xi;
      track.xi_dot =
        (1.0 - params_.velocity_beta) * track.xi_dot + params_.velocity_beta * (delta / dt);
    }
    track.role = det.role;
    track.misses = 0;
    clampTrackToBev(track, bev_cols, bev_cols);

    track_used[pair.track_idx] = true;
    det_used[pair.det_idx] = true;
  }

  const size_t existing_track_count = tracks_.size();
  for (size_t ti = 0; ti < existing_track_count; ++ti) {
    if (!track_used[ti]) {
      tracks_[ti].misses++;
    }
  }

  for (size_t di = 0; di < detections.size(); ++di) {
    if (det_used[di]) {
      continue;
    }
    if (detections[di].inlier_ratio < params_.min_spawn_inlier_ratio) {
      continue;
    }
    if (static_cast<int>(tracks_.size()) >= params_.max_tracks) {
      break;
    }
    LaneTrack track;
    track.id = nextTrackId();
    track.xi = detections[di].xi;
    track.role = detections[di].role;
    track.p_diag = Eigen::Matrix<double, 5, 1>::Constant(25.0);
    track.xi_dot.setZero();
    clampTrackToBev(track, bev_cols, bev_cols);
    tracks_.push_back(track);
  }

  tracks_.erase(
    std::remove_if(
      tracks_.begin(), tracks_.end(),
      [&](const LaneTrack & track) { return track.misses > params_.max_misses; }),
    tracks_.end());

  pruneDuplicateTracks(22.0);
}

std::vector<LaneHypothesis> LaneTracker::tracksToHypotheses(const TemplateCurve & curve) const
{
  std::vector<LaneHypothesis> lanes;
  lanes.reserve(tracks_.size());
  for (const auto & track : tracks_) {
    LaneHypothesis lane;
    lane.lane_id = track.id;
    lane.xi = track.xi;
    lane.role = track.role;
    lane.inlier_ratio = 1.0 / (1.0 + track.p_diag[0] / 50.0);
    lane.score = lane.inlier_ratio;
    lane.polyline = curve.samplePolyline(track.xi);
    lanes.push_back(lane);
  }
  return lanes;
}

TrackedFrameResult LaneTracker::processFrame(
  const cv::Mat & bev_bgr, PipelineParams detect_params, int video_frame_index, double video_fps)
{
  const auto t0 = std::chrono::steady_clock::now();
  TrackedFrameResult result;
  if (bev_bgr.empty()) {
    return result;
  }

  configureParamsForBev(detect_params, bev_bgr.cols, bev_bgr.rows);
  const BevTrackingPrep prep = prepareBevGrayForTracking(bev_bgr, detect_params);
  const cv::Mat & work_gray = prep.gray;
  const cv::Mat & display_bgr = prep.display_bgr;
  const double y_max = bevEffectiveYMax(bev_bgr.rows, detect_params);

  TemplateCurve curve(detect_params);
  curve.setBevExtents(0.0, y_max, 0.0, static_cast<double>(bev_bgr.cols));
  MergeTopology merge_topology(detect_params, &curve);

  const double fps = std::max(video_fps, 1.0);
  double dt = 1.0 / fps;
  if (video_frame_index >= 0 && last_video_frame_ >= 0) {
    dt = static_cast<double>(video_frame_index - last_video_frame_) / fps;
  }
  if (video_frame_index >= 0) {
    last_video_frame_ = video_frame_index;
  }

  const auto t_pred0 = std::chrono::steady_clock::now();
  predict(dt);
  const auto t_pred1 = std::chrono::steady_clock::now();
  result.predict_ms = std::chrono::duration<double, std::milli>(t_pred1 - t_pred0).count();

  const bool run_main = tracks_.empty() || (processed_frames_ % params_.main_detect_interval == 0);

  EdgeExtractor edge_extractor(detect_params);
  cv::Mat edges_img;
  const auto all_edges = edge_extractor.extract(work_gray, &edges_img);
  result.edges = edges_img;

  const double border_margin =
    static_cast<double>(bev_bgr.cols) * detect_params.edge_border_margin_ratio;
  auto working_edges =
    filterBorderEdges(all_edges, border_margin, static_cast<double>(bev_bgr.cols) - border_margin);
  working_edges = filterBevYMaxEdges(working_edges, y_max);

  if (run_main) {
    PipelineParams narrow_params = detect_params;
    if (!tracks_.empty()) {
      narrowParamsForTracks(
        narrow_params, tracks_, params_.narrow_hough_margin_px, detect_params.ekf_hough_gate_sigma);
      if (params_.use_corridor_filter_on_main) {
        working_edges = filterEdgesInTrackCorridors(working_edges, tracks_, params_, curve, y_max);
        working_edges = subsampleEdges(working_edges, 12000);
      }
    }

    const auto t_main0 = std::chrono::steady_clock::now();
    const auto det = detectLanesInBev(work_gray, narrow_params);
    const auto t_main1 = std::chrono::steady_clock::now();
    result.main_ms = std::chrono::duration<double, std::milli>(t_main1 - t_main0).count();
    fuseMeasurements(det.lanes, static_cast<double>(bev_bgr.cols));
    result.ran_main_detector = true;
  } else {
    const auto t_stripe0 = std::chrono::steady_clock::now();
    stripeUpdate(working_edges, curve, static_cast<double>(bev_bgr.cols), y_max, detect_params);
    const auto t_stripe1 = std::chrono::steady_clock::now();
    result.stripe_ms = std::chrono::duration<double, std::milli>(t_stripe1 - t_stripe0).count();
    result.ran_stripe_update = true;
  }

  result.lanes = tracksToHypotheses(curve);
  MultiLaneExtractor role_extractor(detect_params);
  result.lanes = role_extractor.extract(result.lanes);
  result.merges = merge_topology.analyze(result.lanes);
  result.overlay = drawOverlay(display_bgr, result.lanes, result.merges);
  result.track_count = tracks_.size();

  const auto t1 = std::chrono::steady_clock::now();
  result.total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  ++processed_frames_;
  return result;
}

}  // namespace lie_lane_detection
