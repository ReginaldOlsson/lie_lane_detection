#include "lie_lane_detection/voting/lie_hough_voter.hpp"

#include "lie_lane_detection/common/parallel.hpp"
#include "lie_lane_detection/voting/sparse_accumulator.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <utility>
#include <vector>

#ifdef LIE_HAS_CUDA
#include "lie_lane_detection/voting/cuda_voting.hpp"

#include <cstdlib>
#endif

namespace lie_lane_detection
{

LieHoughVoter::LieHoughVoter(const PipelineParams & params, TemplateCurve * template_curve)
: params_(params), template_curve_(template_curve)
{
}

void LieHoughVoter::binIndicesToXiComponents(
  int ix, int iy, int io, int ik, int is, double & vx, double & vy, double & omega, double & kappa,
  double & sigma) const
{
  const auto lerp = [](int i, int n, double vmin, double vmax) {
    if (n <= 1) {
      return vmin;
    }
    return vmin + (vmax - vmin) * static_cast<double>(i) / static_cast<double>(n - 1);
  };

  vx = lerp(ix, params_.se2_vx_bins, params_.se2_vx_min, params_.se2_vx_max);
  vy = lerp(iy, params_.se2_vy_bins, params_.se2_vy_min, params_.se2_vy_max);
  omega = lerp(io, params_.se2_omega_bins, params_.se2_omega_min, params_.se2_omega_max);
  kappa = lerp(ik, params_.kappa_bins, params_.kappa_min, params_.kappa_max);
  sigma = lerp(is, params_.sigma_bins, params_.sigma_min, params_.sigma_max);
}

XiVector LieHoughVoter::binToXi(int ix, int iy, int io, int ik, int is) const
{
  XiVector xi;
  binIndicesToXiComponents(ix, iy, io, ik, is, xi[0], xi[1], xi[2], xi[3], xi[4]);
  return xi;
}

bool LieHoughVoter::isPeakSeparated(const XiVector & a, const XiVector & b) const
{
  // Lanes are primarily distinguished by lateral offset. Require a minimum
  // lateral gap so the top-k SE(2) peaks span distinct lanes instead of
  // clustering as near-duplicate poses of a single (dominant) lane - the pose
  // distance alone lets same-vx peaks with slightly different omega/vy pass,
  // which starves other lanes once soft voting sharpens one lane's peak.
  if (std::abs(a[0] - b[0]) < 0.5 * params_.min_lane_separation_px) {
    return false;
  }
  return hypothesisDistance(a, b) >= params_.nms_se2_min;
}

namespace
{

inline int flatBinIndex(int ix, int iy, int io, int vx_bins, int vy_bins)
{
  return ix + vy_bins * vx_bins * io + vx_bins * iy;
}

inline int vxBinForX(double x, int vx_bins, double vx_min, double vx_max)
{
  if (vx_bins <= 1) {
    return 0;
  }
  const double t = (x - vx_min) / (vx_max - vx_min);
  const int ix = static_cast<int>(std::lround(t * static_cast<double>(vx_bins - 1)));
  return std::clamp(ix, 0, vx_bins - 1);
}

inline int vxSearchRadius(double vote_threshold_px, int vx_bins, double vx_min, double vx_max)
{
  const double bin_width = (vx_max - vx_min) / static_cast<double>(std::max(vx_bins - 1, 1));
  return std::max(1, static_cast<int>(std::ceil(vote_threshold_px / bin_width)) + 1);
}

#ifdef LIE_HAS_CUDA
// GPU Stage A is used when built with CUDA, a device is present, and the user
// has not disabled it via LIE_LANE_DISABLE_CUDA (set to anything but 0/empty).
bool cudaVotingEnabled()
{
  static const bool enabled = [] {
    const char * disable = std::getenv("LIE_LANE_DISABLE_CUDA");
    if (disable && disable[0] != '\0' && disable[0] != '0') {
      return false;
    }
    return cuda::isAvailable();
  }();
  return enabled;
}
#endif

}  // namespace

std::vector<LaneHypothesis> LieHoughVoter::vote(
  const std::vector<EdgePoint> & edges, cv::Mat * hough_debug_slice)
{
  const int vx_bins = params_.se2_vx_bins;
  const int vy_bins = params_.se2_vy_bins;
  const int omega_bins = params_.se2_omega_bins;
  const int se2_cells = vx_bins * vy_bins * omega_bins;

  std::vector<double> se2_accum(static_cast<size_t>(se2_cells), 0.0);

  const double vx_min = params_.se2_vx_min;
  const double vx_max = params_.se2_vx_max;
  const int ix_radius = vxSearchRadius(params_.vote_threshold_px, vx_bins, vx_min, vx_max);

  const int ik_mid = std::max(0, params_.kappa_bins / 2);
  const int ik_hi = std::max(0, params_.kappa_bins - 1);
  const int is_mid = std::max(0, params_.sigma_bins / 2);
  const int is_hi = std::max(0, params_.sigma_bins - 1);
  const std::vector<std::pair<int, int>> deform_presets = {
    {ik_mid, is_mid}, {ik_hi, is_mid}, {0, is_mid}, {ik_mid, is_hi}, {ik_mid, 0},
  };

  std::vector<XiVector> xi_stage_a(static_cast<size_t>(se2_cells));
  // Per-bin inverse SE(2) pose, so an edge can be transformed into the local
  // frame with a single multiply (the pose is shared by all deform presets).
  std::vector<Sophus::SE2d> se2_inv_lut(static_cast<size_t>(se2_cells));
  std::vector<double> vx_lut(static_cast<size_t>(vx_bins));
  for (int ix = 0; ix < vx_bins; ++ix) {
    double vx = 0.0;
    double vy = 0.0;
    double omega = 0.0;
    double kappa = 0.0;
    double sigma = 0.0;
    binIndicesToXiComponents(ix, 0, 0, 0, 0, vx, vy, omega, kappa, sigma);
    vx_lut[static_cast<size_t>(ix)] = vx;
    (void)vy;
    (void)omega;
    for (int iy = 0; iy < vy_bins; ++iy) {
      for (int io = 0; io < omega_bins; ++io) {
        const int idx = flatBinIndex(ix, iy, io, vx_bins, vy_bins);
        const XiVector xi = binToXi(ix, iy, io, ik_mid, is_mid);
        xi_stage_a[static_cast<size_t>(idx)] = xi;
        se2_inv_lut[static_cast<size_t>(idx)] = xiToSE2(xi).inverse();
      }
    }
  }

  // (kappa, sigma) values for the deform presets (independent of the pose).
  std::vector<std::pair<double, double>> preset_ks;
  preset_ks.reserve(deform_presets.size());
  for (const auto & [ik, is] : deform_presets) {
    const XiVector xi = binToXi(0, 0, 0, ik, is);
    preset_ks.emplace_back(xi[3], xi[4]);
  }

  auto minDistanceSq = [&](int ix, int iy, int io, const Vec2 & p) {
    const int idx = flatBinIndex(ix, iy, io, vx_bins, vy_bins);
    const Vec2 pl = se2_inv_lut[static_cast<size_t>(idx)] * p;
    double best = std::numeric_limits<double>::max();
    for (const auto & [kappa, sigma] : preset_ks) {
      const double d = template_curve_->distanceInLocalFrame(pl.x(), pl.y(), kappa, sigma);
      best = std::min(best, d * d);
    }
    return best;
  };

  const double vote_thresh = params_.vote_threshold_px;
  const double vote_thresh_sq = vote_thresh * vote_thresh;

  // Soft (Gaussian) voting weights in-gate contributions by exp(-d^2/(2 sigma^2))
  // for sharper, more stable peaks; falls back to raw magnitude when disabled.
  const bool soft_voting = params_.use_soft_voting;
  const double soft_sigma = std::max(params_.soft_vote_sigma_px, 1e-3);
  const double inv_two_sigma_sq = 1.0 / (2.0 * soft_sigma * soft_sigma);
  auto voteWeight = [&](double mag, double d_sq) {
    return soft_voting ? mag * std::exp(-d_sq * inv_two_sigma_sq) : mag;
  };

  auto merge_accum = [](std::vector<double> a, const std::vector<double> & b) {
    DenseAccumulator::mergeInPlace(a, b);
    return a;
  };

  // Stage A: localized SE(2) voting. Prefer the GPU kernel when available;
  // otherwise use the TBB parallel_reduce below. Both produce the same accum.
  bool stage_a_done = false;
#ifdef LIE_HAS_CUDA
  if (cudaVotingEnabled() && !edges.empty()) {
    // Flatten the per-bin inverse pose LUT to [R00,R01,R10,R11,tx,ty] per cell,
    // matching cuda_voting.cu's expected layout and index order.
    std::vector<float> lut_flat(static_cast<size_t>(se2_cells) * 6);
    for (int i = 0; i < se2_cells; ++i) {
      const Sophus::SE2d & g = se2_inv_lut[static_cast<size_t>(i)];
      const Eigen::Matrix2d R = g.rotationMatrix();
      const Vec2 t = g.translation();
      float * dst = lut_flat.data() + static_cast<size_t>(i) * 6;
      dst[0] = static_cast<float>(R(0, 0));
      dst[1] = static_cast<float>(R(0, 1));
      dst[2] = static_cast<float>(R(1, 0));
      dst[3] = static_cast<float>(R(1, 1));
      dst[4] = static_cast<float>(t.x());
      dst[5] = static_cast<float>(t.y());
    }

    std::vector<float> ex(edges.size());
    std::vector<float> ey(edges.size());
    std::vector<float> emag(edges.size());
    for (size_t i = 0; i < edges.size(); ++i) {
      ex[i] = static_cast<float>(edges[i].x);
      ey[i] = static_cast<float>(edges[i].y);
      emag[i] = static_cast<float>(edges[i].magnitude);
    }

    std::vector<float> vx_lut_f(static_cast<size_t>(vx_bins));
    for (int ix = 0; ix < vx_bins; ++ix) {
      vx_lut_f[static_cast<size_t>(ix)] = static_cast<float>(vx_lut[static_cast<size_t>(ix)]);
    }

    std::vector<float> preset_kappa;
    std::vector<float> preset_sigma;
    preset_kappa.reserve(preset_ks.size());
    preset_sigma.reserve(preset_ks.size());
    for (const auto & [kappa, sigma] : preset_ks) {
      preset_kappa.push_back(static_cast<float>(kappa));
      preset_sigma.push_back(static_cast<float>(sigma));
    }

    cuda::StageAConfig cfg;
    cfg.vx_bins = vx_bins;
    cfg.vy_bins = vy_bins;
    cfg.omega_bins = omega_bins;
    cfg.num_presets = static_cast<int>(preset_ks.size());
    cfg.ix_radius = ix_radius;
    cfg.vx_min = static_cast<float>(vx_min);
    cfg.vx_max = static_cast<float>(vx_max);
    cfg.y_min = static_cast<float>(template_curve_->localFrameYMin());
    cfg.y_span = static_cast<float>(template_curve_->localFrameYSpan());
    cfg.vote_thresh = static_cast<float>(vote_thresh);
    cfg.soft_voting = soft_voting ? 1 : 0;
    cfg.inv_two_sigma_sq = static_cast<float>(inv_two_sigma_sq);

    stage_a_done = cuda::stageAVote(
      cfg, ex, ey, emag, lut_flat, vx_lut_f, preset_kappa, preset_sigma, se2_accum);
  }
#endif

  if (!stage_a_done) {
    // Stage A (CPU): localized SE(2) voting (TBB parallel_reduce).
    se2_accum = tbb::parallel_reduce(
      tbb::blocked_range<size_t>(0, edges.size()),
      std::vector<double>(static_cast<size_t>(se2_cells), 0.0),
      [&](const tbb::blocked_range<size_t> & range, std::vector<double> local) {
        for (size_t ei = range.begin(); ei != range.end(); ++ei) {
          const EdgePoint & edge = edges[ei];
          const Vec2 p(edge.x, edge.y);
          const int ix_center = vxBinForX(edge.x, vx_bins, vx_min, vx_max);
          const int ix_lo = std::max(0, ix_center - ix_radius);
          const int ix_hi = std::min(vx_bins - 1, ix_center + ix_radius);

          for (int ix = ix_lo; ix <= ix_hi; ++ix) {
            const double dx = edge.x - vx_lut[static_cast<size_t>(ix)];
            if (std::abs(dx) > vote_thresh * 1.5) {
              continue;
            }
            for (int iy = 0; iy < vy_bins; ++iy) {
              for (int io = 0; io < omega_bins; ++io) {
                const int idx = flatBinIndex(ix, iy, io, vx_bins, vy_bins);
                const double d_sq = minDistanceSq(ix, iy, io, p);
                if (d_sq < vote_thresh_sq) {
                  local[static_cast<size_t>(idx)] += voteWeight(edge.magnitude, d_sq);
                }
              }
            }
          }
        }
        return local;
      },
      merge_accum);
  }

  std::vector<SE2Peak> se2_peaks;
  se2_peaks.reserve(static_cast<size_t>(se2_cells / 8));
  for (int io = 0; io < omega_bins; ++io) {
    for (int iy = 0; iy < vy_bins; ++iy) {
      for (int ix = 0; ix < vx_bins; ++ix) {
        const int idx = flatBinIndex(ix, iy, io, vx_bins, vy_bins);
        const double votes = se2_accum[static_cast<size_t>(idx)];
        if (votes <= 0.0) {
          continue;
        }
        SE2Peak peak;
        peak.ix = ix;
        peak.iy = iy;
        peak.io = io;
        peak.votes = votes;
        peak.xi = xi_stage_a[static_cast<size_t>(idx)];
        se2_peaks.push_back(peak);
      }
    }
  }

  std::sort(se2_peaks.begin(), se2_peaks.end(), [](const SE2Peak & a, const SE2Peak & b) {
    return a.votes > b.votes;
  });

  std::vector<SE2Peak> selected_se2;
  selected_se2.reserve(static_cast<size_t>(params_.top_k_peaks));
  for (const auto & peak : se2_peaks) {
    bool separated = true;
    for (const auto & kept : selected_se2) {
      if (!isPeakSeparated(peak.xi, kept.xi)) {
        separated = false;
        break;
      }
    }
    if (separated) {
      selected_se2.push_back(peak);
    }
    if (static_cast<int>(selected_se2.size()) >= params_.top_k_peaks) {
      break;
    }
  }

  if (hough_debug_slice) {
    *hough_debug_slice = cv::Mat::zeros(vy_bins, vx_bins, CV_32F);
    for (const auto & peak : se2_peaks) {
      hough_debug_slice->at<float>(peak.iy, peak.ix) = static_cast<float>(peak.votes);
    }
    cv::normalize(*hough_debug_slice, *hough_debug_slice, 0, 255, cv::NORM_MINMAX);
    hough_debug_slice->convertTo(*hough_debug_slice, CV_8U);
  }

  // Stage B: refine (kappa, sigma) per peak in parallel.
  std::vector<LaneHypothesis> hypotheses(selected_se2.size());
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, selected_se2.size()),
    [&](const tbb::blocked_range<size_t> & range) {
      // (a=lateral, b=forward) local-frame edge coords + magnitude, reused
      // across all (kappa,sigma) bins of a peak.
      struct LocalEdge
      {
        double a;
        double b;
        double mag;
      };
      std::vector<LocalEdge> local_edges;

      for (size_t pi = range.begin(); pi != range.end(); ++pi) {
        const SE2Peak & se2_peak = selected_se2[pi];
        double best_votes = 0.0;
        XiVector best_xi = se2_peak.xi;
        const int ix_lo = std::max(0, se2_peak.ix - 1);
        const int ix_hi = std::min(vx_bins - 1, se2_peak.ix + 1);

        // The SE(2) pose is identical for every (kappa,sigma) bin of this peak,
        // so transform the gated edges into the local frame exactly once instead
        // of recomputing xiToSE2()/inverse() inside the innermost bin loop.
        const double x_gate_lo = vx_lut[static_cast<size_t>(ix_lo)] - vote_thresh;
        const double x_gate_hi = vx_lut[static_cast<size_t>(ix_hi)] + vote_thresh;
        const Sophus::SE2d g_inv = xiToSE2(se2_peak.xi).inverse();
        local_edges.clear();
        for (const auto & edge : edges) {
          if (edge.x < x_gate_lo || edge.x > x_gate_hi) {
            continue;
          }
          const Vec2 pl = g_inv * Vec2(edge.x, edge.y);
          local_edges.push_back({pl.x(), pl.y(), edge.magnitude});
        }

        auto tallyVotes = [&](double kappa, double sigma) {
          double votes = 0.0;
          for (const auto & le : local_edges) {
            const double d = template_curve_->distanceInLocalFrame(le.a, le.b, kappa, sigma);
            const double d_sq = d * d;
            if (d_sq < vote_thresh_sq) {
              votes += voteWeight(le.mag, d_sq);
            }
          }
          return votes;
        };

        int best_ik = 0;
        int best_is = 0;
        std::vector<std::pair<XiVector, double>> vote_bins;
        vote_bins.reserve(static_cast<size_t>(params_.kappa_bins * params_.sigma_bins));
        for (int ik = 0; ik < params_.kappa_bins; ++ik) {
          for (int is = 0; is < params_.sigma_bins; ++is) {
            XiVector xi = binToXi(se2_peak.ix, se2_peak.iy, se2_peak.io, ik, is);
            const double votes = tallyVotes(xi[3], xi[4]);
            vote_bins.emplace_back(xi, votes);
            if (votes > best_votes) {
              best_votes = votes;
              best_xi = xi;
              best_ik = ik;
              best_is = is;
            }
          }
        }

        // RHT-style: average similar hypotheses (Lewis et al. Algorithm 2).
        if (best_votes > 0.0) {
          const double merge_thresh = best_votes * params_.hough_hypothesis_merge_ratio;
          XiVector xi_avg = XiVector::Zero();
          double weight_sum = 0.0;
          for (const auto & [xi, votes] : vote_bins) {
            if (votes >= merge_thresh) {
              xi_avg += votes * xi;
              weight_sum += votes;
            }
          }
          if (weight_sum > 0.0) {
            best_xi = xi_avg / weight_sum;
          }
        }

        // Local fine search around coarse (kappa, sigma) peak.
        for (int dik = -1; dik <= 1; ++dik) {
          const int ik = std::clamp(best_ik + dik, 0, params_.kappa_bins - 1);
          for (int dis = -1; dis <= 1; ++dis) {
            const int is = std::clamp(best_is + dis, 0, params_.sigma_bins - 1);
            if (dik == 0 && dis == 0) {
              continue;
            }
            XiVector xi = binToXi(se2_peak.ix, se2_peak.iy, se2_peak.io, ik, is);
            const double votes = tallyVotes(xi[3], xi[4]);
            if (votes > best_votes) {
              best_votes = votes;
              best_xi = xi;
            }
          }
        }

        LaneHypothesis hyp;
        hyp.xi = best_xi;
        hyp.vote_count = best_votes;
        hyp.score = best_votes;
        hypotheses[pi] = hyp;
      }
    });

  std::vector<LaneHypothesis> final_hyps;
  std::sort(
    hypotheses.begin(), hypotheses.end(),
    [](const LaneHypothesis & a, const LaneHypothesis & b) { return a.score > b.score; });
  for (const auto & hyp : hypotheses) {
    bool keep = true;
    for (const auto & kept : final_hyps) {
      if (hypothesisDistance(hyp.xi, kept.xi) < params_.nms_se2_min) {
        keep = false;
        break;
      }
    }
    if (keep) {
      final_hyps.push_back(hyp);
    }
    if (static_cast<int>(final_hyps.size()) >= params_.top_k_peaks) {
      break;
    }
  }

  return final_hyps;
}

}  // namespace lie_lane_detection
