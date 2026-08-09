#include "lie_lane_detection/voting/line_lie_hough_voter.hpp"

#include "lie_lane_detection/common/parallel.hpp"
#include "lie_lane_detection/voting/sparse_accumulator.hpp"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace lie_lane_detection
{

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

inline double angleDiff(double a, double b)
{
  double d = std::abs(a - b);
  while (d > CV_PI) {
    d -= CV_PI;
  }
  return std::min(d, CV_PI - d);
}

}  // namespace

LineLieHoughVoter::LineLieHoughVoter(const PipelineParams & params, TemplateCurve * template_curve)
: params_(params), template_curve_(template_curve)
{
}

void LineLieHoughVoter::binIndicesToXiComponents(
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

XiVector LineLieHoughVoter::binToXi(int ix, int iy, int io, int ik, int is) const
{
  XiVector xi;
  binIndicesToXiComponents(ix, iy, io, ik, is, xi[0], xi[1], xi[2], xi[3], xi[4]);
  return xi;
}

bool LineLieHoughVoter::isPeakSeparated(const XiVector & a, const XiVector & b) const
{
  return hypothesisDistance(a, b) >= params_.nms_se2_min;
}

bool LineLieHoughVoter::lineSupportsXi(
  const LineSegment & line, const XiVector & xi, double vote_thresh_sq, double angle_thresh) const
{
  // SE(2) adjoint: evaluate segment in hypothesis-local frame (translation removed).
  const Sophus::SE2d g_inv = xiToSE2(xi).inverse();
  const Vec2 p0 = g_inv * Vec2(line.x1, line.y1);
  const Vec2 p1 = g_inv * Vec2(line.x2, line.y2);
  const Vec2 pm = g_inv * Vec2(line.mx, line.my);

  XiVector xi_local = XiVector::Zero();
  xi_local[3] = xi[3];
  xi_local[4] = xi[4];

  LineSegment local_line = line;
  local_line.x1 = p0.x();
  local_line.y1 = p0.y();
  local_line.x2 = p1.x();
  local_line.y2 = p1.y();
  local_line.mx = pm.x();
  local_line.my = pm.y();

  const double dist = template_curve_->segmentDistanceToCurve(xi_local, local_line);
  if (dist * dist > vote_thresh_sq) {
    return false;
  }
  double t = 0.0;
  template_curve_->nearestPoint(xi_local, pm, &t);
  const Vec2 tau = template_curve_->tangentAt(xi_local, t);
  const double curve_angle = std::atan2(tau.y(), tau.x());
  const double local_line_angle = std::atan2(p1.y() - p0.y(), p1.x() - p0.x());
  return angleDiff(curve_angle, local_line_angle) <= angle_thresh;
}

std::vector<LaneHypothesis> LineLieHoughVoter::vote(
  const std::vector<LineSegment> & lines, cv::Mat * hough_debug_slice)
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
  const int is_mid = std::max(0, params_.sigma_bins / 2);
  const int ik_hi = std::max(0, params_.kappa_bins - 1);
  const std::vector<std::pair<int, int>> deform_presets = {
    {ik_mid, is_mid}, {ik_hi, is_mid}, {0, is_mid}, {ik_mid, std::max(0, params_.sigma_bins - 1)},
    {ik_mid, 0},
  };

  std::vector<XiVector> xi_stage_a(static_cast<size_t>(se2_cells));
  std::vector<double> vx_lut(static_cast<size_t>(vx_bins));
  for (int ix = 0; ix < vx_bins; ++ix) {
    double vx = 0.0;
    double vy = 0.0;
    double omega = 0.0;
    double kappa = 0.0;
    double sigma = 0.0;
    binIndicesToXiComponents(ix, 0, 0, 0, 0, vx, vy, omega, kappa, sigma);
    vx_lut[static_cast<size_t>(ix)] = vx;
    for (int iy = 0; iy < vy_bins; ++iy) {
      for (int io = 0; io < omega_bins; ++io) {
        xi_stage_a[static_cast<size_t>(flatBinIndex(ix, iy, io, vx_bins, vy_bins))] =
          binToXi(ix, iy, io, ik_mid, is_mid);
      }
    }
  }

  const double vote_thresh = params_.vote_threshold_px;
  const double vote_thresh_sq = vote_thresh * vote_thresh;
  const double angle_thresh = params_.line_angle_threshold_rad;

  auto minDistanceAndAngleOk = [&](int ix, int iy, int io, const LineSegment & line) {
    for (const auto & [ik, is] : deform_presets) {
      const XiVector xi = binToXi(ix, iy, io, ik, is);
      if (lineSupportsXi(line, xi, vote_thresh_sq, angle_thresh)) {
        return true;
      }
    }
    return false;
  };

  auto merge_accum = [](std::vector<double> a, const std::vector<double> & b) {
    DenseAccumulator::mergeInPlace(a, b);
    return a;
  };

  se2_accum = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, lines.size()),
    std::vector<double>(static_cast<size_t>(se2_cells), 0.0),
    [&](const tbb::blocked_range<size_t> & range, std::vector<double> local) {
      for (size_t li = range.begin(); li != range.end(); ++li) {
        const LineSegment & line = lines[li];
        const int ix_center = vxBinForX(line.mx, vx_bins, vx_min, vx_max);
        const int ix_lo = std::max(0, ix_center - ix_radius);
        const int ix_hi = std::min(vx_bins - 1, ix_center + ix_radius);

        for (int ix = ix_lo; ix <= ix_hi; ++ix) {
          const double dx = line.mx - vx_lut[static_cast<size_t>(ix)];
          if (std::abs(dx) > vote_thresh * 1.5) {
            continue;
          }
          for (int iy = 0; iy < vy_bins; ++iy) {
            for (int io = 0; io < omega_bins; ++io) {
              const int idx = flatBinIndex(ix, iy, io, vx_bins, vy_bins);
              if (minDistanceAndAngleOk(ix, iy, io, line)) {
                local[static_cast<size_t>(idx)] += line.length;
              }
            }
          }
        }
      }
      return local;
    },
    merge_accum);

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

  std::vector<LaneHypothesis> hypotheses(selected_se2.size());
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, selected_se2.size()),
    [&](const tbb::blocked_range<size_t> & range) {
      for (size_t pi = range.begin(); pi != range.end(); ++pi) {
        const SE2Peak & se2_peak = selected_se2[pi];
        double best_votes = 0.0;
        XiVector best_xi = se2_peak.xi;
        const int ix_lo = std::max(0, se2_peak.ix - 1);
        const int ix_hi = std::min(vx_bins - 1, se2_peak.ix + 1);

        int best_ik = 0;
        int best_is = 0;
        std::vector<std::pair<XiVector, double>> vote_bins;
        vote_bins.reserve(static_cast<size_t>(params_.kappa_bins * params_.sigma_bins));
        for (int ik = 0; ik < params_.kappa_bins; ++ik) {
          for (int is = 0; is < params_.sigma_bins; ++is) {
            XiVector xi = binToXi(se2_peak.ix, se2_peak.iy, se2_peak.io, ik, is);
            double votes = 0.0;
            for (const auto & line : lines) {
              if (
                line.mx < vx_lut[static_cast<size_t>(ix_lo)] - vote_thresh ||
                line.mx > vx_lut[static_cast<size_t>(ix_hi)] + vote_thresh) {
                continue;
              }
              if (lineSupportsXi(line, xi, vote_thresh_sq, angle_thresh)) {
                votes += line.length;
              }
            }
            vote_bins.emplace_back(xi, votes);
            if (votes > best_votes) {
              best_votes = votes;
              best_xi = xi;
              best_ik = ik;
              best_is = is;
            }
          }
        }

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

        for (int dik = -1; dik <= 1; ++dik) {
          const int ik = std::clamp(best_ik + dik, 0, params_.kappa_bins - 1);
          for (int dis = -1; dis <= 1; ++dis) {
            const int is = std::clamp(best_is + dis, 0, params_.sigma_bins - 1);
            if (dik == 0 && dis == 0) {
              continue;
            }
            XiVector xi = binToXi(se2_peak.ix, se2_peak.iy, se2_peak.io, ik, is);
            double votes = 0.0;
            for (const auto & line : lines) {
              if (
                line.mx < vx_lut[static_cast<size_t>(ix_lo)] - vote_thresh ||
                line.mx > vx_lut[static_cast<size_t>(ix_hi)] + vote_thresh) {
                continue;
              }
              if (lineSupportsXi(line, xi, vote_thresh_sq, angle_thresh)) {
                votes += line.length;
              }
            }
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
