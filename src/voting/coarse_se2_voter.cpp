#include "lie_lane_detection/voting/coarse_se2_voter.hpp"

#include "lie_lane_detection/common/parallel.hpp"
#include "lie_lane_detection/voting/sparse_accumulator.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace lie_lane_detection
{

namespace
{

inline int flatBinIndex(int ix, int iy, int io, int vx_bins, int vy_bins)
{
  return ix + vy_bins * vx_bins * io + vx_bins * iy;
}

inline double lerpBin(int i, int n, double vmin, double vmax)
{
  if (n <= 1) {
    return vmin;
  }
  return vmin + (vmax - vmin) * static_cast<double>(i) / static_cast<double>(n - 1);
}

inline double angleDiff(double a, double b)
{
  double d = std::abs(a - b);
  while (d > CV_PI) {
    d -= CV_PI;
  }
  return std::min(d, CV_PI - d);
}

XiVector gridToXi(const CoarseSE2Voter::GridConfig & grid, int ix, int iy, int io)
{
  XiVector xi = XiVector::Zero();
  xi[0] = lerpBin(ix, grid.vx_bins, grid.vx_min, grid.vx_max);
  xi[1] = lerpBin(iy, grid.vy_bins, grid.vy_min, grid.vy_max);
  xi[2] = lerpBin(io, grid.omega_bins, grid.omega_min, grid.omega_max);
  return xi;
}

void splatTrilinear(
  std::vector<double> & accum, const CoarseSE2Voter::GridConfig & grid, double fx, double fy,
  double fo, double weight)
{
  const int vx_bins = grid.vx_bins;
  const int vy_bins = grid.vy_bins;
  const int omega_bins = grid.omega_bins;
  const auto clampi = [](int v, int lo, int hi) { return std::clamp(v, lo, hi); };

  const double ix_f = fx * static_cast<double>(std::max(vx_bins - 1, 1));
  const double iy_f = fy * static_cast<double>(std::max(vy_bins - 1, 1));
  const double io_f = fo * static_cast<double>(std::max(omega_bins - 1, 1));

  const int ix0 = clampi(static_cast<int>(std::floor(ix_f)), 0, vx_bins - 1);
  const int iy0 = clampi(static_cast<int>(std::floor(iy_f)), 0, vy_bins - 1);
  const int io0 = clampi(static_cast<int>(std::floor(io_f)), 0, omega_bins - 1);
  const int ix1 = clampi(ix0 + 1, 0, vx_bins - 1);
  const int iy1 = clampi(iy0 + 1, 0, vy_bins - 1);
  const int io1 = clampi(io0 + 1, 0, omega_bins - 1);

  const double tx = ix_f - static_cast<double>(ix0);
  const double ty = iy_f - static_cast<double>(iy0);
  const double to = io_f - static_cast<double>(io0);

  const auto add = [&](int ix, int iy, int io, double w) {
    const size_t idx = static_cast<size_t>(flatBinIndex(ix, iy, io, vx_bins, vy_bins));
    if (idx < accum.size()) {
      accum[idx] += weight * w;
    }
  };

  for (int dio = 0; dio <= 1; ++dio) {
    for (int diy = 0; diy <= 1; ++diy) {
      for (int dix = 0; dix <= 1; ++dix) {
        const double wx = dix == 0 ? (1.0 - tx) : tx;
        const double wy = diy == 0 ? (1.0 - ty) : ty;
        const double wo = dio == 0 ? (1.0 - to) : to;
        const int ix = dix == 0 ? ix0 : ix1;
        const int iy = diy == 0 ? iy0 : iy1;
        const int io = dio == 0 ? io0 : io1;
        add(ix, iy, io, wx * wy * wo);
      }
    }
  }
}

}  // namespace

CoarseSE2Voter::CoarseSE2Voter(const PipelineParams & params, TemplateCurve * template_curve)
: params_(params),
  template_curve_(template_curve),
  ceres_fitter_(params, template_curve),
  ransac_gate_(params, template_curve)
{
}

CoarseSE2Voter::GridConfig CoarseSE2Voter::makeCoarseGrid() const
{
  GridConfig g;
  g.vx_bins = std::max(8, params_.pyramid_coarse_bins);
  g.vy_bins = std::max(1, params_.se2_vy_bins > 1 ? params_.pyramid_coarse_bins / 4 : 1);
  g.omega_bins = std::max(8, params_.pyramid_coarse_bins);
  g.vx_min = params_.se2_vx_min;
  g.vx_max = params_.se2_vx_max;
  g.vy_min = params_.se2_vy_min;
  g.vy_max = params_.se2_vy_max;
  g.omega_min = params_.se2_omega_min;
  g.omega_max = params_.se2_omega_max;
  return g;
}

CoarseSE2Voter::GridConfig CoarseSE2Voter::makeRefineGrid(const SE2PeakCandidate & peak) const
{
  const int factor = std::max(2, params_.pyramid_refine_factor);
  GridConfig g = makeCoarseGrid();
  g.vx_bins = std::min(params_.se2_vx_bins, g.vx_bins * factor);
  g.omega_bins = std::min(params_.se2_omega_bins, g.omega_bins * factor);
  const double vx_span = (params_.se2_vx_max - params_.se2_vx_min) / static_cast<double>(g.vx_bins);
  const double om_span =
    (params_.se2_omega_max - params_.se2_omega_min) / static_cast<double>(g.omega_bins);
  g.vx_min = std::max(params_.se2_vx_min, peak.vx - vx_span * 3.0);
  g.vx_max = std::min(params_.se2_vx_max, peak.vx + vx_span * 3.0);
  g.omega_min = std::max(params_.se2_omega_min, peak.omega - om_span * 3.0);
  g.omega_max = std::min(params_.se2_omega_max, peak.omega + om_span * 3.0);
  return g;
}

double CoarseSE2Voter::softVoteWeight(
  double dist_sq, double feature_weight, double angle_err_rad) const
{
  const double sigma_sq = std::max(1.0, params_.soft_vote_sigma_px * params_.soft_vote_sigma_px);
  const double vote_thresh_sq = params_.vote_threshold_px * params_.vote_threshold_px;
  if (dist_sq > vote_thresh_sq * 4.0) {
    return 0.0;
  }
  const double w_dist = std::exp(-dist_sq / (2.0 * sigma_sq));
  const double w_ang = params_.use_soft_voting
                         ? std::max(0.0, std::cos(angle_err_rad))
                         : (angle_err_rad < params_.line_angle_threshold_rad ? 1.0 : 0.0);
  return feature_weight * w_dist * w_ang;
}

void CoarseSE2Voter::voteEdgesIntoAccum(
  const std::vector<EdgePoint> & edges, const GridConfig & grid, std::vector<double> & accum) const
{
  const int se2_cells = grid.vx_bins * grid.vy_bins * grid.omega_bins;
  accum.assign(static_cast<size_t>(se2_cells), 0.0);

  const double vote_thresh_sq = params_.vote_threshold_px * params_.vote_threshold_px;
  const double vx_span = std::max(1e-6, grid.vx_max - grid.vx_min);
  const double vx_bin_w = vx_span / static_cast<double>(std::max(grid.vx_bins - 1, 1));
  const int ix_radius =
    std::max(1, static_cast<int>(std::ceil(params_.vote_threshold_px / vx_bin_w)) + 1);

  accum = tbb::parallel_reduce(
    tbb::blocked_range<size_t>(0, edges.size()),
    std::vector<double>(static_cast<size_t>(se2_cells), 0.0),
    [&](const tbb::blocked_range<size_t> & range, std::vector<double> local) {
      for (size_t ei = range.begin(); ei != range.end(); ++ei) {
        const EdgePoint & edge = edges[ei];
        if (
          params_.use_longitudinal_line_filter &&
          !isLongitudinalSegment(edge.orientation, params_.longitudinal_max_deviation_rad)) {
          continue;
        }
        const Vec2 p(edge.x, edge.y);
        const double vx_t = (edge.x - grid.vx_min) / vx_span;
        const int ix_center = std::clamp(
          static_cast<int>(std::lround(vx_t * static_cast<double>(std::max(grid.vx_bins - 1, 1)))),
          0, grid.vx_bins - 1);
        const int ix_lo = std::max(0, ix_center - ix_radius);
        const int ix_hi = std::min(grid.vx_bins - 1, ix_center + ix_radius);

        for (int io = 0; io < grid.omega_bins; ++io) {
          for (int iy = 0; iy < grid.vy_bins; ++iy) {
            for (int ix = ix_lo; ix <= ix_hi; ++ix) {
              const XiVector xi = gridToXi(grid, ix, iy, io);
              const Vec2 q = template_curve_->nearestPoint(xi, p);
              const double dx = p.x() - q.x();
              const double dy = p.y() - q.y();
              const double dist_sq = dx * dx + dy * dy;
              if (dist_sq >= vote_thresh_sq * 4.0) {
                continue;
              }
              const Vec2 tau = template_curve_->tangentAt(xi, 0.5);
              const double curve_angle = std::atan2(tau.y(), tau.x());
              const double w =
                softVoteWeight(dist_sq, edge.magnitude, angleDiff(curve_angle, edge.orientation));
              if (w <= 0.0) {
                continue;
              }
              if (params_.use_soft_voting) {
                const double fx = (xi[0] - grid.vx_min) / vx_span;
                const double fy = (xi[1] - grid.vy_min) / std::max(1e-6, grid.vy_max - grid.vy_min);
                const double fo =
                  (xi[2] - grid.omega_min) / std::max(1e-6, grid.omega_max - grid.omega_min);
                splatTrilinear(local, grid, fx, fy, fo, w);
              } else {
                const int idx = flatBinIndex(ix, iy, io, grid.vx_bins, grid.vy_bins);
                local[static_cast<size_t>(idx)] += w;
              }
            }
          }
        }
      }
      return local;
    },
    [](std::vector<double> a, const std::vector<double> & b) {
      DenseAccumulator::mergeInPlace(a, b);
      return a;
    });
}

void CoarseSE2Voter::voteLinesIntoAccum(
  const std::vector<LineSegment> & lines, const GridConfig & grid,
  std::vector<double> & accum) const
{
  std::vector<EdgePoint> pseudo;
  pseudo.reserve(lines.size());
  for (const auto & line : lines) {
    EdgePoint e;
    e.x = line.mx;
    e.y = line.my;
    e.magnitude = line.length;
    e.orientation = line.angle;
    pseudo.push_back(e);
  }
  voteEdgesIntoAccum(pseudo, grid, accum);
}

std::vector<SE2PeakCandidate> CoarseSE2Voter::extractPeaks(
  const std::vector<double> & accum, const GridConfig & grid) const
{
  std::vector<SE2PeakCandidate> peaks;
  for (int io = 0; io < grid.omega_bins; ++io) {
    for (int iy = 0; iy < grid.vy_bins; ++iy) {
      for (int ix = 0; ix < grid.vx_bins; ++ix) {
        const int idx = flatBinIndex(ix, iy, io, grid.vx_bins, grid.vy_bins);
        const double votes = accum[static_cast<size_t>(idx)];
        if (votes <= 0.0) {
          continue;
        }
        auto peak = refinePeakQuadratic(
          accum, grid.vx_bins, grid.vy_bins, grid.omega_bins, ix, iy, io, params_);
        peak.votes = votes;
        peaks.push_back(peak);
      }
    }
  }
  std::sort(peaks.begin(), peaks.end(), [](const SE2PeakCandidate & a, const SE2PeakCandidate & b) {
    return a.votes > b.votes;
  });
  if (static_cast<int>(peaks.size()) > params_.pyramid_refine_top_k) {
    peaks.resize(static_cast<size_t>(params_.pyramid_refine_top_k));
  }
  if (params_.use_peak_mean_shift) {
    peaks = meanShiftPeaks(peaks, params_.nms_se2_min);
  }
  return peaks;
}

std::vector<LaneHypothesis> CoarseSE2Voter::voteEdges(
  const std::vector<EdgePoint> & edges, cv::Mat * hough_debug_slice)
{
  if (edges.empty() || template_curve_ == nullptr) {
    return {};
  }

  const GridConfig coarse = makeCoarseGrid();
  std::vector<double> coarse_accum;
  voteEdgesIntoAccum(edges, coarse, coarse_accum);
  auto peaks = extractPeaks(coarse_accum, coarse);

  // Refine top peaks on finer local grids (pyramid level 1).
  std::vector<SE2PeakCandidate> refined_peaks;
  refined_peaks.reserve(peaks.size());
  for (const auto & peak : peaks) {
    const GridConfig fine = makeRefineGrid(peak);
    std::vector<double> fine_accum;
    voteEdgesIntoAccum(edges, fine, fine_accum);
    auto local_peaks = extractPeaks(fine_accum, fine);
    if (!local_peaks.empty()) {
      refined_peaks.push_back(local_peaks.front());
    } else {
      refined_peaks.push_back(peak);
    }
  }
  peaks = std::move(refined_peaks);

  if (hough_debug_slice) {
    *hough_debug_slice = cv::Mat::zeros(coarse.vy_bins, coarse.vx_bins, CV_32F);
    for (int iy = 0; iy < coarse.vy_bins; ++iy) {
      for (int ix = 0; ix < coarse.vx_bins; ++ix) {
        double sum = 0.0;
        for (int io = 0; io < coarse.omega_bins; ++io) {
          sum += coarse_accum[static_cast<size_t>(
            flatBinIndex(ix, iy, io, coarse.vx_bins, coarse.vy_bins))];
        }
        hough_debug_slice->at<float>(iy, ix) = static_cast<float>(sum);
      }
    }
    cv::normalize(*hough_debug_slice, *hough_debug_slice, 0, 255, cv::NORM_MINMAX);
    hough_debug_slice->convertTo(*hough_debug_slice, CV_8U);
  }

  std::vector<LaneHypothesis> hypotheses;
  hypotheses.reserve(peaks.size());
  for (const auto & peak : peaks) {
    LaneHypothesis seed;
    seed.xi = peak.xi;
    seed.xi[3] = 0.0;
    seed.xi[4] = 0.0;
    seed.vote_count = peak.votes;
    seed.score = peak.votes;

    LaneHypothesis hyp = params_.use_ceres_fitter ? ceres_fitter_.fitEdges(seed, edges) : seed;
    hyp.vote_count = peak.votes;
    if (hyp.score <= 0.0) {
      hyp.score = peak.votes;
    }

    bool separated = true;
    for (const auto & kept : hypotheses) {
      if (hypothesisDistance(hyp.xi, kept.xi) < params_.nms_se2_min) {
        separated = false;
        break;
      }
    }
    if (separated) {
      hypotheses.push_back(hyp);
    }
    if (static_cast<int>(hypotheses.size()) >= params_.top_k_peaks) {
      break;
    }
  }
  return hypotheses;
}

std::vector<LaneHypothesis> CoarseSE2Voter::voteLines(
  const std::vector<LineSegment> & lines, cv::Mat * hough_debug_slice)
{
  std::vector<EdgePoint> pseudo;
  pseudo.reserve(lines.size() * 3);
  for (const auto & line : lines) {
    const auto add = [&](double x, double y) {
      EdgePoint e;
      e.x = x;
      e.y = y;
      e.magnitude = line.length;
      e.orientation = line.angle;
      pseudo.push_back(e);
    };
    add(line.x1, line.y1);
    add(line.mx, line.my);
    add(line.x2, line.y2);
  }
  auto hyps = voteEdges(pseudo, hough_debug_slice);
  if (params_.use_ceres_fitter) {
    std::vector<LaneHypothesis> filtered;
    filtered.reserve(hyps.size());
    for (auto & hyp : hyps) {
      std::vector<LineSegment> fit_lines = lines;
      if (params_.use_line_ransac_init_gate) {
        const auto gated = ransac_gate_.filterSeed(hyp, lines);
        if (!gated.valid) {
          continue;
        }
        hyp.xi = gated.xi;
        fit_lines = gated.inlier_lines;
      }
      hyp = ceres_fitter_.fitLines(hyp, fit_lines);
      filtered.push_back(hyp);
    }
    hyps = std::move(filtered);
  }
  return hyps;
}

}  // namespace lie_lane_detection
