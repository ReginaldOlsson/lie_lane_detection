#include "lie_lane_detection/voting/peak_refinement.hpp"

#include <algorithm>
#include <cmath>

namespace lie_lane_detection
{

namespace
{

inline int flatBinIndex(int ix, int iy, int io, int vx_bins, int vy_bins)
{
  return ix + vy_bins * vx_bins * io + vx_bins * iy;
}

inline double binCenter(int i, int n, double vmin, double vmax)
{
  if (n <= 1) {
    return vmin;
  }
  return vmin + (vmax - vmin) * static_cast<double>(i) / static_cast<double>(n - 1);
}

double sampleAccum(
  const std::vector<double> & accum,
  int vx_bins,
  int vy_bins,
  int omega_bins,
  int ix,
  int iy,
  int io)
{
  ix = std::clamp(ix, 0, vx_bins - 1);
  iy = std::clamp(iy, 0, vy_bins - 1);
  io = std::clamp(io, 0, omega_bins - 1);
  return accum[static_cast<size_t>(flatBinIndex(ix, iy, io, vx_bins, vy_bins))];
}

}  // namespace

SE2PeakCandidate refinePeakQuadratic(
  const std::vector<double> & accum,
  int vx_bins,
  int vy_bins,
  int omega_bins,
  int ix,
  int iy,
  int io,
  const PipelineParams & params)
{
  SE2PeakCandidate out;
  out.votes = sampleAccum(accum, vx_bins, vy_bins, omega_bins, ix, iy, io);

  // 1D parabolic fit along vx (dominant highway axis).
  const double c = sampleAccum(accum, vx_bins, vy_bins, omega_bins, ix, iy, io);
  const double l = sampleAccum(accum, vx_bins, vy_bins, omega_bins, ix - 1, iy, io);
  const double r = sampleAccum(accum, vx_bins, vy_bins, omega_bins, ix + 1, iy, io);
  double sub_ix = static_cast<double>(ix);
  if (l + r - 2.0 * c != 0.0) {
    sub_ix += 0.5 * (l - r) / (l + r - 2.0 * c);
  }
  sub_ix = std::clamp(sub_ix, 0.0, static_cast<double>(vx_bins - 1));

  out.vx = binCenter(static_cast<int>(std::lround(sub_ix)), vx_bins, params.se2_vx_min, params.se2_vx_max);
  out.vy = binCenter(iy, vy_bins, params.se2_vy_min, params.se2_vy_max);
  out.omega = binCenter(io, omega_bins, params.se2_omega_min, params.se2_omega_max);
  out.xi[0] = out.vx;
  out.xi[1] = out.vy;
  out.xi[2] = out.omega;
  return out;
}

std::vector<SE2PeakCandidate> meanShiftPeaks(
  const std::vector<SE2PeakCandidate> & peaks,
  double bandwidth,
  int max_iterations)
{
  if (peaks.empty()) {
    return {};
  }
  const double bw_sq = std::max(1e-6, bandwidth * bandwidth);
  std::vector<SE2PeakCandidate> modes;
  modes.reserve(peaks.size());
  std::vector<bool> merged(peaks.size(), false);

  for (size_t i = 0; i < peaks.size(); ++i) {
    if (merged[i]) {
      continue;
    }
    SE2PeakCandidate mode = peaks[i];
    for (int iter = 0; iter < max_iterations; ++iter) {
      double w_sum = 0.0;
      XiVector xi_acc = XiVector::Zero();
      double vote_sum = 0.0;
      for (size_t j = 0; j < peaks.size(); ++j) {
        const double d = hypothesisDistance(mode.xi, peaks[j].xi);
        const double w = std::exp(-d * d / bw_sq) * std::max(1.0, peaks[j].votes);
        xi_acc += w * peaks[j].xi;
        vote_sum += w * peaks[j].votes;
        w_sum += w;
      }
      if (w_sum < 1e-9) {
        break;
      }
      const XiVector next = xi_acc / w_sum;
      if (hypothesisDistance(next, mode.xi) < 1e-4) {
        mode.xi = next;
        break;
      }
      mode.xi = next;
      mode.votes = vote_sum / w_sum;
    }
    modes.push_back(mode);
    for (size_t j = 0; j < peaks.size(); ++j) {
      if (hypothesisDistance(mode.xi, peaks[j].xi) < bandwidth) {
        merged[j] = true;
      }
    }
  }
  std::sort(modes.begin(), modes.end(), [](const SE2PeakCandidate & a, const SE2PeakCandidate & b) {
      return a.votes > b.votes;
    });
  return modes;
}

}  // namespace lie_lane_detection
