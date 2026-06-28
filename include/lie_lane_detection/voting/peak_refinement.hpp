#pragma once

#include <vector>

#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

struct SE2PeakCandidate
{
  double vx{0.0};
  double vy{0.0};
  double omega{0.0};
  double votes{0.0};
  XiVector xi{XiVector::Zero()};
};

/// Quadratic sub-bin refinement on a 3×3×3 neighborhood of accumulator samples.
SE2PeakCandidate refinePeakQuadratic(
  const std::vector<double> & accum,
  int vx_bins,
  int vy_bins,
  int omega_bins,
  int ix,
  int iy,
  int io,
  const PipelineParams & params);

/// Mean-shift clustering of peaks in SE(2) × (κ,σ) space.
std::vector<SE2PeakCandidate> meanShiftPeaks(
  const std::vector<SE2PeakCandidate> & peaks,
  double bandwidth,
  int max_iterations = 15);

}  // namespace lie_lane_detection
