#pragma once

#include <vector>

// Custom CUDA kernels for the Lie-Hough voting hot path. This header is
// intentionally free of Eigen / Sophus / OpenCV so it can be compiled by nvcc
// as part of the .cu translation unit. All geometry (SE(2) inverse poses) is
// precomputed on the host and passed in as flat float buffers.
namespace lie_lane_detection
{
namespace cuda
{

/// True only when built with LIE_HAS_CUDA *and* a usable CUDA device is present.
/// Cheap after the first call (result is cached).
bool isAvailable();

/// Scalar configuration mirroring the CPU Stage A loop.
struct StageAConfig
{
  int vx_bins{0};
  int vy_bins{0};
  int omega_bins{0};
  int num_presets{0};
  int ix_radius{0};
  float vx_min{0.0f};
  float vx_max{0.0f};
  float y_min{0.0f};
  float y_span{1.0f};
  float vote_thresh{0.0f};
  // Soft (Gaussian) voting: weight each in-gate contribution by
  // exp(-d^2 * inv_two_sigma_sq). When soft_voting == 0 the vote is the raw
  // magnitude (hard threshold).
  int soft_voting{0};
  float inv_two_sigma_sq{0.0f};
};

/// GPU implementation of Lie-Hough Stage A localized SE(2) accumulation.
///
/// Buffer layouts (all row-major, indexed by the same flatBinIndex the CPU uses):
///   - se2_inv_lut: 6 floats per SE(2) cell = [R00, R01, R10, R11, tx, ty],
///     i.e. the inverse pose acting on a point as (a,b) = R*p + t.
///   - vx_lut: one float per vx bin (the lateral bin center).
///   - preset_kappa / preset_sigma: num_presets deform presets.
///   - accum_out: resized to vx_bins*vy_bins*omega_bins and filled with votes.
///
/// Returns false if the GPU path is unavailable or a CUDA call failed, in which
/// case the caller must fall back to the CPU implementation.
bool stageAVote(
  const StageAConfig & cfg, const std::vector<float> & edge_x, const std::vector<float> & edge_y,
  const std::vector<float> & edge_mag, const std::vector<float> & se2_inv_lut,
  const std::vector<float> & vx_lut, const std::vector<float> & preset_kappa,
  const std::vector<float> & preset_sigma, std::vector<double> & accum_out);

}  // namespace cuda
}  // namespace lie_lane_detection
