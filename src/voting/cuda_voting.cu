#include "lie_lane_detection/voting/cuda_voting.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

namespace lie_lane_detection
{
namespace cuda
{

namespace
{

// Mirror of LieHoughVoter::flatBinIndex (host) so bin indices agree exactly.
__host__ __device__ __forceinline__ int flatBinIndex(
  int ix, int iy, int io, int vx_bins, int vy_bins)
{
  return ix + vy_bins * vx_bins * io + vx_bins * iy;
}

// Device port of TemplateCurve::distanceInLocalFrame, returning the *squared*
// distance (the caller only compares against a squared threshold, so the sqrt
// is unnecessary). Point (a,b) is already in the SE(2)-local frame.
__device__ __forceinline__ float distSqLocal(
  float a, float b, float kappa, float sigma, float y_min, float y_span)
{
  const float A = kappa * y_span;
  const float B = sigma * y_span;

  float t = (b - y_min) / y_span;
  t = fminf(fmaxf(t, 0.0f), 1.0f);

#pragma unroll
  for (int iter = 0; iter < 8; ++iter) {
    const float xc = (A * t + B) * t;
    const float yc = y_min + y_span * t;
    const float xc_p = 2.0f * A * t + B;
    const float yc_p = y_span;
    const float grad = (xc - a) * xc_p + (yc - b) * yc_p;
    const float hess = xc_p * xc_p + (xc - a) * (2.0f * A) + yc_p * yc_p;
    if (fabsf(hess) < 1e-12f) {
      break;
    }
    float t_new = t - grad / hess;
    t_new = fminf(fmaxf(t_new, 0.0f), 1.0f);
    const float dt = t_new - t;
    t = t_new;
    if (fabsf(dt) < 1e-6f) {
      break;
    }
  }

  const float xc = (A * t + B) * t;
  const float yc = y_min + y_span * t;
  const float dx = a - xc;
  const float dy = b - yc;
  return dx * dx + dy * dy;
}

// One thread per edge; each edge accumulates into the localized SE(2) window.
__global__ void stageAKernel(
  const float * __restrict__ edge_x,
  const float * __restrict__ edge_y,
  const float * __restrict__ edge_mag,
  int num_edges,
  const float * __restrict__ se2_inv_lut,
  const float * __restrict__ vx_lut,
  const float * __restrict__ preset_kappa,
  const float * __restrict__ preset_sigma,
  int num_presets,
  int vx_bins,
  int vy_bins,
  int omega_bins,
  int ix_radius,
  float vx_min,
  float vx_max,
  float y_min,
  float y_span,
  float vote_thresh,
  float * __restrict__ accum)
{
  const int e = blockIdx.x * blockDim.x + threadIdx.x;
  if (e >= num_edges) {
    return;
  }

  const float px = edge_x[e];
  const float py = edge_y[e];
  const float mag = edge_mag[e];
  const float vote_thresh_sq = vote_thresh * vote_thresh;
  const float dx_gate = vote_thresh * 1.5f;

  // vxBinForX
  int ix_center = 0;
  if (vx_bins > 1) {
    const float tt = (px - vx_min) / (vx_max - vx_min);
    ix_center = static_cast<int>(lroundf(tt * static_cast<float>(vx_bins - 1)));
    ix_center = min(max(ix_center, 0), vx_bins - 1);
  }
  const int ix_lo = max(0, ix_center - ix_radius);
  const int ix_hi = min(vx_bins - 1, ix_center + ix_radius);

  for (int ix = ix_lo; ix <= ix_hi; ++ix) {
    const float dx = px - vx_lut[ix];
    if (fabsf(dx) > dx_gate) {
      continue;
    }
    for (int iy = 0; iy < vy_bins; ++iy) {
      for (int io = 0; io < omega_bins; ++io) {
        const int idx = flatBinIndex(ix, iy, io, vx_bins, vy_bins);
        const float * g = se2_inv_lut + 6 * idx;
        const float a = g[0] * px + g[1] * py + g[4];
        const float b = g[2] * px + g[3] * py + g[5];

        float best = 3.4e38f;
        for (int k = 0; k < num_presets; ++k) {
          const float d2 = distSqLocal(a, b, preset_kappa[k], preset_sigma[k], y_min, y_span);
          best = fminf(best, d2);
        }
        if (best < vote_thresh_sq) {
          atomicAdd(&accum[idx], mag);
        }
      }
    }
  }
}

// Persistent, grow-only device buffers reused across vote() calls. The BEV
// peeling loop invokes vote() several times per frame, so allocating once and
// reusing avoids repeated cudaMalloc/cudaFree (each ~0.1-0.3 ms) that would
// otherwise dwarf the kernel itself. Access is serialized by a mutex; vote() is
// never called concurrently, but this keeps the path safe if that changes.
struct GpuContext
{
  std::mutex mu;
  float * d_x{nullptr};
  float * d_y{nullptr};
  float * d_mag{nullptr};
  float * d_lut{nullptr};
  float * d_vx{nullptr};
  float * d_pk{nullptr};
  float * d_ps{nullptr};
  float * d_acc{nullptr};
  size_t cap_edges{0};
  size_t cap_lut{0};
  size_t cap_vx{0};
  size_t cap_preset{0};
  size_t cap_acc{0};
};

GpuContext & gpuContext()
{
  static GpuContext ctx;
  return ctx;
}

// Ensure *p can hold at least `need` floats, reallocating (grow-only) if not.
bool ensureCapacity(float ** p, size_t * cap, size_t need)
{
  if (*p != nullptr && *cap >= need) {
    return true;
  }
  if (*p != nullptr) {
    cudaFree(*p);
    *p = nullptr;
  }
  *cap = 0;
  if (need == 0) {
    return true;
  }
  if (cudaMalloc(reinterpret_cast<void **>(p), need * sizeof(float)) != cudaSuccess) {
    *p = nullptr;
    return false;
  }
  *cap = need;
  return true;
}

bool timingEnabled()
{
  static const bool on = [] {
      const char * v = std::getenv("LIE_CUDA_TIMING");
      return v && v[0] != '\0' && v[0] != '0';
    }();
  return on;
}

bool checkAvailable()
{
  int count = 0;
  const cudaError_t err = cudaGetDeviceCount(&count);
  return err == cudaSuccess && count > 0;
}

}  // namespace

bool isAvailable()
{
  static std::once_flag once;
  static bool available = false;
  std::call_once(once, [] {available = checkAvailable();});
  return available;
}

bool stageAVote(
  const StageAConfig & cfg,
  const std::vector<float> & edge_x,
  const std::vector<float> & edge_y,
  const std::vector<float> & edge_mag,
  const std::vector<float> & se2_inv_lut,
  const std::vector<float> & vx_lut,
  const std::vector<float> & preset_kappa,
  const std::vector<float> & preset_sigma,
  std::vector<double> & accum_out)
{
  if (!isAvailable()) {
    return false;
  }

  const int se2_cells = cfg.vx_bins * cfg.vy_bins * cfg.omega_bins;
  if (se2_cells <= 0) {
    return false;
  }
  accum_out.assign(static_cast<size_t>(se2_cells), 0.0);

  const int num_edges = static_cast<int>(edge_x.size());
  if (num_edges == 0) {
    return true;
  }

  auto ok = [](cudaError_t e) {return e == cudaSuccess;};

  GpuContext & ctx = gpuContext();
  std::lock_guard<std::mutex> lock(ctx.mu);

  if (!ensureCapacity(&ctx.d_x, &ctx.cap_edges, edge_x.size())) {return false;}
  // d_y, d_mag share the edge capacity but need their own allocations.
  if (!ensureCapacity(&ctx.d_y, &ctx.cap_edges, edge_x.size())) {return false;}
  if (!ensureCapacity(&ctx.d_mag, &ctx.cap_edges, edge_x.size())) {return false;}
  if (!ensureCapacity(&ctx.d_lut, &ctx.cap_lut, se2_inv_lut.size())) {return false;}
  if (!ensureCapacity(&ctx.d_vx, &ctx.cap_vx, vx_lut.size())) {return false;}
  if (!ensureCapacity(&ctx.d_pk, &ctx.cap_preset, preset_kappa.size())) {return false;}
  if (!ensureCapacity(&ctx.d_ps, &ctx.cap_preset, preset_sigma.size())) {return false;}
  if (!ensureCapacity(&ctx.d_acc, &ctx.cap_acc, static_cast<size_t>(se2_cells))) {return false;}

  const bool timing = timingEnabled();
  cudaEvent_t t0, t1;
  if (timing) {
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
  }

  auto up = [&](float * dst, const std::vector<float> & src) {
      return cudaMemcpy(dst, src.data(), src.size() * sizeof(float), cudaMemcpyHostToDevice);
    };
  if (!ok(up(ctx.d_x, edge_x)) || !ok(up(ctx.d_y, edge_y)) || !ok(up(ctx.d_mag, edge_mag)) ||
    !ok(up(ctx.d_lut, se2_inv_lut)) || !ok(up(ctx.d_vx, vx_lut)) ||
    !ok(up(ctx.d_pk, preset_kappa)) || !ok(up(ctx.d_ps, preset_sigma)))
  {
    return false;
  }
  if (!ok(cudaMemset(ctx.d_acc, 0, static_cast<size_t>(se2_cells) * sizeof(float)))) {
    return false;
  }

  const int block = 128;
  const int grid = (num_edges + block - 1) / block;
  if (timing) {cudaEventRecord(t0);}
  stageAKernel<<<grid, block>>>(
    ctx.d_x, ctx.d_y, ctx.d_mag, num_edges, ctx.d_lut, ctx.d_vx, ctx.d_pk, ctx.d_ps,
    cfg.num_presets, cfg.vx_bins, cfg.vy_bins, cfg.omega_bins, cfg.ix_radius,
    cfg.vx_min, cfg.vx_max, cfg.y_min, cfg.y_span, cfg.vote_thresh, ctx.d_acc);
  if (timing) {cudaEventRecord(t1);}

  if (!ok(cudaGetLastError()) || !ok(cudaDeviceSynchronize())) {
    return false;
  }

  std::vector<float> host_acc(static_cast<size_t>(se2_cells));
  if (!ok(cudaMemcpy(
      host_acc.data(), ctx.d_acc, host_acc.size() * sizeof(float), cudaMemcpyDeviceToHost)))
  {
    return false;
  }
  for (int i = 0; i < se2_cells; ++i) {
    accum_out[static_cast<size_t>(i)] = static_cast<double>(host_acc[static_cast<size_t>(i)]);
  }

  if (timing) {
    float kernel_ms = 0.0f;
    cudaEventElapsedTime(&kernel_ms, t0, t1);
    std::fprintf(
      stderr, "[cuda_voting] edges=%d cells=%d kernel=%.3f ms\n", num_edges, se2_cells, kernel_ms);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
  }
  return true;
}

}  // namespace cuda
}  // namespace lie_lane_detection
