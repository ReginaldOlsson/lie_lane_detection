#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/testing/test_helpers.hpp"
#include "lie_lane_detection/voting/cuda_voting.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace lldc = lie_lane_detection;

namespace
{

inline int flatBinIndex(int ix, int iy, int io, int vx_bins, int vy_bins)
{
  return ix + vy_bins * vx_bins * io + vx_bins * iy;
}

// CPU reference for Stage A accumulation, mirroring LieHoughVoter but using the
// authoritative TemplateCurve geometry in double precision. The GPU kernel must
// agree with this within float tolerance.
std::vector<double> cpuStageA(
  const lldc::PipelineParams & params, const lldc::TemplateCurve & curve,
  const std::vector<lldc::EdgePoint> & edges, const std::vector<Sophus::SE2d> & se2_inv_lut,
  const std::vector<double> & vx_lut, const std::vector<std::pair<double, double>> & preset_ks)
{
  const int vx_bins = params.se2_vx_bins;
  const int vy_bins = params.se2_vy_bins;
  const int omega_bins = params.se2_omega_bins;
  const int se2_cells = vx_bins * vy_bins * omega_bins;
  const double vote_thresh = params.vote_threshold_px;
  const double vote_thresh_sq = vote_thresh * vote_thresh;
  const double bin_width =
    (params.se2_vx_max - params.se2_vx_min) / static_cast<double>(std::max(vx_bins - 1, 1));
  const int ix_radius = std::max(1, static_cast<int>(std::ceil(vote_thresh / bin_width)) + 1);

  std::vector<double> accum(static_cast<size_t>(se2_cells), 0.0);
  for (const auto & edge : edges) {
    const lldc::Vec2 p(edge.x, edge.y);
    int ix_center = 0;
    if (vx_bins > 1) {
      const double t = (edge.x - params.se2_vx_min) / (params.se2_vx_max - params.se2_vx_min);
      ix_center = std::clamp(
        static_cast<int>(std::lround(t * static_cast<double>(vx_bins - 1))), 0, vx_bins - 1);
    }
    const int ix_lo = std::max(0, ix_center - ix_radius);
    const int ix_hi = std::min(vx_bins - 1, ix_center + ix_radius);
    for (int ix = ix_lo; ix <= ix_hi; ++ix) {
      if (std::abs(edge.x - vx_lut[static_cast<size_t>(ix)]) > vote_thresh * 1.5) {
        continue;
      }
      for (int iy = 0; iy < vy_bins; ++iy) {
        for (int io = 0; io < omega_bins; ++io) {
          const int idx = flatBinIndex(ix, iy, io, vx_bins, vy_bins);
          const lldc::Vec2 pl = se2_inv_lut[static_cast<size_t>(idx)] * p;
          double best = std::numeric_limits<double>::max();
          for (const auto & [kappa, sigma] : preset_ks) {
            const double d = curve.distanceInLocalFrame(pl.x(), pl.y(), kappa, sigma);
            best = std::min(best, d * d);
          }
          if (best < vote_thresh_sq) {
            accum[static_cast<size_t>(idx)] += edge.magnitude;
          }
        }
      }
    }
  }
  return accum;
}

}  // namespace

TEST(CudaVotingTest, StageAMatchesCpuReference)
{
  if (!lldc::cuda::isAvailable()) {
    GTEST_SKIP() << "No CUDA device available";
  }

  lldc::PipelineParams params;
  params.se2_vx_bins = 21;
  params.se2_vy_bins = 5;
  params.se2_omega_bins = 11;
  params.se2_vx_min = -20.0;
  params.se2_vx_max = 20.0;
  params.vote_threshold_px = 6.0;

  lldc::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 200.0, -100.0, 100.0);

  lldc::XiVector xi_true = lldc::XiVector::Zero();
  xi_true[0] = 8.0;
  const auto edges = lldc::test_helpers::samplePointsOnCurve(curve, xi_true, 120, 1.0);
  ASSERT_FALSE(edges.empty());

  const int vx_bins = params.se2_vx_bins;
  const int vy_bins = params.se2_vy_bins;
  const int omega_bins = params.se2_omega_bins;
  const int se2_cells = vx_bins * vy_bins * omega_bins;

  const int ik_mid = std::max(0, params.kappa_bins / 2);
  const int is_mid = std::max(0, params.sigma_bins / 2);
  const int ik_hi = std::max(0, params.kappa_bins - 1);
  const int is_hi = std::max(0, params.sigma_bins - 1);

  auto lerp = [](int i, int n, double vmin, double vmax) {
    if (n <= 1) {
      return vmin;
    }
    return vmin + (vmax - vmin) * static_cast<double>(i) / static_cast<double>(n - 1);
  };
  auto binToXi = [&](int ix, int iy, int io, int ik, int is) {
    lldc::XiVector xi;
    xi[0] = lerp(ix, vx_bins, params.se2_vx_min, params.se2_vx_max);
    xi[1] = lerp(iy, vy_bins, params.se2_vy_min, params.se2_vy_max);
    xi[2] = lerp(io, omega_bins, params.se2_omega_min, params.se2_omega_max);
    xi[3] = lerp(ik, params.kappa_bins, params.kappa_min, params.kappa_max);
    xi[4] = lerp(is, params.sigma_bins, params.sigma_min, params.sigma_max);
    return xi;
  };

  std::vector<Sophus::SE2d> se2_inv_lut(static_cast<size_t>(se2_cells));
  std::vector<double> vx_lut(static_cast<size_t>(vx_bins));
  for (int ix = 0; ix < vx_bins; ++ix) {
    vx_lut[static_cast<size_t>(ix)] = lerp(ix, vx_bins, params.se2_vx_min, params.se2_vx_max);
    for (int iy = 0; iy < vy_bins; ++iy) {
      for (int io = 0; io < omega_bins; ++io) {
        const int idx = flatBinIndex(ix, iy, io, vx_bins, vy_bins);
        se2_inv_lut[static_cast<size_t>(idx)] =
          lldc::xiToSE2(binToXi(ix, iy, io, ik_mid, is_mid)).inverse();
      }
    }
  }

  const std::vector<std::pair<int, int>> deform_presets = {
    {ik_mid, is_mid}, {ik_hi, is_mid}, {0, is_mid}, {ik_mid, is_hi}, {ik_mid, 0}};
  std::vector<std::pair<double, double>> preset_ks;
  for (const auto & [ik, is] : deform_presets) {
    const lldc::XiVector xi = binToXi(0, 0, 0, ik, is);
    preset_ks.emplace_back(xi[3], xi[4]);
  }

  const std::vector<double> cpu_accum =
    cpuStageA(params, curve, edges, se2_inv_lut, vx_lut, preset_ks);

  // Build the flat GPU buffers exactly like LieHoughVoter does.
  std::vector<float> lut_flat(static_cast<size_t>(se2_cells) * 6);
  for (int i = 0; i < se2_cells; ++i) {
    const auto & g = se2_inv_lut[static_cast<size_t>(i)];
    const Eigen::Matrix2d R = g.rotationMatrix();
    const lldc::Vec2 t = g.translation();
    float * dst = lut_flat.data() + static_cast<size_t>(i) * 6;
    dst[0] = static_cast<float>(R(0, 0));
    dst[1] = static_cast<float>(R(0, 1));
    dst[2] = static_cast<float>(R(1, 0));
    dst[3] = static_cast<float>(R(1, 1));
    dst[4] = static_cast<float>(t.x());
    dst[5] = static_cast<float>(t.y());
  }
  std::vector<float> ex(edges.size()), ey(edges.size()), emag(edges.size());
  for (size_t i = 0; i < edges.size(); ++i) {
    ex[i] = static_cast<float>(edges[i].x);
    ey[i] = static_cast<float>(edges[i].y);
    emag[i] = static_cast<float>(edges[i].magnitude);
  }
  std::vector<float> vx_lut_f(vx_lut.begin(), vx_lut.end());
  std::vector<float> pk, ps;
  for (const auto & [k, s] : preset_ks) {
    pk.push_back(static_cast<float>(k));
    ps.push_back(static_cast<float>(s));
  }

  const double bin_width =
    (params.se2_vx_max - params.se2_vx_min) / static_cast<double>(std::max(vx_bins - 1, 1));
  lldc::cuda::StageAConfig cfg;
  cfg.vx_bins = vx_bins;
  cfg.vy_bins = vy_bins;
  cfg.omega_bins = omega_bins;
  cfg.num_presets = static_cast<int>(preset_ks.size());
  cfg.ix_radius =
    std::max(1, static_cast<int>(std::ceil(params.vote_threshold_px / bin_width)) + 1);
  cfg.vx_min = static_cast<float>(params.se2_vx_min);
  cfg.vx_max = static_cast<float>(params.se2_vx_max);
  cfg.y_min = static_cast<float>(curve.localFrameYMin());
  cfg.y_span = static_cast<float>(curve.localFrameYSpan());
  cfg.vote_thresh = static_cast<float>(params.vote_threshold_px);

  std::vector<double> gpu_accum;
  ASSERT_TRUE(lldc::cuda::stageAVote(cfg, ex, ey, emag, lut_flat, vx_lut_f, pk, ps, gpu_accum));
  ASSERT_EQ(gpu_accum.size(), cpu_accum.size());

  double cpu_sum = 0.0, gpu_sum = 0.0, max_abs_diff = 0.0;
  int cpu_argmax = 0, gpu_argmax = 0;
  for (int i = 0; i < se2_cells; ++i) {
    cpu_sum += cpu_accum[static_cast<size_t>(i)];
    gpu_sum += gpu_accum[static_cast<size_t>(i)];
    max_abs_diff = std::max(
      max_abs_diff,
      std::abs(cpu_accum[static_cast<size_t>(i)] - gpu_accum[static_cast<size_t>(i)]));
    if (cpu_accum[static_cast<size_t>(i)] > cpu_accum[static_cast<size_t>(cpu_argmax)]) {
      cpu_argmax = i;
    }
    if (gpu_accum[static_cast<size_t>(i)] > gpu_accum[static_cast<size_t>(gpu_argmax)]) {
      gpu_argmax = i;
    }
  }

  EXPECT_GT(cpu_sum, 0.0);
  // Peak bin must match, and total votes agree within 1% (float threshold flips
  // at a few boundary edges are acceptable).
  EXPECT_EQ(cpu_argmax, gpu_argmax);
  EXPECT_NEAR(gpu_sum, cpu_sum, cpu_sum * 0.01);
  EXPECT_LT(max_abs_diff, 3.0 * edges.front().magnitude + 1e-3);
}
