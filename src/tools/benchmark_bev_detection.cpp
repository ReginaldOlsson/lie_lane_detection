// Offline micro-benchmark for the BEV lane detection pipeline.
// Runs detectLanesInBev N times on a BEV image (or a synthetic 3-lane BEV) and
// reports median per-stage timings (edge / vote / fit / post / total). Use it to
// measure the effect of pipeline optimizations without a running ROS stack.
//
// Usage:
//   benchmark_bev_detection [--image PATH] [--iters N] [--warmup W]

#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <iostream>
#include <string>
#include <vector>

namespace lie = lie_lane_detection;

namespace
{
double median(std::vector<double> v)
{
  if (v.empty()) {
    return 0.0;
  }
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return (v.size() % 2 == 0) ? 0.5 * (v[m - 1] + v[m]) : v[m];
}

cv::Mat makeSyntheticBev(const lie::PipelineParams & params)
{
  const int w = static_cast<int>(params.bev_width_m / params.bev_resolution_m_per_px);
  const int h = static_cast<int>(params.bev_length_m / params.bev_resolution_m_per_px);
  cv::Mat bev(h, w, CV_8UC3, cv::Scalar(40, 40, 40));
  lie::TemplateCurve curve(params);
  curve.setBevExtents(0.0, static_cast<double>(h), 0.0, static_cast<double>(w));
  const std::array<double, 4> offsets = {w * 0.2, w * 0.4, w * 0.6, w * 0.8};
  for (double vx : offsets) {
    lie::XiVector xi = lie::XiVector::Zero();
    xi[0] = vx;
    xi[4] = 0.05;  // mild lateral drift for a non-trivial fit
    const auto poly = curve.samplePolyline(xi, 120);
    for (size_t i = 1; i < poly.size(); ++i) {
      cv::line(
        bev, cv::Point(static_cast<int>(poly[i - 1].x()), static_cast<int>(poly[i - 1].y())),
        cv::Point(static_cast<int>(poly[i].x()), static_cast<int>(poly[i].y())),
        cv::Scalar(220, 220, 220), 3);
    }
  }
  return bev;
}
}  // namespace

int main(int argc, char ** argv)
{
  std::string image_path;
  int iters = 50;
  int warmup = 5;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--iters" && i + 1 < argc) {
      iters = std::stoi(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::stoi(argv[++i]);
    }
  }

  lie::PipelineParams params;
  cv::Mat bev;
  if (image_path.empty()) {
    bev = makeSyntheticBev(params);
    std::cout << "Synthetic BEV " << bev.cols << "x" << bev.rows << "\n";
  } else {
    bev = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bev.empty()) {
      std::cerr << "Failed to read " << image_path << "\n";
      return 1;
    }
    std::cout << "Loaded " << image_path << " " << bev.cols << "x" << bev.rows << "\n";
  }

  std::vector<double> edge, vote, fit, post, total;
  edge.reserve(iters);
  vote.reserve(iters);
  fit.reserve(iters);
  post.reserve(iters);
  total.reserve(iters);

  size_t lanes = 0;
  for (int i = 0; i < warmup + iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    const auto r = lie::detectLanesInBev(bev, params);
    const auto t1 = std::chrono::steady_clock::now();
    if (i < warmup) {
      continue;
    }
    edge.push_back(r.edge_ms);
    vote.push_back(r.vote_ms);
    fit.push_back(r.fit_ms);
    post.push_back(r.post_ms);
    total.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    lanes = r.lanes.size();
  }

  std::cout << "iters=" << iters << " lanes=" << lanes << "  (median ms)\n";
  std::cout << "  edge  = " << median(edge) << "\n";
  std::cout << "  vote  = " << median(vote) << "\n";
  std::cout << "  fit   = " << median(fit) << "\n";
  std::cout << "  post  = " << median(post) << "\n";
  std::cout << "  TOTAL = " << median(total) << "\n";
  return 0;
}
