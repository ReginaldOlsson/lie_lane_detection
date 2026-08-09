// Generate synthetic BEV test images and optionally run lane detection on each.
//
// Usage:
//   generate_test_images --output /path/to/dir [--detect]

#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/testing/synthetic_bev_generator.hpp"

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static lie_lane_detection::PipelineParams paramsForSynthetic(const cv::Mat & bev)
{
  lie_lane_detection::PipelineParams p;
  p.use_steerable_filter = true;
  p.connect_dashed_edges = true;
  p.top_k_peaks = 10;
  p.max_lane_hypotheses = 10;
  p.use_iterative_peeling = true;
  p.vote_threshold_px = 6.0;
  p.inlier_threshold_px = 8.0;
  p.min_inlier_ratio = 0.40;
  p.min_inliers = 10;
  p.kappa_min = -0.2;
  p.kappa_max = 0.2;
  p.sigma_min = -0.15;
  p.sigma_max = 0.15;
  lie_lane_detection::configureParamsForBev(p, bev.cols, bev.rows);
  lie_lane_detection::enhanceParamsForCurvature(p);
  return p;
}

static void writeCurveEvaluation(
  const fs::path & out_dir, const std::vector<lie_lane_detection::XiVector> & gt,
  const std::vector<lie_lane_detection::LaneHypothesis> & detected, double elapsed_ms)
{
  std::ofstream eval(out_dir / "curve_eval.txt");
  eval << "Curve / pose evaluation (greedy vx match)\n";
  eval << "=========================================\n";
  eval << "GT boundaries: " << gt.size() << "\n";
  eval << "Detected: " << detected.size() << "\n";
  eval << "Elapsed ms: " << elapsed_ms << "\n\n";

  std::vector<bool> gt_used(gt.size(), false);
  double sum_vx = 0.0;
  double sum_omega = 0.0;
  double sum_kappa = 0.0;
  double sum_sigma = 0.0;
  int matched = 0;

  constexpr double kVxGatePx = 35.0;
  for (const auto & lane : detected) {
    size_t best_j = 0;
    double best_dx = std::numeric_limits<double>::max();
    for (size_t j = 0; j < gt.size(); ++j) {
      if (gt_used[j]) {
        continue;
      }
      const double dx = std::abs(lane.xi[0] - gt[j][0]);
      if (dx < best_dx) {
        best_dx = dx;
        best_j = j;
      }
    }
    if (best_dx > kVxGatePx) {
      eval << "unmatched det vx=" << lane.xi[0] << " (no GT within " << kVxGatePx << " px)\n";
      continue;
    }

    gt_used[best_j] = true;
    ++matched;
    const auto & g = gt[best_j];
    const double err_vx = lane.xi[0] - g[0];
    const double err_w = lane.xi[2] - g[2];
    const double err_k = lane.xi[3] - g[3];
    const double err_s = lane.xi[4] - g[4];
    sum_vx += std::abs(err_vx);
    sum_omega += std::abs(err_w);
    sum_kappa += std::abs(err_k);
    sum_sigma += std::abs(err_s);

    eval << "match gt[" << best_j << "] vs lane " << lane.lane_id << "\n";
    eval << "  dvx=" << err_vx << " dw=" << err_w << " dk=" << err_k << " ds=" << err_s << "\n";
    eval << "  gt  (vx,w,k,s)=(" << g[0] << "," << g[2] << "," << g[3] << "," << g[4] << ")\n";
    eval << "  det (vx,w,k,s)=(" << lane.xi[0] << "," << lane.xi[2] << "," << lane.xi[3] << ","
         << lane.xi[4] << ")\n\n";
  }

  for (size_t j = 0; j < gt.size(); ++j) {
    if (!gt_used[j]) {
      eval << "missed gt[" << j << "] vx=" << gt[j][0] << " k=" << gt[j][3] << " s=" << gt[j][4]
           << "\n";
    }
  }

  if (matched > 0) {
    eval << "\nMean abs error (matched): vx=" << (sum_vx / matched)
         << " omega=" << (sum_omega / matched) << " kappa=" << (sum_kappa / matched)
         << " sigma=" << (sum_sigma / matched) << "\n";
  }
  eval << "Recall: " << matched << "/" << gt.size() << "\n";
}

static void runDetection(
  const cv::Mat & bev, const fs::path & out_dir, const lie_lane_detection::PipelineParams & params,
  const std::vector<lie_lane_detection::XiVector> & ground_truth)
{
  cv::Mat prepared = bev.clone();
  if (out_dir.filename().string().find("ipm") != std::string::npos) {
    prepared = lie_lane_detection::prepareBevImage(prepared);
  }

  const auto result = lie_lane_detection::detectLanesInBev(prepared, params);

  cv::imwrite((out_dir / "edges.png").string(), result.edges);
  cv::imwrite((out_dir / "hough_slice.png").string(), result.hough_slice);
  cv::imwrite((out_dir / "overlay.png").string(), result.overlay);

  std::ofstream report(out_dir / "detection_report.txt");
  report << "Detected lanes: " << result.lanes.size() << "\n";
  report << "Edge points: " << result.edge_point_count << "\n";
  report << "Elapsed ms: " << result.elapsed_ms << "\n";
  report << "Merge events: " << result.merges.size() << "\n";
  for (const auto & lane : result.lanes) {
    report << "lane " << lane.lane_id << " vx=" << lane.xi[0] << " w=" << lane.xi[2]
           << " k=" << lane.xi[3] << " s=" << lane.xi[4] << " inliers=" << lane.inlier_ratio
           << "\n";
  }

  writeCurveEvaluation(out_dir, ground_truth, result.lanes, result.elapsed_ms);
}

int main(int argc, char ** argv)
{
  fs::path output_root = "/home/mosal/rviz_ws/lane_detection_test_images";
  bool run_detect = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--output" && i + 1 < argc) {
      output_root = fs::path(argv[++i]);
    } else if (arg == "--detect") {
      run_detect = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: generate_test_images [--output DIR] [--detect]\n"
                   "  Generates 9 synthetic BEV test scenes with ground-truth metadata.\n"
                   "  --detect  Also run lane detection and save overlay per scene.\n";
      return 0;
    }
  }

  fs::create_directories(output_root);

  lie_lane_detection::SyntheticSceneSpec spec;
  spec.width = 240;
  spec.height = 800;

  lie_lane_detection::SyntheticBEVGenerator generator;
  const auto scenes = generator.generateAll(spec);

  std::ofstream index(output_root / "README.txt");
  index << "Lie Lane Detection — Test Image Suite\n";
  index << "======================================\n\n";

  index << "real_dataset/\n";
  index << "  Four famous benchmark frames (TuSimple x3, CULane x1) for perspective-camera\n";
  index << "  testing. See real_dataset/README.txt for sources and IPM usage notes.\n\n";
  index << "Synthetic scenes (BEV/IPM-ready)\n";
  index << "--------------------------------\n\n";

  for (const auto & scene : scenes) {
    const fs::path scene_dir = output_root / scene.name;
    fs::create_directories(scene_dir);

    const std::string input_name = scene.name + ".png";
    cv::imwrite((scene_dir / input_name).string(), scene.image);
    cv::imwrite((output_root / (scene.name + ".png")).string(), scene.image);

    std::ofstream gt(scene_dir / "ground_truth.txt");
    gt << scene.description << "\n";
    gt << "Expected boundaries: " << scene.ground_truth_xi.size() << "\n";
    for (size_t i = 0; i < scene.ground_truth_xi.size(); ++i) {
      const auto & xi = scene.ground_truth_xi[i];
      gt << "  gt[" << i << "] vx=" << xi[0] << " vy=" << xi[1] << " w=" << xi[2] << " k=" << xi[3]
         << " s=" << xi[4] << "\n";
    }

    index << scene.name << "/\n";
    index << "  " << scene.description << "\n";
    index << "  GT boundaries: " << scene.ground_truth_xi.size() << "\n";
    index << "  Image: " << scene.name << ".png\n\n";

    std::cout << "Created: " << scene_dir / input_name << "\n";

    if (run_detect) {
      const auto params = paramsForSynthetic(scene.image);
      runDetection(scene.image, scene_dir, params, scene.ground_truth_xi);
      std::cout << "  Detection output -> " << scene_dir << "\n";
    }
  }

  index << "Run detection on one image:\n";
  index << "  lane_detect_offline --image <path> --output <dir>\n";
  index.close();

  std::cout << "\nAll scenes written to: " << output_root << "\n";
  std::cout << "See README.txt for the full list.\n";
  return 0;
}
