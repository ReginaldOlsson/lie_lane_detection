// Compare edge-based vs line-first Lie-Hough pipelines on BEV images.
//
// Usage:
//   compare_lane_detection --scenes DIR [--output DIR]
//   compare_lane_detection --image PATH [--output DIR] [--perspective]
//   compare_lane_detection --real-dataset DIR [--output DIR]

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>

#include "lie_lane_detection/core/types.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/pipeline/line_lane_detection_runner.hpp"

namespace fs = std::filesystem;

struct EvalMetrics
{
  int gt_count{0};
  int matched{0};
  double mean_omega_err{0.0};
  double mean_kappa_err{0.0};
  double mean_vx_err{0.0};
};

static lie_lane_detection::PipelineParams defaultParams()
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
  p.line_hough_threshold = 22;
  p.line_min_length_px = 16.0;
  p.line_max_gap_px = 14.0;
  p.line_angle_threshold_rad = 0.5;
  return p;
}

static std::vector<lie_lane_detection::XiVector> loadGroundTruth(const fs::path & gt_file)
{
  std::vector<lie_lane_detection::XiVector> gt;
  if (!fs::exists(gt_file)) {
    return gt;
  }

  std::ifstream in(gt_file);
  std::string line;
  const std::regex row(
    R"(gt\[\d+\]\s+vx=([-\d.]+)\s+vy=([-\d.]+)\s+w=([-\d.]+)\s+k=([-\d.]+)\s+s=([-\d.]+))");

  while (std::getline(in, line)) {
    std::smatch m;
    if (!std::regex_search(line, m, row)) {
      continue;
    }
    lie_lane_detection::XiVector xi = lie_lane_detection::XiVector::Zero();
    xi[0] = std::stod(m[1].str());
    xi[1] = std::stod(m[2].str());
    xi[2] = std::stod(m[3].str());
    xi[3] = std::stod(m[4].str());
    xi[4] = std::stod(m[5].str());
    gt.push_back(xi);
  }
  return gt;
}

static EvalMetrics evaluateAgainstGt(
  const std::vector<lie_lane_detection::XiVector> & gt,
  const std::vector<lie_lane_detection::LaneHypothesis> & lanes,
  std::ostream * detail = nullptr)
{
  EvalMetrics m;
  m.gt_count = static_cast<int>(gt.size());
  if (gt.empty()) {
    return m;
  }

  std::vector<bool> gt_used(gt.size(), false);
  double sum_vx = 0.0;
  double sum_omega = 0.0;
  double sum_kappa = 0.0;

  constexpr double kVxGatePx = 35.0;
  for (const auto & lane : lanes) {
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
      if (detail) {
        *detail << "  unmatched det vx=" << lane.xi[0] << "\n";
      }
      continue;
    }

    gt_used[best_j] = true;
    ++m.matched;
    const auto & g = gt[best_j];
    sum_vx += std::abs(lane.xi[0] - g[0]);
    sum_omega += std::abs(lane.xi[2] - g[2]);
    sum_kappa += std::abs(lane.xi[3] - g[3]);

    if (detail) {
      *detail << "  match gt[" << best_j << "] dvx=" << (lane.xi[0] - g[0])
              << " dw=" << (lane.xi[2] - g[2])
              << " dk=" << (lane.xi[3] - g[3]) << "\n";
    }
  }

  if (m.matched > 0) {
    m.mean_vx_err = sum_vx / m.matched;
    m.mean_omega_err = sum_omega / m.matched;
    m.mean_kappa_err = sum_kappa / m.matched;
  }
  return m;
}

static std::string laneSummary(const std::vector<lie_lane_detection::LaneHypothesis> & lanes)
{
  std::ostringstream oss;
  for (const auto & lane : lanes) {
    oss << "  lane" << lane.lane_id
        << " vx=" << std::fixed << std::setprecision(1) << lane.xi[0]
        << " w=" << std::setprecision(3) << lane.xi[2]
        << " k=" << lane.xi[3]
        << " inl=" << std::setprecision(2) << lane.inlier_ratio << "\n";
  }
  return oss.str();
}

static void compareOnImage(
  const cv::Mat & bev,
  const fs::path & out_dir,
  const std::string & label,
  lie_lane_detection::PipelineParams params,
  const std::vector<lie_lane_detection::XiVector> & gt,
  lie_lane_detection::BevDetectionResult * edge_out = nullptr,
  lie_lane_detection::BevDetectionResult * line_out = nullptr,
  EvalMetrics * edge_eval = nullptr,
  EvalMetrics * line_eval = nullptr)
{
  fs::create_directories(out_dir);

  lie_lane_detection::configureParamsForBev(params, bev.cols, bev.rows);
  lie_lane_detection::enhanceParamsForCurvature(params);

  const auto edge_result = lie_lane_detection::detectLanesInBev(bev, params);
  const auto line_result = lie_lane_detection::detectLanesInBevFromLines(bev, params);
  if (edge_out) {
    *edge_out = edge_result;
  }
  if (line_out) {
    *line_out = line_result;
  }

  cv::imwrite((out_dir / "edge_overlay.png").string(), edge_result.overlay);
  cv::imwrite((out_dir / "line_overlay.png").string(), line_result.overlay);
  cv::imwrite((out_dir / "edge_hough.png").string(), edge_result.hough_slice);
  cv::imwrite((out_dir / "line_hough.png").string(), line_result.hough_slice);
  cv::imwrite((out_dir / "line_segments.png").string(), line_result.edges);

  std::ofstream report(out_dir / "comparison.txt");
  report << "Pipeline comparison: " << label << "\n";
  report << "Image size: " << bev.cols << "x" << bev.rows << "\n\n";

  report << "=== Edge-pixel Lie-Hough ===\n";
  report << "Lanes: " << edge_result.lanes.size() << "\n";
  report << "Edge points: " << edge_result.edge_point_count << "\n";
  report << "Elapsed ms: " << edge_result.elapsed_ms << "\n";
  report << laneSummary(edge_result.lanes);
  if (!gt.empty()) {
    report << "GT evaluation:\n";
    const EvalMetrics em = evaluateAgainstGt(gt, edge_result.lanes, &report);
    if (edge_eval) {
      *edge_eval = em;
    }
    report << "  recall: " << em.matched << "/" << em.gt_count
           << "  mean|omega|=" << em.mean_omega_err
           << "  mean|kappa|=" << em.mean_kappa_err << "\n";
  }
  report << "\n";

  report << "=== Line-first Lie-Hough ===\n";
  report << "Lanes: " << line_result.lanes.size() << "\n";
  report << "Line segments: " << line_result.line_segment_count << "\n";
  report << "Edge points (for RANSAC): " << line_result.edge_point_count << "\n";
  report << "Elapsed ms: " << line_result.elapsed_ms << "\n";
  report << "  line Hough ms: " << line_result.line_hough_ms << "\n";
  report << "  Lie vote ms: " << line_result.lie_vote_ms << "\n";
  report << laneSummary(line_result.lanes);
  if (!gt.empty()) {
    report << "GT evaluation:\n";
    const EvalMetrics lm = evaluateAgainstGt(gt, line_result.lanes, &report);
    if (line_eval) {
      *line_eval = lm;
    }
    report << "  recall: " << lm.matched << "/" << lm.gt_count
           << "  mean|omega|=" << lm.mean_omega_err
           << "  mean|kappa|=" << lm.mean_kappa_err << "\n";
  }
  report << "\n";

  const double speedup = edge_result.elapsed_ms / std::max(line_result.elapsed_ms, 1e-6);
  report << "=== Summary ===\n";
  report << "Latency ratio (edge/line): " << speedup << "x\n";
  report << "Lane count delta (line - edge): "
         << static_cast<int>(line_result.lanes.size()) -
            static_cast<int>(edge_result.lanes.size()) << "\n";

  std::cout << label << ": edge " << edge_result.lanes.size() << " lanes / "
            << std::fixed << std::setprecision(0) << edge_result.elapsed_ms << " ms | line "
            << line_result.lanes.size() << " lanes / "
            << line_result.elapsed_ms << " ms ("
            << line_result.line_segment_count << " segments)";
  if (!gt.empty() && edge_eval && line_eval) {
    std::cout << " | recall E:" << edge_eval->matched << "/" << edge_eval->gt_count
              << " L:" << line_eval->matched << "/" << line_eval->gt_count
              << " |w| E:" << std::setprecision(3) << edge_eval->mean_omega_err
              << " L:" << line_eval->mean_omega_err;
  }
  std::cout << "\n";
}

static void findSceneImages(const fs::path & root, std::vector<fs::path> & scenes)
{
  if (!fs::exists(root)) {
    return;
  }
  for (const auto & entry : fs::directory_iterator(root)) {
    if (!entry.is_directory()) {
      continue;
    }
    if (entry.path().filename() == "comparison" ||
      entry.path().filename() == "real_dataset")
    {
      continue;
    }
    for (const auto & file : fs::directory_iterator(entry.path())) {
      if (file.path().extension() == ".png" &&
        file.path().stem().string() == entry.path().filename().string())
      {
        scenes.push_back(file.path());
        break;
      }
    }
  }
  std::sort(scenes.begin(), scenes.end());
}

static void findJpegImages(const fs::path & root, std::vector<fs::path> & images)
{
  if (!fs::exists(root)) {
    return;
  }
  for (const auto & entry : fs::directory_iterator(root)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto ext = entry.path().extension().string();
    if (ext == ".jpg" || ext == ".jpeg" || ext == ".JPG") {
      images.push_back(entry.path());
    }
  }
  std::sort(images.begin(), images.end());
}

static cv::Mat loadBevImage(
  const cv::Mat & raw,
  bool perspective,
  lie_lane_detection::PipelineParams & params,
  const fs::path & out_dir,
  const std::string & label)
{
  if (!perspective) {
    return lie_lane_detection::prepareBevImage(raw);
  }

  lie_lane_detection::setDefaultHighwayIpmRoi(params, raw.cols, raw.rows);
  lie_lane_detection::configureParamsForPerspectiveIpm(params);
  cv::Mat warped = lie_lane_detection::warpPerspectiveToBev(raw, params);
  if (warped.empty()) {
    return cv::Mat();
  }
  fs::create_directories(out_dir);
  cv::imwrite((out_dir / (label + "_ipm_bev.png")).string(), warped);
  return lie_lane_detection::prepareBevImage(warped);
}

int main(int argc, char ** argv)
{
  std::string image_path;
  fs::path scenes_dir;
  fs::path real_dataset_dir;
  fs::path output_dir = "/tmp/lie_lane_compare";
  bool perspective = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--scenes" && i + 1 < argc) {
      scenes_dir = fs::path(argv[++i]);
    } else if (arg == "--real-dataset" && i + 1 < argc) {
      real_dataset_dir = fs::path(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = fs::path(argv[++i]);
    } else if (arg == "--perspective") {
      perspective = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout <<
        "Usage:\n"
        "  compare_lane_detection --scenes DIR [--output DIR]\n"
        "  compare_lane_detection --image PATH [--output DIR] [--perspective]\n"
        "  compare_lane_detection --real-dataset DIR [--output DIR]\n";
      return 0;
    }
  }

  fs::create_directories(output_dir);
  auto params = defaultParams();

  if (!image_path.empty()) {
    cv::Mat raw = cv::imread(image_path, cv::IMREAD_COLOR);
    if (raw.empty()) {
      std::cerr << "Failed to read: " << image_path << "\n";
      return 1;
    }
    const std::string label = fs::path(image_path).stem().string();
    cv::Mat bev = loadBevImage(raw, perspective, params, output_dir / "single", label);
    if (bev.empty()) {
      std::cerr << "Failed to prepare BEV image\n";
      return 1;
    }
    compareOnImage(bev, output_dir / "single", label, params, {});
    std::cout << "Wrote comparison to " << output_dir / "single" << "\n";
    return 0;
  }

  if (!real_dataset_dir.empty()) {
    std::vector<fs::path> images;
    findJpegImages(real_dataset_dir, images);
    if (images.empty()) {
      std::cerr << "No JPEG images in " << real_dataset_dir << "\n";
      return 1;
    }

    std::ofstream summary(output_dir / "real_summary.txt");
    summary << "Real dataset comparison (perspective IPM, no GT)\n\n";
    summary << std::left
            << std::setw(30) << "Image"
            << std::setw(8) << "E_lanes"
            << std::setw(8) << "L_lanes"
            << std::setw(10) << "E_ms"
            << std::setw(10) << "L_ms"
            << std::setw(8) << "Lines"
            << "\n";

    double total_edge_ms = 0.0;
    double total_line_ms = 0.0;
    for (const auto & img_path : images) {
      cv::Mat raw = cv::imread(img_path.string(), cv::IMREAD_COLOR);
      if (raw.empty()) {
        continue;
      }
      auto img_params = params;
      const std::string label = img_path.stem().string();
      const fs::path scene_out = output_dir / label;
      cv::Mat bev = loadBevImage(raw, true, img_params, scene_out, label);
      if (bev.empty()) {
        continue;
      }
      lie_lane_detection::BevDetectionResult edge_result;
      lie_lane_detection::BevDetectionResult line_result;
      compareOnImage(bev, scene_out, label, img_params, {}, &edge_result, &line_result);

      summary << std::left
              << std::setw(30) << label
              << std::setw(8) << edge_result.lanes.size()
              << std::setw(8) << line_result.lanes.size()
              << std::setw(10) << std::fixed << std::setprecision(1) << edge_result.elapsed_ms
              << std::setw(10) << line_result.elapsed_ms
              << std::setw(8) << line_result.line_segment_count
              << "\n";
      total_edge_ms += edge_result.elapsed_ms;
      total_line_ms += line_result.elapsed_ms;
    }

    summary << "\nMean edge ms: " << (total_edge_ms / images.size()) << "\n";
    summary << "Mean line ms: " << (total_line_ms / images.size()) << "\n";
    summary.close();
    std::cout << "\nReal summary: " << output_dir / "real_summary.txt" << "\n";
    return 0;
  }

  if (scenes_dir.empty()) {
    scenes_dir = "/home/mosal/rviz_ws/lane_detection_test_images";
  }

  std::vector<fs::path> scenes;
  findSceneImages(scenes_dir, scenes);
  if (scenes.empty()) {
    std::cerr << "No scene PNGs found under " << scenes_dir << "\n";
    return 1;
  }

  std::ofstream summary(output_dir / "summary.txt");
  summary << "Edge vs Line-first pipeline comparison\n";
  summary << "Scenes: " << scenes.size() << "\n\n";
  summary << std::left
          << std::setw(28) << "Scene"
          << std::setw(8) << "E_lanes"
          << std::setw(8) << "L_lanes"
          << std::setw(10) << "E_ms"
          << std::setw(10) << "L_ms"
          << std::setw(8) << "Lines"
          << std::setw(10) << "E_recall"
          << std::setw(10) << "L_recall"
          << std::setw(8) << "E_|w|"
          << std::setw(8) << "L_|w|"
          << "\n";

  double total_edge_ms = 0.0;
  double total_line_ms = 0.0;

  for (const auto & scene_path : scenes) {
    cv::Mat bev = cv::imread(scene_path.string(), cv::IMREAD_COLOR);
    if (bev.empty()) {
      continue;
    }
    bev = lie_lane_detection::prepareBevImage(bev);
    const std::string label = scene_path.parent_path().filename().string();
    const fs::path scene_out = output_dir / label;
    const auto gt = loadGroundTruth(scene_path.parent_path() / "ground_truth.txt");

    lie_lane_detection::BevDetectionResult edge_result;
    lie_lane_detection::BevDetectionResult line_result;
    EvalMetrics edge_eval;
    EvalMetrics line_eval;
    compareOnImage(
      bev, scene_out, label, params, gt,
      &edge_result, &line_result, &edge_eval, &line_eval);

    summary << std::left
            << std::setw(28) << label
            << std::setw(8) << edge_result.lanes.size()
            << std::setw(8) << line_result.lanes.size()
            << std::setw(10) << std::fixed << std::setprecision(1) << edge_result.elapsed_ms
            << std::setw(10) << line_result.elapsed_ms
            << std::setw(8) << line_result.line_segment_count;

    if (!gt.empty()) {
      summary << std::setw(10) << (std::to_string(edge_eval.matched) + "/" + std::to_string(edge_eval.gt_count))
              << std::setw(10) << (std::to_string(line_eval.matched) + "/" + std::to_string(line_eval.gt_count))
              << std::setw(8) << std::setprecision(3) << edge_eval.mean_omega_err
              << std::setw(8) << line_eval.mean_omega_err;
    }
    summary << "\n";

    total_edge_ms += edge_result.elapsed_ms;
    total_line_ms += line_result.elapsed_ms;
  }

  summary << "\nTotal edge ms: " << total_edge_ms << "\n";
  summary << "Total line ms: " << total_line_ms << "\n";
  summary << "Mean edge ms: " << (total_edge_ms / scenes.size()) << "\n";
  summary << "Mean line ms: " << (total_line_ms / scenes.size()) << "\n";
  summary << "Mean speedup (edge/line): " << (total_edge_ms / std::max(total_line_ms, 1e-6)) << "x\n";
  summary.close();

  std::cout << "\nSummary written to " << output_dir / "summary.txt" << "\n";
  return 0;
}
