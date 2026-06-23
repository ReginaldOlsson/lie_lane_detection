// Run lane detection on a forward-camera video.
//
// Usage:
//   lane_detect_video_offline --video PATH --output DIR [--auto-ipm] [--frontal-raw] [--both]

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>

#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/pipeline/line_lane_detection_runner.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"

namespace fs = std::filesystem;

enum class InputMode
{
  AUTO_IPM,
  FRONTAL_RAW
};

static lie_lane_detection::PipelineParams defaultParams()
{
  lie_lane_detection::PipelineParams p;
  p.use_steerable_filter = true;
  p.connect_dashed_edges = true;
  p.top_k_peaks = 10;
  p.max_lane_hypotheses = 10;
  p.use_iterative_peeling = true;
  p.edge_low_threshold = 25.0;
  p.edge_high_threshold = 70.0;
  p.vote_threshold_px = 8.0;
  p.inlier_threshold_px = 10.0;
  p.min_inlier_ratio = 0.30;
  p.min_inliers = 15;
  p.kappa_min = -0.28;
  p.kappa_max = 0.28;
  p.sigma_min = -0.55;
  p.sigma_max = 0.55;
  p.line_hough_threshold = 28;
  p.line_min_length_px = 20.0;
  p.line_max_gap_px = 16.0;
  p.line_angle_threshold_rad = 0.75;
  return p;
}

static std::string laneSummary(const std::vector<lie_lane_detection::LaneHypothesis> & lanes)
{
  std::ostringstream oss;
  for (const auto & lane : lanes) {
    oss << "lane" << lane.lane_id
        << "(vx=" << std::fixed << std::setprecision(0) << lane.xi[0]
        << ",inl=" << std::setprecision(2) << lane.inlier_ratio << ") ";
  }
  return oss.str();
}

static cv::Mat prepareInput(
  const cv::Mat & frame,
  InputMode mode,
  lie_lane_detection::PipelineParams & params,
  lie_lane_detection::FrontalHomographyResult * hg_out = nullptr)
{
  if (mode == InputMode::FRONTAL_RAW) {
    lie_lane_detection::configureParamsForFrontalImage(params, frame.cols, frame.rows);
    return lie_lane_detection::prepareFrontalImage(frame);
  }

  lie_lane_detection::FrontalHomographyResult hg =
    lie_lane_detection::estimateFrontalHomography(frame, params);
  if (hg_out) {
    *hg_out = hg;
  }
  params = hg.params;
  lie_lane_detection::configureParamsForBev(params, hg.bev.cols, hg.bev.rows);
  lie_lane_detection::configureParamsForPerspectiveIpm(params);
  return hg.bev;
}

int main(int argc, char ** argv)
{
  std::string video_path;
  fs::path output_dir = "/tmp/lie_lane_video";
  InputMode mode = InputMode::AUTO_IPM;
  bool both = true;
  bool save_video = true;
  int stride = 30;
  int max_frames = 20;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--video" && i + 1 < argc) {
      video_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = fs::path(argv[++i]);
    } else if (arg == "--stride" && i + 1 < argc) {
      stride = std::stoi(argv[++i]);
    } else if (arg == "--max-frames" && i + 1 < argc) {
      max_frames = std::stoi(argv[++i]);
    } else if (arg == "--auto-ipm") {
      mode = InputMode::AUTO_IPM;
    } else if (arg == "--frontal-raw") {
      mode = InputMode::FRONTAL_RAW;
    } else if (arg == "--both") {
      both = true;
    } else if (arg == "--edge-only") {
      both = false;
    } else if (arg == "--no-save-video") {
      save_video = false;
    } else if (arg == "--help" || arg == "-h") {
      std::cout <<
        "Usage: lane_detect_video_offline --video PATH --output DIR [options]\n"
        "  --auto-ipm       Estimate VP + homography per frame, detect on BEV (default)\n"
        "  --frontal-raw    Detect on raw image (experimental, poor quality)\n"
        "  --both           Edge + line pipelines (default)\n"
        "  --stride N       Every Nth frame (default 30)\n"
        "  --max-frames N   Max frames (default 20)\n";
      return 0;
    }
  }

  if (video_path.empty()) {
    std::cerr << "Provide --video PATH\n";
    return 1;
  }

  cv::VideoCapture cap(video_path);
  if (!cap.isOpened()) {
    std::cerr << "Failed to open video: " << video_path << "\n";
    return 1;
  }

  const int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
  const double fps = cap.get(cv::CAP_PROP_FPS);
  const int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
  const int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

  fs::create_directories(output_dir);
  fs::create_directories(output_dir / "frames");
  fs::create_directories(output_dir / "edge_overlays");
  fs::create_directories(output_dir / "line_overlays");
  if (mode == InputMode::AUTO_IPM) {
    fs::create_directories(output_dir / "auto_ipm_roi");
    fs::create_directories(output_dir / "frontal_overlays");
  }

  const char * mode_str = (mode == InputMode::AUTO_IPM) ? "auto-ipm" : "frontal-raw";
  std::cout << "Video: " << video_path << " " << width << "x" << height
            << " @" << fps << " fps, ~" << total_frames << " frames\n";
  std::cout << "Mode: " << mode_str << ", stride=" << stride
            << ", max_frames=" << max_frames << "\n";

  cv::VideoWriter edge_writer;
  cv::VideoWriter line_writer;
  cv::VideoWriter edge_frontal_writer;
  cv::VideoWriter line_frontal_writer;
  cv::Size bev_writer_size(width, height);
  cv::Size frontal_writer_size(width, height);

  std::ofstream report(output_dir / "video_report.txt");
  report << "Video lane detection\n";
  report << "Mode: " << mode_str << "\n";
  report << "Video: " << video_path << "\n\n";
  report << std::left
         << std::setw(8) << "Frame"
         << std::setw(8) << "E_lanes"
         << std::setw(8) << "L_lanes"
         << std::setw(10) << "E_ms"
         << std::setw(10) << "L_ms"
         << "Notes\n";

  auto params = defaultParams();
  int frame_idx = 0;
  int processed = 0;
  double sum_edge_ms = 0.0;
  double sum_line_ms = 0.0;

  cv::Mat frame;
  while (processed < max_frames && cap.read(frame)) {
    if (frame_idx % stride != 0) {
      ++frame_idx;
      continue;
    }

    lie_lane_detection::FrontalHomographyResult hg;
    cv::Mat prepared = prepareInput(frame, mode, params, &hg);

    if (processed == 0 && mode == InputMode::AUTO_IPM && !prepared.empty()) {
      bev_writer_size = prepared.size();
      frontal_writer_size = frame.size();
      if (save_video) {
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        const double out_fps = std::max(fps / stride, 1.0);
        edge_writer.open(
          (output_dir / "edge_overlay.mp4").string(), fourcc, out_fps, bev_writer_size);
        edge_frontal_writer.open(
          (output_dir / "edge_frontal_overlay.mp4").string(), fourcc,
          out_fps, frontal_writer_size);
        if (both) {
          line_writer.open(
            (output_dir / "line_overlay.mp4").string(), fourcc, out_fps, bev_writer_size);
          line_frontal_writer.open(
            (output_dir / "line_frontal_overlay.mp4").string(), fourcc,
            out_fps, frontal_writer_size);
        }
      }
    } else if (processed == 0 && mode == InputMode::FRONTAL_RAW && save_video) {
      const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
      const double out_fps = std::max(fps / stride, 1.0);
      edge_writer.open(
        (output_dir / "edge_overlay.mp4").string(), fourcc, out_fps, frontal_writer_size);
      if (both) {
        line_writer.open(
          (output_dir / "line_overlay.mp4").string(), fourcc, out_fps, frontal_writer_size);
      }
    }

    if (prepared.empty()) {
      ++frame_idx;
      continue;
    }

    const auto edge_result = lie_lane_detection::detectLanesInBev(prepared, params);
    lie_lane_detection::BevDetectionResult line_result;
    if (both) {
      line_result = lie_lane_detection::detectLanesInBevFromLines(prepared, params);
    }

    const std::string tag =
      (std::ostringstream{} << std::setfill('0') << std::setw(5) << frame_idx).str();
    cv::imwrite((output_dir / "frames" / (tag + "_input.png")).string(), prepared);
    cv::imwrite((output_dir / "edge_overlays" / (tag + "_edge.png")).string(), edge_result.overlay);
    if (both) {
      cv::imwrite(
        (output_dir / "line_overlays" / (tag + "_line.png")).string(),
        line_result.overlay);
    }
    if (mode == InputMode::AUTO_IPM) {
      cv::imwrite((output_dir / "auto_ipm_roi" / (tag + "_roi.png")).string(), hg.debug_roi);
      const cv::Mat edge_frontal = lie_lane_detection::drawFrontalOverlay(
        frame, edge_result.lanes, edge_result.merges, hg.H_img2bev);
      cv::imwrite(
        (output_dir / "frontal_overlays" / (tag + "_edge.png")).string(),
        edge_frontal);
      if (save_video && edge_frontal_writer.isOpened()) {
        edge_frontal_writer.write(edge_frontal);
      }
      if (both) {
        const cv::Mat line_frontal = lie_lane_detection::drawFrontalOverlay(
          frame, line_result.lanes, line_result.merges, hg.H_img2bev);
        cv::imwrite(
          (output_dir / "frontal_overlays" / (tag + "_line.png")).string(),
          line_frontal);
        if (save_video && line_frontal_writer.isOpened()) {
          line_frontal_writer.write(line_frontal);
        }
      }
    }

    if (save_video && edge_writer.isOpened()) {
      edge_writer.write(edge_result.overlay);
    }
    if (both && save_video && line_writer.isOpened()) {
      line_writer.write(line_result.overlay);
    }

    report << std::left
           << std::setw(8) << frame_idx
           << std::setw(8) << edge_result.lanes.size()
           << std::setw(8) << (both ? line_result.lanes.size() : 0)
           << std::setw(10) << std::fixed << std::setprecision(1) << edge_result.elapsed_ms
           << std::setw(10) << (both ? line_result.elapsed_ms : 0.0)
           << laneSummary(edge_result.lanes) << "\n";

    std::cout << "frame " << frame_idx << ": edge " << edge_result.lanes.size()
              << " / " << edge_result.elapsed_ms << " ms";
    if (both) {
      std::cout << " | line " << line_result.lanes.size()
                << " / " << line_result.elapsed_ms << " ms";
    }
    if (mode == InputMode::AUTO_IPM && hg.vanishing_point.valid) {
      std::cout << " | VP(" << std::fixed << std::setprecision(0)
                << hg.vanishing_point.x << "," << hg.vanishing_point.y << ")";
    }
    std::cout << "\n";

    sum_edge_ms += edge_result.elapsed_ms;
    sum_line_ms += both ? line_result.elapsed_ms : 0.0;
    ++processed;
    ++frame_idx;
  }

  if (processed > 0) {
    report << "\nMean edge ms: " << (sum_edge_ms / processed) << "\n";
    report << "Mean line ms: " << (sum_line_ms / processed) << "\n";
  }
  report.close();

  std::cout << "\nResults in " << output_dir << "\n";
  return 0;
}
