// Batch lane detection on a folder of forward-camera images (e.g. Boreas camera/).
//
// When calib/ is present (or --calib-dir), builds metric ground-plane IPM from
// Boreas P_camera.txt + T_camera_lidar.txt. Otherwise falls back to auto-VP IPM.
//
// Usage:
//   lane_detect_dataset_offline --dataset /path/to/boreas-seq [--output DIR]
//   lane_detect_dataset_offline --images /path/to/camera --calib-dir /path/to/calib

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"
#include "lie_lane_detection/preprocessing/boreas_calib.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"
#include "lie_lane_detection/preprocessing/road_feature_segmenter.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"
#include "offline_display.hpp"

namespace fs = std::filesystem;
namespace lie = lie_lane_detection;

namespace
{

std::vector<fs::path> listImages(const fs::path & dir)
{
  std::vector<fs::path> paths;
  for (const auto & entry : fs::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const std::string ext = entry.path().extension().string();
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp") {
      paths.push_back(entry.path());
    }
  }
  std::sort(paths.begin(), paths.end());
  return paths;
}

lie::PipelineParams defaultParams()
{
  lie::PipelineParams p;
  p.use_steerable_filter = true;
  p.connect_dashed_edges = true;
  p.edge_anisotropic_blur = true;
  p.edge_thin = true;
  p.edge_low_threshold = 25.0;
  p.edge_high_threshold = 55.0;
  p.top_k_peaks = 8;
  p.max_lane_hypotheses = 8;
  p.use_iterative_peeling = true;
  p.vote_threshold_px = 9.0;
  p.inlier_threshold_px = 10.0;
  p.min_inlier_ratio = 0.36;
  p.min_inliers = 16;
  p.min_inlier_y_coverage = 0.32;
  p.max_output_lanes = 6;
  p.min_lane_relative_score = 0.40;
  p.max_lane_omega_deviation_rad = 0.16;
  p.min_lane_edge_orientation_ratio = 0.44;
  p.kappa_min = -0.28;
  p.kappa_max = 0.28;
  p.sigma_min = -0.55;
  p.sigma_max = 0.55;
  p.bev_bottom_exclude_px = 250.0;
  p.bev_bottom_edge_margin_px = 30.0;
  return p;
}

std::string laneSummary(const std::vector<lie::LaneHypothesis> & lanes)
{
  std::ostringstream oss;
  for (const auto & lane : lanes) {
    oss << "lane" << lane.lane_id
        << "(vx=" << std::fixed << std::setprecision(1) << lane.xi[0]
        << ",inl=" << std::setprecision(2) << lane.inlier_ratio << ") ";
  }
  return oss.str();
}

}  // namespace

int main(int argc, char ** argv)
{
  fs::path dataset_root;
  fs::path images_dir;
  fs::path calib_dir;
  fs::path output_dir = "/tmp/lie_lane_dataset";
  std::string calib_image;
  int stride = 30;
  int max_frames = 30;
  bool save_video = true;
  bool force_auto_ipm = false;
  bool auto_ground_ipm = false;
  bool show_windows = false;
  bool wait_ms_set = false;
  int wait_ms = 0;
  std::string road_mask_model;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--dataset" && i + 1 < argc) {
      dataset_root = fs::path(argv[++i]);
    } else if (arg == "--images" && i + 1 < argc) {
      images_dir = fs::path(argv[++i]);
    } else if (arg == "--calib-dir" && i + 1 < argc) {
      calib_dir = fs::path(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = fs::path(argv[++i]);
    } else if (arg == "--calib-image" && i + 1 < argc) {
      calib_image = argv[++i];
    } else if (arg == "--stride" && i + 1 < argc) {
      stride = std::max(1, std::stoi(argv[++i]));
    } else if (arg == "--max-frames" && i + 1 < argc) {
      max_frames = std::max(1, std::stoi(argv[++i]));
    } else if (arg == "--no-video") {
      save_video = false;
    } else if (arg == "--auto-ipm") {
      force_auto_ipm = true;
    } else if (arg == "--auto-ground-ipm") {
      auto_ground_ipm = true;
    } else if (arg == "--show" || arg == "--imshow") {
      show_windows = true;
    } else if (arg == "--wait-ms" && i + 1 < argc) {
      wait_ms = std::stoi(argv[++i]);
      wait_ms_set = true;
    } else if (arg == "--road-mask-model" && i + 1 < argc) {
      road_mask_model = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout <<
        "Usage: lane_detect_dataset_offline --dataset BOREAS_SEQ | --images CAMERA_DIR\n"
        "       [--calib-dir DIR] [--output DIR] [--stride N] [--max-frames N]\n"
        "  --dataset       Boreas root with camera/ (+ calib/ if present)\n"
        "  --images        Folder of PNG/JPG frames\n"
        "  --calib-dir     Boreas calib/ (optional; manual IPM trapezoid is default)\n"
        "  --auto-ipm      Force VP-based IPM instead of Boreas manual trapezoid\n"
        "  --auto-ground-ipm  Use P+T_camera_lidar ground quad instead of manual src\n"
        "  --calib-image   Frame for VP calibration when --auto-ipm is used\n"
        "  --stride        Process every Nth frame (default 30)\n"
        "  --max-frames    Cap processed frames (default 30)\n"
        "  --show          cv::imshow per frame (edges/ipm_overlay/frontal_overlay)\n"
        "  --wait-ms N     waitKey delay per frame (default 1 with --show; 0=step)\n"
        "  --road-mask-model PATH  ONNX tiny road/lane/snow segmenter for hybrid gate\n"
        "  --no-video      Skip overlay MP4\n";
      return 0;
    }
  }

  if (images_dir.empty()) {
    if (dataset_root.empty()) {
      std::cerr << "Provide --dataset or --images\n";
      return 1;
    }
    images_dir = dataset_root / "camera";
    if (calib_dir.empty()) {
      calib_dir = dataset_root / "calib";
    }
  }
  if (!fs::is_directory(images_dir)) {
    std::cerr << "Image directory not found: " << images_dir << "\n";
    return 1;
  }

  const auto all_images = listImages(images_dir);
  if (all_images.empty()) {
    std::cerr << "No images in " << images_dir << "\n";
    return 1;
  }

  if (show_windows && !wait_ms_set) {
    wait_ms = 1;
  }

  fs::create_directories(output_dir);
  fs::create_directories(output_dir / "frames");

  lie::PipelineParams params = defaultParams();
  lie::IPMTransformer ipm(params);
  lie::RoadFeatureSegmenter road_segmenter;
  if (!road_mask_model.empty()) {
    params.use_road_feature_segmenter = true;
    params.road_segmenter_onnx_path = road_mask_model;
    if (!road_segmenter.load(road_mask_model)) {
      std::cerr << "Failed to load road mask ONNX: " << road_mask_model << "\n";
      return 1;
    }
    road_segmenter.updateParams(params);
    std::cout << "Hybrid road segmenter: " << road_mask_model << "\n";
  }
  cv::Mat H_img2bev;
  std::string ipm_mode = "auto-vp";
  cv::Mat calib_debug;

  const bool has_boreas_calib =
    !calib_dir.empty() && fs::is_directory(calib_dir) &&
    fs::exists(calib_dir / "P_camera.txt");

  const bool use_boreas_manual =
    !force_auto_ipm && has_boreas_calib && !auto_ground_ipm;
  const bool use_boreas_ground = !force_auto_ipm && has_boreas_calib && auto_ground_ipm;

  if (use_boreas_manual) {
    lie::BoreasCalib boreas;
    if (!lie::loadBoreasCalib(calib_dir, boreas)) {
      std::cerr << "Failed to load Boreas calib from " << calib_dir << "\n";
      return 1;
    }
    if (!lie::configureBoreasManualIpmSrc(boreas, params, H_img2bev, &ipm)) {
      std::cerr << "Boreas manual IPM trapezoid failed\n";
      return 1;
    }
    ipm_mode = "boreas-manual-src";
    lie::configureParamsForBev(params, ipm.bevWidthPx(), ipm.bevHeightPx());
    lie::configureParamsForPerspectiveIpm(params);

    const fs::path preview_path = calib_image.empty() ? all_images.front() : fs::path(calib_image);
    calib_debug = cv::imread(preview_path.string(), cv::IMREAD_COLOR);
    if (!calib_debug.empty()) {
      cv::Mat bev_preview = ipm.warpToBev(calib_debug);
      lie::drawIpmRoiOnBev(bev_preview, H_img2bev, params);
      cv::imwrite((output_dir / "calib_bev.png").string(), bev_preview);
      lie::drawIpmMetricDstOnImage(calib_debug, H_img2bev, params);
      lie::drawIpmSrcRoi(calib_debug, params, cv::Scalar(0, 255, 255), 2);
    }

    std::cout << "Boreas manual IPM src trapezoid"
              << " BEV=" << ipm.bevWidthPx() << "x" << ipm.bevHeightPx() << "\n";
  } else if (use_boreas_ground) {
    lie::BoreasCalib boreas;
    if (!lie::loadBoreasCalib(calib_dir, boreas)) {
      std::cerr << "Failed to load Boreas calib from " << calib_dir << "\n";
      return 1;
    }
    if (!lie::configureBoreasGroundIpm(boreas, params, H_img2bev, &ipm)) {
      std::cerr << "Boreas ground-plane IPM failed\n";
      return 1;
    }
    ipm_mode = "boreas-ground-calib";
    lie::configureParamsForBev(params, ipm.bevWidthPx(), ipm.bevHeightPx());
    lie::configureParamsForPerspectiveIpm(params);

    const fs::path preview_path = calib_image.empty() ? all_images.front() : fs::path(calib_image);
    calib_debug = cv::imread(preview_path.string(), cv::IMREAD_COLOR);
    if (!calib_debug.empty()) {
      cv::Mat bev_preview = ipm.warpToBev(calib_debug);
      lie::drawIpmRoiOnBev(bev_preview, H_img2bev, params);
      cv::imwrite((output_dir / "calib_bev.png").string(), bev_preview);
      lie::drawIpmMetricDstOnImage(calib_debug, H_img2bev, params);
      lie::drawIpmSrcRoi(calib_debug, params, cv::Scalar(0, 255, 255), 2);
    }

    std::cout << "Boreas calib IPM from " << calib_dir
              << " BEV=" << ipm.bevWidthPx() << "x" << ipm.bevHeightPx() << "\n";
  } else {
    if (has_boreas_calib && force_auto_ipm) {
      std::cerr << "Note: --auto-ipm ignores Boreas calib/; omit it to use the manual IPM trapezoid (better on Boreas).\n";
    }
    lie::VanishingPointTracker vp_tracker;
    const fs::path calib_path = calib_image.empty() ? all_images.front() : fs::path(calib_image);
    calib_debug = cv::imread(calib_path.string(), cv::IMREAD_COLOR);
    if (calib_debug.empty()) {
      std::cerr << "Failed to read calibration image: " << calib_path << "\n";
      return 1;
    }

    std::cout << "Auto-IPM calibration from: " << calib_path.filename() << "\n";
    const lie::FrontalHomographyResult hg =
      lie::estimateFrontalHomography(calib_debug, params, &vp_tracker);
    if (!hg.valid || hg.H_img2bev.empty()) {
      std::cerr << "Auto-IPM calibration failed\n";
      return 1;
    }

    params = hg.params;
    lie::configureParamsForBev(params, hg.bev.cols, hg.bev.rows);
    lie::configureParamsForPerspectiveIpm(params);
    ipm.updateParams(params);
    if (!ipm.computeHomography(nullptr)) {
      std::cerr << "Failed to build frozen homography\n";
      return 1;
    }
    H_img2bev = hg.H_img2bev.clone();
    calib_debug = hg.debug_roi.clone();
    cv::imwrite((output_dir / "calib_ipm_roi.png").string(), hg.debug_roi);
    cv::imwrite((output_dir / "calib_bev.png").string(), hg.bev);
    std::cout << "Frozen auto-IPM: VP=(" << hg.vanishing_point.x << "," << hg.vanishing_point.y
              << ") fallback=" << hg.used_fallback_roi << "\n";
  }

  if (!calib_debug.empty()) {
    cv::imwrite((output_dir / "calib_input.png").string(), calib_debug);
  }

  std::ofstream calib_report(output_dir / "calibration.txt");
  calib_report << "ipm_mode=" << ipm_mode << "\n";
  calib_report << "dataset=" << (dataset_root.empty() ? images_dir.string() : dataset_root.string())
               << "\n";
  if (use_boreas_manual || use_boreas_ground) {
    calib_report << "calib_dir=" << calib_dir.string() << "\n";
  }
  calib_report << "bev_size=" << ipm.bevWidthPx() << "x" << ipm.bevHeightPx() << "\n";
  calib_report << "ipm_src_points (image px):";
  for (size_t i = 0; i < params.ipm_src_points.size(); i += 2) {
    calib_report << " (" << params.ipm_src_points[i] << "," << params.ipm_src_points[i + 1]
                 << ")";
  }
  calib_report << "\n";
  calib_report << "ipm_dst_points (m):";
  for (size_t i = 0; i < params.ipm_dst_points.size(); i += 2) {
    calib_report << " (" << params.ipm_dst_points[i] << "," << params.ipm_dst_points[i + 1]
                 << ")";
  }
  calib_report << "\n";
  calib_report.close();

  std::ofstream summary(output_dir / "summary.csv");
  summary << "frame,lanes,edges_ms,vote_ms,fit_ms,total_ms,edge_points,lane_summary\n";

  cv::VideoWriter video;
  int processed = 0;
  double total_detect_ms = 0.0;

  for (size_t idx = 0; idx < all_images.size() && processed < max_frames; idx += static_cast<size_t>(stride)) {
    const fs::path & img_path = all_images[idx];
    cv::Mat frame = cv::imread(img_path.string(), cv::IMREAD_COLOR);
    if (frame.empty()) {
      std::cerr << "Skip unreadable: " << img_path << "\n";
      continue;
    }

    cv::Mat bev = ipm.warpToBev(frame);
    if (bev.empty()) {
      std::cerr << "Warp failed: " << img_path.filename() << "\n";
      continue;
    }
    bev = lie::prepareBevImage(bev);
    lie::maskBevBottomExclude(bev, params);

    const lie::BevDetectionResult det = lie::detectLanesInBev(
      bev, params, /*configure=*/false,
      road_segmenter.isReady() ? &road_segmenter : nullptr);
    cv::Mat overlay = lie::drawOverlay(bev, det.lanes, det.merges);
    cv::Mat frontal_overlay = lie::drawFrontalOverlay(frame, det.lanes, det.merges, H_img2bev);
    lie::drawIpmRoiOnBev(overlay, H_img2bev, params);
    lie::drawIpmSrcRoi(frontal_overlay, params, cv::Scalar(0, 255, 255), 2);
    lie::drawIpmMetricDstOnImage(frontal_overlay, H_img2bev, params);

    const std::string stem = img_path.stem().string();
    cv::Mat bev_vis = bev.clone();
    lie::drawIpmRoiOnBev(bev_vis, H_img2bev, params);
    cv::imwrite((output_dir / "frames" / (stem + "_bev.png")).string(), bev_vis);
    const cv::Mat filtered_edges =
      lie::filterBevEdgeArtifacts(det.edges, bev, params, H_img2bev);
    const cv::Mat edge_vis = lie::offline_display::composeCleanEdgeView(filtered_edges);
    cv::imwrite((output_dir / "frames" / (stem + "_edges.png")).string(), edge_vis);
    if (!det.road_feature_debug.empty()) {
      cv::imwrite((output_dir / "frames" / (stem + "_road_mask.png")).string(), det.road_feature_debug);
    }
    cv::imwrite((output_dir / "frames" / (stem + "_overlay.png")).string(), overlay);
    cv::imwrite((output_dir / "frames" / (stem + "_frontal_overlay.png")).string(), frontal_overlay);

    if (save_video) {
      if (!video.isOpened()) {
        const int fourcc = cv::VideoWriter::fourcc('m', 'p', '4', 'v');
        video.open(
          (output_dir / "overlay.mp4").string(), fourcc, 5.0,
          cv::Size(overlay.cols, overlay.rows));
      }
      if (video.isOpened()) {
        video.write(overlay);
      }
    }

    summary << stem << "," << det.lanes.size() << ","
            << det.edge_ms << "," << det.vote_ms << "," << det.fit_ms << ","
            << det.elapsed_ms << "," << det.edge_point_count << ","
            << "\"" << laneSummary(det.lanes) << "\"\n";

    total_detect_ms += det.elapsed_ms;
    ++processed;

    std::cout << "[" << processed << "/" << max_frames << "] " << stem
              << " lanes=" << det.lanes.size()
              << " ms=" << std::fixed << std::setprecision(1) << det.elapsed_ms
              << " " << laneSummary(det.lanes) << "\n";

    if (show_windows) {
      lie::offline_display::show("edges", edge_vis);
      if (!det.road_feature_debug.empty()) {
        lie::offline_display::show("road_mask", det.road_feature_debug);
      }
      lie::offline_display::show("ipm_overlay", overlay);
      lie::offline_display::show("frontal_overlay", frontal_overlay);
      if (!lie::offline_display::wait(wait_ms)) {
        break;
      }
    }
  }

  if (show_windows) {
    lie::offline_display::destroyAll();
  }

  summary.close();
  if (video.isOpened()) {
    video.release();
  }

  std::cout << "\nProcessed " << processed << " frame(s) from " << all_images.size()
            << " total (stride=" << stride << ")\n";
  if (processed > 0) {
    std::cout << "Mean detect ms: " << (total_detect_ms / processed) << "\n";
  }
  std::cout << "Output: " << output_dir << "\n";
  std::cout << "  calibration.txt, summary.csv, frames/*, overlay.mp4\n";
  std::cout << "  frames: *_edges.png, *_overlay.png, *_frontal_overlay.png\n";
  return 0;
}
