// Offline runner: test lane detection on a BEV/IPM image (no ROS, no camera).
// Usage:
//   lane_detect_offline --image /path/to/bev.png --output /path/to/out_dir [--skip-ipm]

#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"
#include "lie_lane_detection/preprocessing/boreas_calib.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"
#include "offline_display.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char ** argv)
{
  std::string image_path;
  fs::path output_dir = "/tmp/lie_lane_results";
  bool synthetic = false;
  bool perspective = false;
  bool frontal = false;
  bool auto_ipm = false;
  bool show_windows = false;
  int wait_ms = 0;
  fs::path boreas_calib_dir;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = fs::path(argv[++i]);
    } else if (arg == "--boreas-calib" && i + 1 < argc) {
      boreas_calib_dir = fs::path(argv[++i]);
    } else if (arg == "--synthetic") {
      synthetic = true;
    } else if (arg == "--perspective") {
      perspective = true;
    } else if (arg == "--frontal") {
      perspective = false;
      frontal = true;
    } else if (arg == "--auto-ipm") {
      perspective = false;
      frontal = false;
      auto_ipm = true;
    } else if (arg == "--show" || arg == "--imshow") {
      show_windows = true;
    } else if (arg == "--wait-ms" && i + 1 < argc) {
      wait_ms = std::stoi(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: lane_detect_offline [--synthetic | --image PATH] [--output DIR]\n"
                   "       [--perspective | --auto-ipm | --boreas-calib DIR | --frontal]\n"
                   "  --synthetic     Generate synthetic 3-lane BEV and run detection\n"
                   "  --image PATH    Load image (BEV, or forward camera with IPM flags)\n"
                   "  --perspective   Manual highway IPM trapezoid\n"
                   "  --auto-ipm      Auto VP + homography from image (recommended for frontal)\n"
                   "  --boreas-calib  Boreas calib/ with P_camera + T_camera_lidar\n"
                   "  --frontal       Detect on raw forward camera (no IPM)\n"
                   "  --show          cv::imshow windows (input/BEV/edges/overlay)\n"
                   "  --wait-ms N     waitKey delay (0=until key, default 0)\n"
                   "  --output DIR    Save overlay, edges, hough, report.txt\n";
      return 0;
    }
  }

  fs::create_directories(output_dir);

  lie_lane_detection::PipelineParams params;
  params.use_steerable_filter = true;
  params.connect_dashed_edges = true;
  params.top_k_peaks = 8;
  params.max_lane_hypotheses = 8;
  params.use_iterative_peeling = true;
  params.edge_low_threshold = 20.0;
  params.edge_high_threshold = 60.0;
  params.vote_threshold_px = 7.0;
  params.inlier_threshold_px = 9.0;
  params.min_inlier_ratio = 0.42;
  params.min_inliers = 12;
  params.kappa_min = -0.2;
  params.kappa_max = 0.2;
  params.sigma_min = -0.15;
  params.sigma_max = 0.15;

  cv::Mat bev;
  cv::Mat input_view;
  cv::Mat ipm_roi_view;
  if (synthetic || image_path.empty()) {
    const int w = static_cast<int>(params.bev_width_m / params.bev_resolution_m_per_px);
    const int h = static_cast<int>(params.bev_length_m / params.bev_resolution_m_per_px);
    bev = cv::Mat(h, w, CV_8UC3, cv::Scalar(40, 40, 40));

    lie_lane_detection::TemplateCurve curve(params);
    curve.setBevExtents(0.0, static_cast<double>(h), 0.0, static_cast<double>(w));

    const std::array<double, 3> offsets = {w * 0.25, w * 0.5, w * 0.75};
    for (double vx : offsets) {
      lie_lane_detection::XiVector xi = lie_lane_detection::XiVector::Zero();
      xi[0] = vx;
      const auto poly = curve.samplePolyline(xi, 80);
      for (size_t i = 1; i < poly.size(); ++i) {
        cv::line(
          bev, cv::Point(static_cast<int>(poly[i - 1].x()), static_cast<int>(poly[i - 1].y())),
          cv::Point(static_cast<int>(poly[i].x()), static_cast<int>(poly[i].y())),
          cv::Scalar(220, 220, 220), 3);
      }
    }
    cv::imwrite((output_dir / "input_synthetic_bev.png").string(), bev);
    std::cout << "Generated synthetic BEV: " << w << "x" << h << "\n";
  } else {
    bev = cv::imread(image_path, cv::IMREAD_COLOR);
    if (bev.empty()) {
      std::cerr << "Failed to read image: " << image_path << "\n";
      return 1;
    }
    cv::imwrite((output_dir / "input_image.png").string(), bev);
    input_view = bev.clone();
    std::cout << "Loaded image: " << image_path << " (" << bev.cols << "x" << bev.rows << ")\n";
    if (perspective) {
      lie_lane_detection::setDefaultHighwayIpmRoi(params, bev.cols, bev.rows);
      lie_lane_detection::configureParamsForPerspectiveIpm(params);
      cv::Mat warped = lie_lane_detection::warpPerspectiveToBev(bev, params);
      if (warped.empty()) {
        std::cerr << "IPM warp failed\n";
        return 1;
      }
      cv::imwrite((output_dir / "ipm_bev.png").string(), warped);
      std::cout << "IPM BEV: " << warped.cols << "x" << warped.rows << "\n";
      bev = lie_lane_detection::prepareBevImage(warped);
    } else if (!boreas_calib_dir.empty()) {
      lie_lane_detection::BoreasCalib calib;
      cv::Mat H;
      lie_lane_detection::IPMTransformer ipm(params);
      if (!lie_lane_detection::loadBoreasCalib(boreas_calib_dir, calib)) {
        std::cerr << "Failed to load Boreas calib from " << boreas_calib_dir << "\n";
        return 1;
      }
      if (!lie_lane_detection::configureBoreasManualIpmSrc(calib, params, H, &ipm)) {
        std::cerr << "Boreas manual IPM failed\n";
        return 1;
      }
      lie_lane_detection::configureParamsForBev(params, ipm.bevWidthPx(), ipm.bevHeightPx());
      lie_lane_detection::configureParamsForPerspectiveIpm(params);
      cv::Mat warped = ipm.warpToBev(bev);
      if (warped.empty()) {
        std::cerr << "Boreas IPM warp failed\n";
        return 1;
      }
      cv::imwrite((output_dir / "ipm_bev.png").string(), warped);
      std::cout << "Boreas manual IPM src trapezoid BEV: " << warped.cols << "x" << warped.rows
                << "\n";
      ipm_roi_view = input_view.clone();
      lie_lane_detection::drawIpmMetricDstOnImage(ipm_roi_view, H, params);
      lie_lane_detection::drawIpmSrcRoi(ipm_roi_view, params, cv::Scalar(0, 255, 255), 2);
      cv::Mat bev_preview = warped.clone();
      lie_lane_detection::drawIpmRoiOnBev(bev_preview, H, params);
      cv::imwrite((output_dir / "ipm_bev_roi.png").string(), bev_preview);
      bev = lie_lane_detection::prepareBevImage(warped);
    } else if (auto_ipm) {
      const auto hg = lie_lane_detection::estimateFrontalHomography(bev, params);
      if (!hg.valid) {
        std::cerr << "Auto homography failed\n";
        return 1;
      }
      cv::imwrite((output_dir / "auto_ipm_roi.png").string(), hg.debug_roi);
      cv::imwrite((output_dir / "ipm_bev.png").string(), hg.bev);
      ipm_roi_view = hg.debug_roi.clone();
      params = hg.params;
      lie_lane_detection::configureParamsForBev(params, hg.bev.cols, hg.bev.rows);
      lie_lane_detection::configureParamsForPerspectiveIpm(params);
      std::cout << "Auto-IPM VP: (" << hg.vanishing_point.x << ", " << hg.vanishing_point.y
                << ") fallback=" << hg.used_fallback_roi << "\n";
      std::cout << "BEV: " << hg.bev.cols << "x" << hg.bev.rows << "\n";
      bev = hg.bev;
    } else if (frontal) {
      lie_lane_detection::configureParamsForFrontalImage(params, bev.cols, bev.rows);
      std::cout << "Frontal mode (no IPM): " << bev.cols << "x" << bev.rows << "\n";
      bev = lie_lane_detection::prepareFrontalImage(bev);
    } else {
      bev = lie_lane_detection::prepareBevImage(bev);
    }
  }

  const auto result = lie_lane_detection::detectLanesInBev(bev, params);
  const cv::Mat edge_view = lie_lane_detection::offline_display::composeEdgeView(bev, result.edges);

  cv::imwrite((output_dir / "edges.png").string(), edge_view);
  cv::imwrite((output_dir / "hough_slice.png").string(), result.hough_slice);
  cv::imwrite((output_dir / "overlay.png").string(), result.overlay);

  std::ofstream report(output_dir / "report.txt");
  report << "Lie Lane Detection — Offline Report\n";
  report << "====================================\n";
  report << "Edge points: " << result.edge_point_count << "\n";
  report << "Elapsed ms: " << result.elapsed_ms << "\n";
  report << "Lanes detected: " << result.lanes.size() << "\n";
  report << "Merge events: " << result.merges.size() << "\n\n";

  for (const auto & lane : result.lanes) {
    report << "Lane " << lane.lane_id << "  xi=(vx=" << lane.xi[0] << ", vy=" << lane.xi[1]
           << ", w=" << lane.xi[2] << ", k=" << lane.xi[3] << ", s=" << lane.xi[4] << ")"
           << "  score=" << lane.score << "  inliers=" << lane.inlier_ratio << "\n";
  }
  for (const auto & m : result.merges) {
    report << "Merge: lanes " << m.lane_a_id << "+" << m.lane_b_id
           << " type=" << static_cast<int>(m.type) << " pt=(" << m.merge_point.x() << ","
           << m.merge_point.y() << ")\n";
  }
  report.close();

  std::cout << "\nResults saved to: " << output_dir << "\n";
  std::cout << "  edges.png, hough_slice.png, overlay.png, report.txt\n";
  std::cout << "Detected " << result.lanes.size() << " lane(s), " << result.edge_point_count
            << " edge points in " << result.elapsed_ms << " ms\n";
  for (const auto & lane : result.lanes) {
    std::cout << "  Lane " << lane.lane_id << ": vx=" << lane.xi[0] << " kappa=" << lane.xi[3]
              << " inliers=" << lane.inlier_ratio << "\n";
  }

  if (show_windows) {
    using lie_lane_detection::offline_display::destroyAll;
    using lie_lane_detection::offline_display::show;
    using lie_lane_detection::offline_display::wait;
    if (!input_view.empty()) {
      show("input", input_view);
    }
    if (!ipm_roi_view.empty()) {
      show("ipm_roi", ipm_roi_view);
    }
    show("bev", bev);
    show("edges", edge_view);
    if (!result.hough_slice.empty()) {
      show("hough", result.hough_slice);
    }
    show("overlay", result.overlay);
    std::cout << "imshow: any key=close, q/Esc=quit\n";
    if (!wait(wait_ms)) {
      destroyAll();
      return 0;
    }
    destroyAll();
  }
  return 0;
}
