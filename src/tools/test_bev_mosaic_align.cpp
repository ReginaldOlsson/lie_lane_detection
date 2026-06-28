// Offline pairwise BEV alignment test (ECC / ORB) with debug images.
//
// Usage:
//   test_bev_mosaic_align --prev first.png --curr second.png --output /tmp/bev_align

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include <opencv2/imgcodecs.hpp>

#include "lie_lane_detection/mosaic/bev_registration.hpp"

namespace fs = std::filesystem;

namespace
{

lie_lane_detection::BevRegistrationMethod parseMethod(const std::string & value)
{
  if (value == "ecc") {
    return lie_lane_detection::BevRegistrationMethod::ECC;
  }
  if (value == "orb") {
    return lie_lane_detection::BevRegistrationMethod::ORB;
  }
  return lie_lane_detection::BevRegistrationMethod::ECC_THEN_ORB;
}

}  // namespace

int main(int argc, char ** argv)
{
  std::string prev_path;
  std::string curr_path;
  fs::path output_dir = "/tmp/bev_align";
  std::string method = "auto";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--prev" && i + 1 < argc) {
      prev_path = argv[++i];
    } else if (arg == "--curr" && i + 1 < argc) {
      curr_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = fs::path(argv[++i]);
    } else if (arg == "--method" && i + 1 < argc) {
      method = argv[++i];
    } else if (arg == "--help" || arg == "-h") {
      std::cout <<
        "Usage: test_bev_mosaic_align --prev PATH --curr PATH [--output DIR] [--method auto|ecc|orb]\n";
      return 0;
    }
  }

  if (prev_path.empty() || curr_path.empty()) {
    std::cerr << "Provide --prev and --curr image paths\n";
    return 1;
  }

  const cv::Mat prev = cv::imread(prev_path, cv::IMREAD_COLOR);
  const cv::Mat curr = cv::imread(curr_path, cv::IMREAD_COLOR);
  if (prev.empty() || curr.empty()) {
    std::cerr << "Failed to load images\n";
    return 1;
  }
  if (prev.size() != curr.size()) {
    std::cerr << "Image sizes differ: prev " << prev.cols << "x" << prev.rows
              << " curr " << curr.cols << "x" << curr.rows << "\n";
    return 1;
  }

  lie_lane_detection::BevRegistrationParams params;
  params.method = parseMethod(method == "auto" ? "ecc_then_orb" : method);
  if (method == "auto") {
    params.method = lie_lane_detection::BevRegistrationMethod::ECC_THEN_ORB;
  }

  const auto reg = lie_lane_detection::estimateBevFrameMotion(prev, curr, params);
  const auto debug = lie_lane_detection::makeBevAlignmentDebug(prev, curr, reg, params);

  fs::create_directories(output_dir);
  cv::imwrite((output_dir / "00_prev.png").string(), prev);
  cv::imwrite((output_dir / "01_curr.png").string(), curr);
  cv::imwrite((output_dir / "02_curr_aligned.png").string(), debug.curr_aligned);
  cv::imwrite((output_dir / "03_blend.png").string(), debug.blend);
  cv::imwrite((output_dir / "04_abs_diff.png").string(), debug.abs_diff);
  cv::imwrite((output_dir / "05_side_by_side.png").string(), debug.side_by_side);
  cv::imwrite((output_dir / "06_valid_mask.png").string(), debug.valid_mask);

  std::ofstream report(output_dir / "alignment_report.txt");
  report << std::fixed << std::setprecision(4);
  report << "valid=" << reg.valid << "\n";
  report << "method=" << reg.method_used << "\n";
  report << "correlation=" << reg.correlation << "\n";
  report << "inliers=" << reg.inlier_count << "\n";
  report << "dx_px=" << reg.dx_px << "\n";
  report << "dy_px=" << reg.dy_px << "\n";
  report << "yaw_deg=" << (reg.yaw_rad * 180.0 / CV_PI) << "\n";
  report << "relative_affine_2x3=\n" << reg.relative_affine_2x3 << "\n";
  report << "relative_transform_3x3=\n" << reg.relative_transform_3x3 << "\n";
  report.close();

  std::cout << "Alignment report written to " << (output_dir / "alignment_report.txt") << "\n";
  std::cout << "  valid=" << reg.valid
            << " method=" << reg.method_used
            << " corr=" << reg.correlation
            << " dx=" << reg.dx_px
            << " dy=" << reg.dy_px
            << " yaw_deg=" << (reg.yaw_rad * 180.0 / CV_PI) << "\n";
  std::cout << "Debug images in " << output_dir << "\n";
  return reg.valid ? 0 : 2;
}
