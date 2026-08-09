// Estimate homography / IPM parameters from a frontal camera image.
//
// Usage:
//   estimate_frontal_homography --image PATH --output DIR

#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"

#include <opencv2/imgcodecs.hpp>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace fs = std::filesystem;

int main(int argc, char ** argv)
{
  std::string image_path;
  fs::path output_dir = "/tmp/frontal_homography";

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--image" && i + 1 < argc) {
      image_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      output_dir = fs::path(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: estimate_frontal_homography --image PATH [--output DIR]\n"
                   "  Estimates vanishing point, ipm_src_points, homography H, and BEV warp.\n";
      return 0;
    }
  }

  if (image_path.empty()) {
    std::cerr << "Provide --image PATH\n";
    return 1;
  }

  cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
  if (image.empty()) {
    std::cerr << "Failed to read: " << image_path << "\n";
    return 1;
  }

  fs::create_directories(output_dir);
  const auto hg = lie_lane_detection::estimateFrontalHomography(image);

  cv::imwrite((output_dir / "input.png").string(), image);
  cv::imwrite((output_dir / "auto_ipm_roi.png").string(), hg.debug_roi);
  if (!hg.bev.empty()) {
    cv::imwrite((output_dir / "ipm_bev.png").string(), hg.bev);
  }

  std::ofstream report(output_dir / "homography.txt");
  report << "Frontal homography estimation\n";
  report << "Image: " << image_path << " (" << image.cols << "x" << image.rows << ")\n";
  report << "Valid: " << (hg.valid ? "yes" : "no") << "\n";
  report << "Fallback highway ROI: " << (hg.used_fallback_roi ? "yes" : "no") << "\n";
  report << "Vanishing point: (" << hg.vanishing_point.x << ", " << hg.vanishing_point.y
         << ") conf=" << hg.vanishing_point.confidence << " valid=" << hg.vanishing_point.valid
         << "\n\n";

  report << "ipm_src_points (px): ";
  for (size_t i = 0; i < hg.params.ipm_src_points.size(); i += 2) {
    report << "(" << hg.params.ipm_src_points[i] << "," << hg.params.ipm_src_points[i + 1] << ") ";
  }
  report << "\n";

  report << "ipm_dst_points (m): ";
  for (size_t i = 0; i < hg.params.ipm_dst_points.size(); i += 2) {
    report << "(" << hg.params.ipm_dst_points[i] << "," << hg.params.ipm_dst_points[i + 1] << ") ";
  }
  report << "\n\n";

  report << "H_img2bev (3x3):\n";
  if (!hg.H_img2bev.empty()) {
    for (int r = 0; r < hg.H_img2bev.rows; ++r) {
      for (int c = 0; c < hg.H_img2bev.cols; ++c) {
        report << std::setw(14) << std::fixed << std::setprecision(6)
               << hg.H_img2bev.at<double>(r, c) << " ";
      }
      report << "\n";
    }
  }

  report << "\nYAML snippet for lane_detector.yaml:\n";
  report << "  ipm_src_points: [";
  for (size_t i = 0; i < hg.params.ipm_src_points.size(); ++i) {
    report << hg.params.ipm_src_points[i];
    if (i + 1 < hg.params.ipm_src_points.size()) {
      report << ", ";
    }
  }
  report << "]\n";

  std::cout << "Wrote homography to " << output_dir << "\n";
  if (hg.vanishing_point.valid) {
    std::cout << "VP: (" << hg.vanishing_point.x << ", " << hg.vanishing_point.y << ")\n";
  } else {
    std::cout << "VP not found — used default highway trapezoid\n";
  }
  return hg.valid ? 0 : 1;
}
