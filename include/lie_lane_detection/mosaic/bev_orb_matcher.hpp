#pragma once

#include <opencv2/core.hpp>

#include <vector>

namespace lie_lane_detection
{

enum class BevOrbMatchMethod { FLANN_LSH, LOWE_BF, RADIUS_BF };

struct BevOrbParams
{
  int extract_count{8000};
  int uniform_cap{1500};
  int grid_cell_px{16};
  int max_per_cell{8};
  double scale_factor{1.2};
  int nlevels{4};
  int fast_threshold{20};
  BevOrbMatchMethod match_method{BevOrbMatchMethod::RADIUS_BF};
  double xiang_gao_ratio{2.0};
  double lowe_ratio{0.75};
  int radius_match_px{100};
};

struct BevOrbFeatures
{
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
};

/// ORB detect + grid-uniform keypoints + descriptors (Monocular-VO style).
BevOrbFeatures extractBevOrb(
  const cv::Mat & gray, const cv::Mat & mask, const BevOrbParams & params);

/// Match ORB descriptors between consecutive BEV frames.
std::vector<cv::DMatch> matchBevOrb(
  const BevOrbFeatures & prev, const BevOrbFeatures & curr, const BevOrbParams & params);

void selectUniformKeypointsByGrid(
  std::vector<cv::KeyPoint> & keypoints, int image_rows, int image_cols, int grid_cell_px,
  int max_per_cell, int max_total);

void removeDuplicatedTrainMatches(std::vector<cv::DMatch> & matches);

}  // namespace lie_lane_detection
