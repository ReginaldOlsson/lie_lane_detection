// ORB feature extraction and matching for BEV mosaic registration.
// Adapted from Monocular-Visual-Odometry (feature_match.cpp):
// grid-uniform ORB keypoints, FLANN-LSH / Lowe BF / radius+Hamming matchers.

#include "lie_lane_detection/mosaic/bev_orb_matcher.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

namespace lie_lane_detection
{
namespace
{

cv::Ptr<cv::ORB> makeOrbDetector(const BevOrbParams & params)
{
  return cv::ORB::create(
    params.extract_count,
    static_cast<float>(params.scale_factor),
    params.nlevels,
    31,
    0,
    2,
    cv::ORB::HARRIS_SCORE,
    31,
    params.fast_threshold);
}

std::vector<cv::DMatch> matchFlannLsh(
  const cv::Mat & desc_prev,
  const cv::Mat & desc_curr,
  double xiang_gao_ratio)
{
  if (desc_prev.empty() || desc_curr.empty()) {
    return {};
  }

  cv::FlannBasedMatcher matcher(new cv::flann::LshIndexParams(5, 10, 2));
  std::vector<cv::DMatch> all_matches;
  matcher.match(desc_prev, desc_curr, all_matches);
  if (all_matches.empty()) {
    return {};
  }

  double min_dist = std::numeric_limits<double>::max();
  for (const auto & m : all_matches) {
    min_dist = std::min(min_dist, static_cast<double>(m.distance));
  }
  const double threshold = std::max(min_dist * xiang_gao_ratio, 30.0);

  std::vector<cv::DMatch> good;
  good.reserve(all_matches.size());
  for (const auto & m : all_matches) {
    if (m.distance < threshold) {
      good.push_back(m);
    }
  }
  return good;
}

std::vector<cv::DMatch> matchLoweBf(
  const cv::Mat & desc_prev,
  const cv::Mat & desc_curr,
  double lowe_ratio)
{
  if (desc_prev.empty() || desc_curr.empty()) {
    return {};
  }

  cv::BFMatcher matcher(cv::NORM_HAMMING);
  std::vector<std::vector<cv::DMatch>> knn;
  matcher.knnMatch(desc_prev, desc_curr, knn, 2);

  std::vector<cv::DMatch> good;
  good.reserve(knn.size());
  for (const auto & pair : knn) {
    if (pair.size() < 2) {
      continue;
    }
    if (pair[0].distance < lowe_ratio * pair[1].distance) {
      good.push_back(pair[0]);
    }
  }
  return good;
}

std::vector<cv::DMatch> matchRadiusBrute(
  const BevOrbFeatures & prev,
  const BevOrbFeatures & curr,
  int radius_px,
  double xiang_gao_ratio)
{
  const int n_prev = static_cast<int>(prev.keypoints.size());
  const int n_curr = static_cast<int>(curr.keypoints.size());
  if (n_prev == 0 || n_curr == 0 || prev.descriptors.empty() || curr.descriptors.empty()) {
    return {};
  }

  const float radius_sq = static_cast<float>(radius_px * radius_px);
  std::vector<cv::DMatch> all_matches;
  all_matches.reserve(static_cast<size_t>(n_prev));

  for (int i = 0; i < n_prev; ++i) {
    const cv::Point2f p1 = prev.keypoints[static_cast<size_t>(i)].pt;
    float best_dist = std::numeric_limits<float>::max();
    int best_j = -1;

    for (int j = 0; j < n_curr; ++j) {
      const cv::Point2f p2 = curr.keypoints[static_cast<size_t>(j)].pt;
      const float dx = p1.x - p2.x;
      const float dy = p1.y - p2.y;
      if (dx * dx + dy * dy > radius_sq) {
        continue;
      }

      cv::Mat diff;
      cv::absdiff(prev.descriptors.row(i), curr.descriptors.row(j), diff);
      const float dist = static_cast<float>(cv::sum(diff)[0]) /
        static_cast<float>(prev.descriptors.cols);
      if (dist < best_dist) {
        best_dist = dist;
        best_j = j;
      }
    }

    if (best_j >= 0) {
      all_matches.emplace_back(i, best_j, best_dist);
    }
  }

  if (all_matches.empty()) {
    return {};
  }

  double min_dist = std::numeric_limits<double>::max();
  for (const auto & m : all_matches) {
    min_dist = std::min(min_dist, static_cast<double>(m.distance));
  }
  const double threshold = std::max(min_dist * xiang_gao_ratio, 30.0);

  std::vector<cv::DMatch> good;
  good.reserve(all_matches.size());
  for (const auto & m : all_matches) {
    if (m.distance < threshold) {
      good.push_back(m);
    }
  }
  return good;
}

}  // namespace

void selectUniformKeypointsByGrid(
  std::vector<cv::KeyPoint> & keypoints,
  int image_rows,
  int image_cols,
  int grid_cell_px,
  int max_per_cell,
  int max_total)
{
  if (keypoints.empty() || grid_cell_px <= 0 || max_per_cell <= 0 || max_total <= 0) {
    return;
  }

  const int rows = std::max(1, image_rows / grid_cell_px);
  const int cols = std::max(1, image_cols / grid_cell_px);
  std::vector<std::vector<int>> grid(
    static_cast<size_t>(rows),
    std::vector<int>(static_cast<size_t>(cols), 0));

  std::vector<cv::KeyPoint> selected;
  selected.reserve(static_cast<size_t>(max_total));
  for (const auto & kpt : keypoints) {
    const int row = std::clamp(static_cast<int>(kpt.pt.y) / grid_cell_px, 0, rows - 1);
    const int col = std::clamp(static_cast<int>(kpt.pt.x) / grid_cell_px, 0, cols - 1);
    if (grid[static_cast<size_t>(row)][static_cast<size_t>(col)] >= max_per_cell) {
      continue;
    }
    selected.push_back(kpt);
    grid[static_cast<size_t>(row)][static_cast<size_t>(col)] += 1;
    if (static_cast<int>(selected.size()) >= max_total) {
      break;
    }
  }
  keypoints.swap(selected);
}

void removeDuplicatedTrainMatches(std::vector<cv::DMatch> & matches)
{
  if (matches.size() < 2) {
    return;
  }
  std::sort(
    matches.begin(), matches.end(),
    [](const cv::DMatch & a, const cv::DMatch & b) {
      return a.trainIdx < b.trainIdx;
    });

  std::vector<cv::DMatch> unique;
  unique.reserve(matches.size());
  unique.push_back(matches.front());
  for (size_t i = 1; i < matches.size(); ++i) {
    if (matches[i].trainIdx != matches[i - 1].trainIdx) {
      unique.push_back(matches[i]);
    }
  }
  matches.swap(unique);
}

BevOrbFeatures extractBevOrb(
  const cv::Mat & gray,
  const cv::Mat & mask,
  const BevOrbParams & params)
{
  BevOrbFeatures out;
  if (gray.empty()) {
    return out;
  }

  cv::Ptr<cv::ORB> detector = makeOrbDetector(params);
  detector->detect(gray, out.keypoints, mask);
  selectUniformKeypointsByGrid(
    out.keypoints,
    gray.rows,
    gray.cols,
    params.grid_cell_px,
    params.max_per_cell,
    params.uniform_cap);

  if (out.keypoints.empty()) {
    return out;
  }

  cv::Ptr<cv::ORB> computer = cv::ORB::create(
    params.extract_count,
    static_cast<float>(params.scale_factor),
    params.nlevels);
  computer->compute(gray, out.keypoints, out.descriptors);
  return out;
}

std::vector<cv::DMatch> matchBevOrb(
  const BevOrbFeatures & prev,
  const BevOrbFeatures & curr,
  const BevOrbParams & params)
{
  std::vector<cv::DMatch> matches;
  switch (params.match_method) {
    case BevOrbMatchMethod::FLANN_LSH:
      matches = matchFlannLsh(prev.descriptors, curr.descriptors, params.xiang_gao_ratio);
      break;
    case BevOrbMatchMethod::LOWE_BF:
      matches = matchLoweBf(prev.descriptors, curr.descriptors, params.lowe_ratio);
      break;
    case BevOrbMatchMethod::RADIUS_BF:
      matches = matchRadiusBrute(prev, curr, params.radius_match_px, params.xiang_gao_ratio);
      break;
  }
  removeDuplicatedTrainMatches(matches);
  return matches;
}

}  // namespace lie_lane_detection
