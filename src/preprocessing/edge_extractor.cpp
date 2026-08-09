#include "lie_lane_detection/preprocessing/edge_extractor.hpp"

#include "lie_lane_detection/common/parallel.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <vector>

namespace lie_lane_detection
{

EdgeExtractor::EdgeExtractor(const PipelineParams & params) : params_(params)
{
}

cv::Mat EdgeExtractor::applySteerableBank(const cv::Mat & gray, cv::Mat * orientation) const
{
  cv::Mat gx;
  cv::Mat gy;
  cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
  cv::Sobel(gray, gy, CV_32F, 0, 1, 3);

  constexpr int kNumOrientations = 5;
  std::vector<cv::Mat> oriented(static_cast<size_t>(kNumOrientations));
  tbb::parallel_for(0, kNumOrientations, [&](int o) {
    const double theta = CV_PI * static_cast<double>(o) / static_cast<double>(kNumOrientations);
    const float ct = static_cast<float>(std::cos(theta));
    const float st = static_cast<float>(std::sin(theta));
    cv::Mat directed = ct * gx + st * gy;
    oriented[static_cast<size_t>(o)] = cv::abs(directed);
  });

  cv::Mat response = oriented[0].clone();
  for (int o = 1; o < kNumOrientations; ++o) {
    cv::max(response, oriented[static_cast<size_t>(o)], response);
  }

  if (orientation) {
    cv::cartToPolar(gx, gy, *orientation, *orientation, true);
  }
  return response;
}

double EdgeExtractor::otsuThresholdMasked(
  const cv::Mat & gray, const cv::Mat & road_mask, cv::Mat & binary_out)
{
  if (gray.empty() || road_mask.empty() || gray.size() != road_mask.size()) {
    binary_out = gray.empty() ? cv::Mat{} : gray.clone();
    return -1.0;
  }

  cv::Mat otsu_input = gray.clone();
  otsu_input.setTo(0, ~road_mask);
  const double threshold =
    cv::threshold(otsu_input, binary_out, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  binary_out.setTo(0, ~road_mask);
  return threshold;
}

void EdgeExtractor::thinBinaryEdges(cv::Mat & edges) const
{
  cv::Mat skel = cv::Mat::zeros(edges.size(), CV_8U);
  cv::Mat img = edges.clone();
  const cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
  cv::Mat temp;
  cv::Mat eroded;
  while (cv::countNonZero(img) > 0) {
    cv::erode(img, eroded, element);
    cv::subtract(img, eroded, temp);
    cv::bitwise_or(skel, temp, skel);
    img = eroded;
  }
  edges = skel;
}

void EdgeExtractor::hysteresisThreshold(cv::Mat & magnitude, cv::Mat & /*orientation*/) const
{
  cv::Mat strong;
  cv::Mat weak;
  cv::threshold(magnitude, strong, params_.edge_high_threshold, 255.0, cv::THRESH_BINARY);
  cv::threshold(magnitude, weak, params_.edge_low_threshold, 255.0, cv::THRESH_BINARY);

  cv::Mat edges = cv::Mat::zeros(magnitude.size(), CV_8U);
  for (int y = 1; y < magnitude.rows - 1; ++y) {
    for (int x = 1; x < magnitude.cols - 1; ++x) {
      if (strong.at<uchar>(y, x) == 0) {
        continue;
      }
      edges.at<uchar>(y, x) = 255;
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
          if (weak.at<uchar>(y + dy, x + dx) > 0) {
            edges.at<uchar>(y + dy, x + dx) = 255;
          }
        }
      }
    }
  }
  magnitude = edges;
}

std::vector<EdgePoint> EdgeExtractor::extract(const cv::Mat & bev_bgr, cv::Mat * debug_edges)
{
  cv::Mat gray;
  if (bev_bgr.channels() == 3) {
    cv::cvtColor(bev_bgr, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = bev_bgr;
  }

  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8, 8));
  clahe->apply(gray, gray);

  // Lewis et al. (IVCNZ 2016): anisotropic blur suppresses clutter perpendicular to curves.
  // BEV lanes are ~vertical → stronger blur along x.
  if (params_.edge_anisotropic_blur) {
    cv::GaussianBlur(gray, gray, cv::Size(7, 3), 1.5, 0.8);
  }

  cv::Mat magnitude;
  cv::Mat orientation = cv::Mat::zeros(gray.size(), CV_32F);
  if (params_.use_steerable_filter) {
    magnitude = applySteerableBank(gray, &orientation);
  } else {
    cv::Mat gx;
    cv::Mat gy;
    cv::Sobel(gray, gx, CV_32F, 1, 0, 3);
    cv::Sobel(gray, gy, CV_32F, 0, 1, 3);
    cv::cartToPolar(gx, gy, magnitude, orientation, true);
  }

  cv::Mat mag_u8;
  cv::convertScaleAbs(magnitude, mag_u8);
  if (params_.connect_dashed_edges) {
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 7));
    cv::morphologyEx(mag_u8, mag_u8, cv::MORPH_CLOSE, kernel);
  }
  hysteresisThreshold(mag_u8, orientation);

  if (params_.edge_thin) {
    thinBinaryEdges(mag_u8);
  }

  if (debug_edges) {
    *debug_edges = mag_u8.clone();
  }

  std::vector<cv::Point> nonzero;
  cv::findNonZero(mag_u8, nonzero);

  const bool subpixel = params_.use_subpixel_edges && !magnitude.empty();
  const int mag_cols = magnitude.cols;

  std::vector<EdgePoint> edges(nonzero.size());
  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, nonzero.size()), [&](const tbb::blocked_range<size_t> & range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        const cv::Point & pt = nonzero[i];
        EdgePoint ep;
        ep.x = static_cast<double>(pt.x);
        ep.y = static_cast<double>(pt.y);
        ep.magnitude = static_cast<double>(mag_u8.at<uchar>(pt.y, pt.x));
        if (!orientation.empty()) {
          ep.orientation = orientation.at<float>(pt.y, pt.x);
        }
        // BEV lanes are ~vertical, so lateral (x) precision drives fit quality.
        // Refine x with a parabolic peak fit on the gradient magnitude profile
        // across the 3 horizontal neighbours.
        if (subpixel && pt.x > 0 && pt.x < mag_cols - 1) {
          const float * mrow = magnitude.ptr<float>(pt.y);
          const float ml = mrow[pt.x - 1];
          const float mc = mrow[pt.x];
          const float mr = mrow[pt.x + 1];
          const float denom = ml - 2.0f * mc + mr;
          if (denom < -1e-6f) {  // concave-down => interior peak
            float offset = 0.5f * (ml - mr) / denom;
            offset = std::clamp(offset, -1.0f, 1.0f);
            ep.x += static_cast<double>(offset);
          }
        }
        edges[i] = ep;
      }
    });

  return edges;
}

}  // namespace lie_lane_detection
