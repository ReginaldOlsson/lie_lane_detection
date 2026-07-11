#pragma once

#include <algorithm>
#include <iostream>
#include <string>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

namespace lie_lane_detection
{
namespace offline_display
{

inline cv::Mat fitForScreen(const cv::Mat & img, int max_w = 1280, int max_h = 720)
{
  if (img.empty()) {
    return img;
  }
  const double sx = static_cast<double>(max_w) / static_cast<double>(img.cols);
  const double sy = static_cast<double>(max_h) / static_cast<double>(img.rows);
  const double scale = std::min(1.0, std::min(sx, sy));
  if (scale >= 0.999) {
    return img;
  }
  cv::Mat out;
  // Thin edge maps need nearest-neighbor so 1px lines stay visible when scaled.
  const int interp = (img.channels() == 1) ? cv::INTER_NEAREST : cv::INTER_AREA;
  cv::resize(img, out, cv::Size(), scale, scale, interp);
  return out;
}

/// Composite binary/thin edge map on a dimmed BEV for human viewing (imshow / PNG).
inline cv::Mat composeEdgeView(const cv::Mat & bev_bgr, const cv::Mat & edges_gray)
{
  if (bev_bgr.empty() && edges_gray.empty()) {
    return {};
  }
  if (bev_bgr.empty()) {
    cv::Mat out;
    cv::cvtColor(edges_gray, out, cv::COLOR_GRAY2BGR);
    return out;
  }

  cv::Mat base;
  if (bev_bgr.channels() == 1) {
    cv::cvtColor(bev_bgr, base, cv::COLOR_GRAY2BGR);
  } else {
    base = bev_bgr.clone();
  }
  base.convertTo(base, -1, 0.55, 0);

  if (!edges_gray.empty()) {
    cv::Mat edge_gray = edges_gray;
    if (edges_gray.channels() > 1) {
      cv::cvtColor(edges_gray, edge_gray, cv::COLOR_BGR2GRAY);
    }
    if (edge_gray.size() != base.size()) {
      cv::resize(edge_gray, edge_gray, base.size(), 0, 0, cv::INTER_NEAREST);
    }
    cv::Mat thick;
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::dilate(edge_gray, thick, kernel);
    base.setTo(cv::Scalar(0, 255, 255), thick > 0);
  }
  return base;
}

/// Lane edges on black background (artifact-filtered debug view).
inline cv::Mat composeCleanEdgeView(const cv::Mat & edges_gray)
{
  if (edges_gray.empty()) {
    return cv::Mat();
  }
  cv::Mat gray = edges_gray;
  if (edges_gray.channels() > 1) {
    cv::cvtColor(edges_gray, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Mat out = cv::Mat::zeros(gray.size(), CV_8UC3);
  out.setTo(cv::Scalar(0, 255, 255), gray > 0);
  return out;
}

inline void show(const std::string & window, const cv::Mat & img)
{
  if (img.empty()) {
    return;
  }
  cv::namedWindow(window, cv::WINDOW_NORMAL);
  cv::imshow(window, fitForScreen(img));
}

/// Wait for a key. Returns false if user pressed q/Esc (abort).
inline bool wait(int delay_ms = 0)
{
  const int key = cv::waitKey(delay_ms);
  if (key == 'q' || key == 'Q' || key == 27) {
    return false;
  }
  return true;
}

inline void destroyAll()
{
  cv::destroyAllWindows();
}

}  // namespace offline_display
}  // namespace lie_lane_detection
