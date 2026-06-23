#include "lie_lane_detection/voting/line_hough_extractor.hpp"

#include <cmath>

#include <opencv2/imgproc.hpp>

namespace lie_lane_detection
{

LineHoughExtractor::LineHoughExtractor(const PipelineParams & params)
: params_(params)
{
}

std::vector<LineSegment> LineHoughExtractor::extract(const cv::Mat & edge_image) const
{
  std::vector<LineSegment> segments;
  if (edge_image.empty()) {
    return segments;
  }

  cv::Mat edges = edge_image;
  if (edges.channels() != 1) {
    cv::cvtColor(edge_image, edges, cv::COLOR_BGR2GRAY);
  }

  std::vector<cv::Vec4i> lines;
  cv::HoughLinesP(
    edges,
    lines,
    1.0,
    CV_PI / 180.0,
    params_.line_hough_threshold,
    params_.line_min_length_px,
    params_.line_max_gap_px);

  segments.reserve(lines.size());
  for (const auto & ln : lines) {
    const double x1 = static_cast<double>(ln[0]);
    const double y1 = static_cast<double>(ln[1]);
    const double x2 = static_cast<double>(ln[2]);
    const double y2 = static_cast<double>(ln[3]);
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    const double length = std::hypot(dx, dy);
    if (length < params_.line_min_length_px) {
      continue;
    }

    LineSegment seg;
    seg.x1 = x1;
    seg.y1 = y1;
    seg.x2 = x2;
    seg.y2 = y2;
    seg.length = length;
    seg.angle = std::atan2(dy, dx);
    seg.mx = 0.5 * (x1 + x2);
    seg.my = 0.5 * (y1 + y2);
    segments.push_back(seg);
  }
  return segments;
}

cv::Mat LineHoughExtractor::drawSegments(const cv::Mat & bev_bgr, const std::vector<LineSegment> & lines)
{
  cv::Mat out = bev_bgr.clone();
  for (const auto & line : lines) {
    cv::line(
      out,
      cv::Point(static_cast<int>(line.x1), static_cast<int>(line.y1)),
      cv::Point(static_cast<int>(line.x2), static_cast<int>(line.y2)),
      cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
  }
  return out;
}

}  // namespace lie_lane_detection
