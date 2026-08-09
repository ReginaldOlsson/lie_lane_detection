#include "lie_lane_detection/pipeline/intensity_strip_lane_detector.hpp"

#include "lie_lane_detection/common/parallel.hpp"
#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <sstream>

namespace lie_lane_detection
{
namespace
{

cv::Mat toGray(const cv::Mat & bgr)
{
  if (bgr.empty()) {
    return {};
  }
  if (bgr.channels() == 1) {
    return bgr.clone();
  }
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  return gray;
}

std::vector<float> columnProjection(const cv::Mat & gray_strip, int min_road_gray)
{
  std::vector<float> proj(static_cast<size_t>(gray_strip.cols), 0.f);
  std::vector<int> counts(static_cast<size_t>(gray_strip.cols), 0);
  const uchar min_val = static_cast<uchar>(std::clamp(min_road_gray, 0, 255));

  for (int y = 0; y < gray_strip.rows; ++y) {
    const uchar * row = gray_strip.ptr<uchar>(y);
    for (int x = 0; x < gray_strip.cols; ++x) {
      if (row[x] < min_val) {
        continue;
      }
      proj[static_cast<size_t>(x)] += static_cast<float>(row[x]);
      counts[static_cast<size_t>(x)] += 1;
    }
  }

  for (size_t x = 0; x < proj.size(); ++x) {
    if (counts[x] > 0) {
      proj[x] /= static_cast<float>(counts[x]);
    }
  }
  return proj;
}

void smoothProjection(std::vector<float> & proj, int ksize)
{
  if (ksize < 3 || proj.empty()) {
    return;
  }
  const int k = ksize | 1;
  cv::Mat src(1, static_cast<int>(proj.size()), CV_32F);
  std::memcpy(src.ptr<float>(), proj.data(), proj.size() * sizeof(float));
  cv::Mat dst;
  cv::GaussianBlur(src, dst, cv::Size(k, 1), 0.0);
  std::memcpy(proj.data(), dst.ptr<float>(), proj.size() * sizeof(float));
}

std::vector<int> findProjectionPeaks(
  const std::vector<float> & proj, double min_height_ratio, int min_separation_px, int max_peaks)
{
  if (proj.empty()) {
    return {};
  }

  const float max_val = *std::max_element(proj.begin(), proj.end());
  if (max_val <= 1.f) {
    return {};
  }
  const float min_height = static_cast<float>(min_height_ratio) * max_val;

  std::vector<std::pair<float, int>> candidates;
  for (int x = 1; x + 1 < static_cast<int>(proj.size()); ++x) {
    if (proj[static_cast<size_t>(x)] < min_height) {
      continue;
    }
    if (
      proj[static_cast<size_t>(x)] >= proj[static_cast<size_t>(x - 1)] &&
      proj[static_cast<size_t>(x)] > proj[static_cast<size_t>(x + 1)]) {
      candidates.emplace_back(proj[static_cast<size_t>(x)], x);
    }
  }

  std::sort(candidates.begin(), candidates.end(), [](const auto & a, const auto & b) {
    return a.first > b.first;
  });

  std::vector<int> peaks;
  peaks.reserve(static_cast<size_t>(max_peaks));
  for (const auto & [value, x] : candidates) {
    (void)value;
    bool too_close = false;
    for (int px : peaks) {
      if (std::abs(px - x) < min_separation_px) {
        too_close = true;
        break;
      }
    }
    if (too_close) {
      continue;
    }
    peaks.push_back(x);
    if (static_cast<int>(peaks.size()) >= max_peaks) {
      break;
    }
  }

  std::sort(peaks.begin(), peaks.end());
  return peaks;
}

cv::Rect makeVerticalStripRoi(
  int peak_x, int half_width, int img_cols, int img_rows, int y_top, int y_bottom)
{
  const int x0 = std::max(0, peak_x - half_width);
  const int x1 = std::min(img_cols, peak_x + half_width + 1);
  const int y0 = std::max(0, y_top);
  const int y1 = std::min(img_rows, y_bottom);
  return cv::Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
}

void offsetLaneToGlobal(LaneHypothesis & lane, int dx, int dy)
{
  lane.xi[0] += static_cast<double>(dx);
  lane.xi[1] += static_cast<double>(dy);
  for (auto & p : lane.polyline) {
    p.x() += dx;
    p.y() += dy;
  }
  for (auto & e : lane.supporting_edges) {
    e.x += dx;
    e.y += dy;
  }
}

void mergeLaneIfBetter(std::vector<LaneHypothesis> & lanes, LaneHypothesis lane, double min_sep)
{
  for (auto & existing : lanes) {
    if (std::abs(lane.xi[0] - existing.xi[0]) < min_sep) {
      if (lane.score > existing.score) {
        existing = std::move(lane);
      }
      return;
    }
  }
  lanes.push_back(std::move(lane));
}

struct StripAnalysis
{
  std::vector<float> projection;
  std::vector<int> peaks;
  std::vector<VerticalStripRoi> vertical_strips;
};

struct StripLaneBatch
{
  std::vector<LaneHypothesis> lanes;
};

std::vector<VerticalStripRoi> dedupeVerticalStrips(
  std::vector<VerticalStripRoi> strips, int min_separation_px)
{
  std::sort(
    strips.begin(), strips.end(), [](const VerticalStripRoi & a, const VerticalStripRoi & b) {
      return a.peak_value > b.peak_value;
    });

  std::vector<VerticalStripRoi> kept;
  kept.reserve(strips.size());
  for (auto strip : strips) {
    bool too_close = false;
    for (const auto & other : kept) {
      if (std::abs(strip.peak_x - other.peak_x) < min_separation_px) {
        too_close = true;
        break;
      }
    }
    if (!too_close) {
      kept.push_back(strip);
    }
  }
  std::sort(kept.begin(), kept.end(), [](const VerticalStripRoi & a, const VerticalStripRoi & b) {
    return a.peak_x < b.peak_x;
  });
  return kept;
}

cv::Mat renderProjectionDebug(
  const cv::Mat & gray, const std::vector<std::vector<float>> & projections,
  const std::vector<std::vector<int>> & peaks_per_strip, int num_strips)
{
  if (gray.empty() || projections.empty()) {
    return {};
  }

  const int panel_h = std::max(48, gray.rows / num_strips);
  (void)panel_h;
  cv::Mat debug(gray.rows, gray.cols, CV_8UC3, cv::Scalar(20, 20, 20));

  for (int s = 0; s < static_cast<int>(projections.size()); ++s) {
    const int y0 = s * gray.rows / num_strips;
    const int y1 = (s + 1) * gray.rows / num_strips;
    const int h = y1 - y0;
    const auto & proj = projections[static_cast<size_t>(s)];
    if (proj.empty()) {
      continue;
    }

    const float max_val = *std::max_element(proj.begin(), proj.end());
    if (max_val <= 1.f) {
      continue;
    }

    for (int x = 0; x < static_cast<int>(proj.size()); ++x) {
      const int bar_h = static_cast<int>((proj[static_cast<size_t>(x)] / max_val) * (h - 4));
      const int y_base = y0 + h - 2;
      cv::line(
        debug, cv::Point(x, y_base), cv::Point(x, y_base - bar_h), cv::Scalar(80, 200, 255), 1);
    }

    for (int px : peaks_per_strip[static_cast<size_t>(s)]) {
      cv::line(debug, cv::Point(px, y0), cv::Point(px, y1), cv::Scalar(0, 255, 120), 1);
    }

    cv::line(
      debug, cv::Point(0, y1 - 1), cv::Point(gray.cols - 1, y1 - 1), cv::Scalar(0, 180, 255), 1);
    cv::putText(
      debug, "S" + std::to_string(s), cv::Point(4, y0 + 14), cv::FONT_HERSHEY_SIMPLEX, 0.4,
      cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
  }
  return debug;
}

}  // namespace

IntensityStripParams loadIntensityStripParams(rclcpp::Node & node)
{
  IntensityStripParams p;
  p.num_horizontal_strips = node.declare_parameter<int>("num_horizontal_strips", 12);
  p.max_peaks_per_strip = node.declare_parameter<int>("max_peaks_per_strip", 4);
  p.min_peak_height_ratio = node.declare_parameter<double>("min_peak_height_ratio", 0.32);
  p.min_peak_separation_px = node.declare_parameter<int>("min_peak_separation_px", 42);
  p.vertical_strip_half_width_px = node.declare_parameter<int>("vertical_strip_half_width_px", 52);
  p.projection_smooth_ksize = node.declare_parameter<int>("projection_smooth_ksize", 9);
  p.min_road_gray = node.declare_parameter<int>("min_road_gray", 20);
  if (node.has_parameter("bev_bottom_exclude_px")) {
    p.bev_bottom_exclude_px = node.get_parameter("bev_bottom_exclude_px").as_double();
  } else {
    p.bev_bottom_exclude_px = node.declare_parameter<double>("bev_bottom_exclude_px", 200.0);
  }
  p.dedupe_vertical_rois = node.declare_parameter<bool>("dedupe_vertical_rois", true);
  p.min_lane_merge_px = node.declare_parameter<double>("min_lane_merge_px", 38.0);
  return p;
}

IntensityStripDetectionResult detectLanesIntensityStrips(
  const cv::Mat & bev_bgr, const PipelineParams & detect_params,
  const IntensityStripParams & strip_params)
{
  const auto t0 = std::chrono::steady_clock::now();
  IntensityStripDetectionResult result;
  if (bev_bgr.empty()) {
    return result;
  }

  PipelineParams prep_params = detect_params;
  prep_params.bev_bottom_exclude_px = strip_params.bev_bottom_exclude_px;

  const BevPreprocessResult prep = preprocessBevForLaneDetection(bev_bgr, prep_params);
  const cv::Mat gray = toGray(prep.detect_image.empty() ? prep.display_bgr : prep.detect_image);
  if (gray.empty()) {
    return result;
  }

  const int rows = gray.rows;
  const int cols = gray.cols;
  const int num_strips = std::max(1, strip_params.num_horizontal_strips);
  const int y_max = static_cast<int>(bevEffectiveYMax(rows, prep_params));

  result.horizontal_strip_count = num_strips;
  std::vector<StripAnalysis> strip_analyses(static_cast<size_t>(num_strips));

  tbb::parallel_for(0, num_strips, [&](int s) {
    const int y0 = s * rows / num_strips;
    const int y1 = (s + 1) * rows / num_strips;
    const cv::Mat band = gray.rowRange(y0, y1);

    StripAnalysis analysis;
    analysis.projection = columnProjection(band, strip_params.min_road_gray);
    smoothProjection(analysis.projection, strip_params.projection_smooth_ksize);
    analysis.peaks = findProjectionPeaks(
      analysis.projection, strip_params.min_peak_height_ratio, strip_params.min_peak_separation_px,
      strip_params.max_peaks_per_strip);

    int peak_idx = 0;
    for (int px : analysis.peaks) {
      VerticalStripRoi strip;
      strip.strip_index = s;
      strip.peak_index = peak_idx++;
      strip.peak_x = static_cast<double>(px);
      strip.peak_value = analysis.projection[static_cast<size_t>(px)];
      strip.roi =
        makeVerticalStripRoi(px, strip_params.vertical_strip_half_width_px, cols, y_max, 0, y_max);
      analysis.vertical_strips.push_back(strip);
    }
    strip_analyses[static_cast<size_t>(s)] = std::move(analysis);
  });

  std::vector<std::vector<float>> projections(static_cast<size_t>(num_strips));
  std::vector<std::vector<int>> peaks_per_strip(static_cast<size_t>(num_strips));
  std::vector<VerticalStripRoi> vertical_strips;
  vertical_strips.reserve(static_cast<size_t>(num_strips * strip_params.max_peaks_per_strip));

  for (int s = 0; s < num_strips; ++s) {
    projections[static_cast<size_t>(s)] =
      std::move(strip_analyses[static_cast<size_t>(s)].projection);
    peaks_per_strip[static_cast<size_t>(s)] =
      std::move(strip_analyses[static_cast<size_t>(s)].peaks);
    for (auto & strip : strip_analyses[static_cast<size_t>(s)].vertical_strips) {
      vertical_strips.push_back(std::move(strip));
    }
  }

  if (strip_params.dedupe_vertical_rois) {
    vertical_strips = dedupeVerticalStrips(vertical_strips, strip_params.min_peak_separation_px);
  }
  result.vertical_strips = vertical_strips;

  const size_t num_vertical = vertical_strips.size();
  std::vector<StripLaneBatch> lane_batches(num_vertical);

  tbb::parallel_for(
    tbb::blocked_range<size_t>(0, num_vertical), [&](const tbb::blocked_range<size_t> & range) {
      for (size_t i = range.begin(); i != range.end(); ++i) {
        const auto & strip = vertical_strips[i];
        cv::Mat roi_bgr = prep.display_bgr(strip.roi).clone();
        if (roi_bgr.empty()) {
          continue;
        }

        PipelineParams roi_params = detect_params;
        configureParamsForBev(roi_params, roi_bgr.cols, roi_bgr.rows);
        configureParamsForPerspectiveIpm(roi_params);
        roi_params.bev_bottom_exclude_px = 0.0;
        roi_params.max_output_lanes = 1;

        const BevDetectionResult det = detectLanesInBev(roi_bgr, roi_params);
        StripLaneBatch batch;
        batch.lanes.reserve(det.lanes.size());
        for (auto lane : det.lanes) {
          offsetLaneToGlobal(lane, strip.roi.x, strip.roi.y);
          batch.lanes.push_back(std::move(lane));
        }
        lane_batches[i] = std::move(batch);
      }
    });

  std::vector<LaneHypothesis> merged_lanes;
  uint32_t lane_id = 1;
  for (size_t i = 0; i < num_vertical; ++i) {
    for (auto & lane : lane_batches[i].lanes) {
      lane.lane_id = lane_id++;
      mergeLaneIfBetter(merged_lanes, std::move(lane), strip_params.min_lane_merge_px);
    }
  }

  result.lanes = std::move(merged_lanes);
  result.projection_debug = renderProjectionDebug(gray, projections, peaks_per_strip, num_strips);
  result.overlay = drawIntensityStripOverlay(prep.display_bgr, result, strip_params);

  const auto t1 = std::chrono::steady_clock::now();
  result.elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  return result;
}

cv::Mat drawIntensityStripOverlay(
  const cv::Mat & bev_bgr, const IntensityStripDetectionResult & result,
  const IntensityStripParams & strip_params)
{
  cv::Mat base;
  if (bev_bgr.channels() == 1) {
    cv::cvtColor(bev_bgr, base, cv::COLOR_GRAY2BGR);
  } else {
    base = bev_bgr.clone();
  }

  const int rows = base.rows;
  const int num_strips = std::max(1, result.horizontal_strip_count);

  for (int s = 1; s < num_strips; ++s) {
    const int y = s * rows / num_strips;
    cv::line(
      base, cv::Point(0, y), cv::Point(base.cols - 1, y), cv::Scalar(255, 220, 0), 1, cv::LINE_AA);
  }

  for (const auto & strip : result.vertical_strips) {
    cv::rectangle(base, strip.roi, cv::Scalar(0, 220, 255), 1, cv::LINE_AA);
    const std::string label =
      "S" + std::to_string(strip.strip_index) + "P" + std::to_string(strip.peak_index);
    cv::putText(
      base, label, cv::Point(strip.roi.x + 2, std::max(12, strip.roi.y + 12)),
      cv::FONT_HERSHEY_SIMPLEX, 0.35, cv::Scalar(0, 255, 255), 1, cv::LINE_AA);
  }

  cv::Mat with_lanes = drawOverlay(base, result.lanes, result.merges);

  std::ostringstream banner;
  banner << "H=" << num_strips << " V=" << result.vertical_strips.size()
         << " lanes=" << result.lanes.size();
  cv::putText(
    with_lanes, banner.str(), cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  cv::putText(
    with_lanes, banner.str(), cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
    cv::Scalar(40, 40, 200), 1, cv::LINE_AA);

  (void)strip_params;
  return with_lanes;
}

cv::Mat drawFrontalIntensityStripOverlay(
  const cv::Mat & frontal_bgr, const IntensityStripDetectionResult & result,
  const IntensityStripParams & strip_params, const cv::Mat & H_img2bev)
{
  cv::Mat overlay = drawFrontalOverlay(frontal_bgr, result.lanes, result.merges, H_img2bev);
  if (H_img2bev.empty() || H_img2bev.rows != 3 || H_img2bev.cols != 3) {
    return overlay;
  }

  cv::Mat H_bev2img;
  cv::invert(H_img2bev, H_bev2img, cv::DECOMP_LU);

  const int cols = overlay.cols;
  const int rows = overlay.rows;
  const int num_strips = std::max(1, result.horizontal_strip_count);

  int bev_width = 0;
  int bev_height = 0;
  for (const auto & strip : result.vertical_strips) {
    bev_width = std::max(bev_width, strip.roi.x + strip.roi.width);
    bev_height = std::max(bev_height, strip.roi.y + strip.roi.height);
  }
  for (const auto & lane : result.lanes) {
    for (const auto & p : lane.polyline) {
      bev_width = std::max(bev_width, static_cast<int>(std::ceil(p.x())) + 1);
      bev_height = std::max(bev_height, static_cast<int>(std::ceil(p.y())) + 1);
    }
  }
  if (bev_width <= 0) {
    bev_width = overlay.cols;
  }
  if (bev_height <= 0) {
    bev_height = overlay.rows;
  }

  auto project = [&](double bx, double by) -> cv::Point2f {
    const double w = H_bev2img.at<double>(2, 0) * bx + H_bev2img.at<double>(2, 1) * by +
                     H_bev2img.at<double>(2, 2);
    if (std::abs(w) < 1e-9) {
      return {-1.f, -1.f};
    }
    const double inv_w = 1.0 / w;
    return cv::Point2f(
      static_cast<float>(
        (H_bev2img.at<double>(0, 0) * bx + H_bev2img.at<double>(0, 1) * by +
         H_bev2img.at<double>(0, 2)) *
        inv_w),
      static_cast<float>(
        (H_bev2img.at<double>(1, 0) * bx + H_bev2img.at<double>(1, 1) * by +
         H_bev2img.at<double>(1, 2)) *
        inv_w));
  };

  auto draw_bev_segment = [&](
                            double x0, double y0, double x1, double y1, const cv::Scalar & color) {
    const cv::Point2f p0f = project(x0, y0);
    const cv::Point2f p1f = project(x1, y1);
    const cv::Point p0(static_cast<int>(p0f.x), static_cast<int>(p0f.y));
    const cv::Point p1(static_cast<int>(p1f.x), static_cast<int>(p1f.y));
    if (
      p0.x >= -50 && p0.x < cols + 50 && p0.y >= -50 && p0.y < rows + 50 && p1.x >= -50 &&
      p1.x < cols + 50 && p1.y >= -50 && p1.y < rows + 50) {
      cv::line(overlay, p0, p1, color, 1, cv::LINE_AA);
    }
  };

  for (int s = 1; s < num_strips; ++s) {
    const double y = static_cast<double>(s * bev_height / num_strips);
    draw_bev_segment(0.0, y, static_cast<double>(bev_width), y, cv::Scalar(255, 220, 0));
  }

  for (const auto & strip : result.vertical_strips) {
    const cv::Rect & r = strip.roi;
    const double x0 = static_cast<double>(r.x);
    const double x1 = static_cast<double>(r.x + r.width);
    const double y0 = static_cast<double>(r.y);
    const double y1 = static_cast<double>(r.y + r.height);
    draw_bev_segment(x0, y0, x1, y0, cv::Scalar(0, 220, 255));
    draw_bev_segment(x1, y0, x1, y1, cv::Scalar(0, 220, 255));
    draw_bev_segment(x1, y1, x0, y1, cv::Scalar(0, 220, 255));
    draw_bev_segment(x0, y1, x0, y0, cv::Scalar(0, 220, 255));
  }

  std::ostringstream banner;
  banner << "H=" << num_strips << " V=" << result.vertical_strips.size()
         << " lanes=" << result.lanes.size();
  cv::putText(
    overlay, banner.str(), cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
    cv::Scalar(255, 255, 255), 2, cv::LINE_AA);
  cv::putText(
    overlay, banner.str(), cv::Point(8, 22), cv::FONT_HERSHEY_SIMPLEX, 0.55,
    cv::Scalar(40, 40, 200), 1, cv::LINE_AA);

  (void)strip_params;
  return overlay;
}

}  // namespace lie_lane_detection
