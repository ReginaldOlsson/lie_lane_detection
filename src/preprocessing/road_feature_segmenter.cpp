#include "lie_lane_detection/preprocessing/road_feature_segmenter.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <chrono>
#include <cmath>

namespace lie_lane_detection
{

namespace
{

constexpr int kNumClasses = 4;

cv::Mat resizeProbMap(const cv::Mat & src, const cv::Size & dst_size)
{
  if (src.empty()) {
    return cv::Mat();
  }
  cv::Mat out;
  cv::resize(src, out, dst_size, 0, 0, cv::INTER_LINEAR);
  return out;
}

cv::Mat blobFromBev(const cv::Mat & bev_bgr, int input_w, int input_h)
{
  cv::Mat resized;
  cv::resize(bev_bgr, resized, cv::Size(input_w, input_h));
  cv::Mat float_bgr;
  resized.convertTo(float_bgr, CV_32F, 1.0 / 255.0);
  return cv::dnn::blobFromImage(
    float_bgr, 1.0, cv::Size(input_w, input_h), cv::Scalar(), true, false, CV_32F);
}

}  // namespace

bool RoadFeatureSegmenter::load(const std::string & onnx_path)
{
  ready_ = false;
  if (onnx_path.empty()) {
    return false;
  }
  try {
    net_ = cv::dnn::readNetFromONNX(onnx_path);
  } catch (const cv::Exception & ex) {
    fprintf(
      stderr, "RoadFeatureSegmenter: failed to load ONNX %s: %s\n", onnx_path.c_str(), ex.what());
    return false;
  }
  if (net_.empty()) {
    return false;
  }
  net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
  net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
  ready_ = true;
  return true;
}

void RoadFeatureSegmenter::setInputSize(int width, int height)
{
  input_w_ = std::max(32, width);
  input_h_ = std::max(32, height);
}

void RoadFeatureSegmenter::updateParams(const PipelineParams & params)
{
  setInputSize(params.road_segmenter_input_width, params.road_segmenter_input_height);
  road_threshold_ = params.road_segmenter_road_threshold;
  lane_threshold_ = params.road_segmenter_lane_threshold;
  snow_reject_threshold_ = params.road_segmenter_snow_reject_threshold;
}

RoadFeatureSegmenter::Result RoadFeatureSegmenter::infer(const cv::Mat & bev_bgr)
{
  Result result;
  if (!ready_ || bev_bgr.empty()) {
    return result;
  }

  const auto t0 = std::chrono::steady_clock::now();
  const cv::Mat blob = blobFromBev(bev_bgr, input_w_, input_h_);
  net_.setInput(blob);
  const cv::Mat logits = net_.forward();
  if (logits.dims != 4 || logits.size[0] != 1 || logits.size[1] < kNumClasses) {
    fprintf(stderr, "RoadFeatureSegmenter: unexpected output shape\n");
    return result;
  }

  const int channels = std::min(kNumClasses, logits.size[1]);
  const int rows = logits.size[2];
  const int cols = logits.size[3];
  std::array<cv::Mat, kNumClasses> ch;
  for (int c = 0; c < kNumClasses; ++c) {
    ch[static_cast<size_t>(c)] = cv::Mat::zeros(rows, cols, CV_32F);
  }
  for (int y = 0; y < rows; ++y) {
    for (int x = 0; x < cols; ++x) {
      float max_v = logits.ptr<float>(0, 0)[y * cols + x];
      for (int c = 1; c < channels; ++c) {
        max_v = std::max(max_v, logits.ptr<float>(0, c)[y * cols + x]);
      }
      float sum = 0.0f;
      for (int c = 0; c < channels; ++c) {
        const float e = std::exp(logits.ptr<float>(0, c)[y * cols + x] - max_v);
        ch[static_cast<size_t>(c)].at<float>(y, x) = e;
        sum += e;
      }
      if (sum > 1e-9f) {
        for (int c = 0; c < channels; ++c) {
          ch[static_cast<size_t>(c)].at<float>(y, x) /= sum;
        }
      }
    }
  }

  const cv::Size bev_size(bev_bgr.cols, bev_bgr.rows);
  result.ignore_prob = resizeProbMap(ch[0], bev_size);
  result.road_prob = resizeProbMap(ch[1], bev_size);
  result.lane_prob = resizeProbMap(ch[2], bev_size);
  result.snow_prob = resizeProbMap(ch[3], bev_size);

  result.drivable_mask = cv::Mat::zeros(bev_size, CV_8U);
  result.debug_bgr = cv::Mat::zeros(bev_size, CV_8UC3);
  for (int y = 0; y < bev_size.height; ++y) {
    for (int x = 0; x < bev_size.width; ++x) {
      const float p_road = result.road_prob.at<float>(y, x);
      const float p_lane = result.lane_prob.at<float>(y, x);
      const float p_snow = result.snow_prob.at<float>(y, x);
      const bool snow = p_snow >= static_cast<float>(snow_reject_threshold_);
      const bool roadish = p_road >= static_cast<float>(road_threshold_) ||
                           p_lane >= static_cast<float>(lane_threshold_);
      if (roadish && !snow) {
        result.drivable_mask.at<uchar>(y, x) = 255;
      }
      if (p_road > 0.25f) {
        result.debug_bgr.at<cv::Vec3b>(y, x)[1] =
          static_cast<uchar>(std::min(255.0f, p_road * 255.0f));
      }
      if (p_lane > 0.20f) {
        result.debug_bgr.at<cv::Vec3b>(y, x)[0] =
          static_cast<uchar>(std::min(255.0f, p_lane * 220.0f));
        result.debug_bgr.at<cv::Vec3b>(y, x)[1] = static_cast<uchar>(std::max(
          result.debug_bgr.at<cv::Vec3b>(y, x)[1],
          static_cast<uchar>(std::min(255.0f, p_lane * 180.0f))));
      }
      if (p_snow > 0.20f) {
        result.debug_bgr.at<cv::Vec3b>(y, x)[2] =
          static_cast<uchar>(std::min(255.0f, p_snow * 255.0f));
      }
    }
  }

  result.infer_ms =
    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  return result;
}

void applyRoadFeatureMask(cv::Mat & bev_bgr, const cv::Mat & drivable_mask)
{
  if (bev_bgr.empty() || drivable_mask.empty()) {
    return;
  }
  if (drivable_mask.size() != bev_bgr.size()) {
    cv::Mat mask_resized;
    cv::resize(drivable_mask, mask_resized, bev_bgr.size(), 0, 0, cv::INTER_NEAREST);
    bev_bgr.setTo(cv::Scalar(0, 0, 0), mask_resized < 128);
    return;
  }
  bev_bgr.setTo(cv::Scalar(0, 0, 0), drivable_mask < 128);
}

cv::Mat maskEdgesWithRoadFeatures(const cv::Mat & edges_gray, const cv::Mat & drivable_mask)
{
  if (edges_gray.empty()) {
    return edges_gray;
  }
  cv::Mat edges = edges_gray.clone();
  if (drivable_mask.empty()) {
    return edges;
  }
  cv::Mat mask = drivable_mask;
  if (mask.size() != edges.size()) {
    cv::resize(drivable_mask, mask, edges.size(), 0, 0, cv::INTER_NEAREST);
  }
  edges.setTo(0, mask < 128);
  return edges;
}

}  // namespace lie_lane_detection
