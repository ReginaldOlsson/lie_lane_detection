#include "lie_lane_detection/mosaic/odom_bev_mosaic_accumulator.hpp"

#include <algorithm>
#include <array>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace lie_lane_detection
{
namespace
{

cv::Point2f bodyMetersToMap(const Pose2d & pose, const double lateral_m, const double forward_m)
{
  const double c = std::cos(pose.yaw_rad);
  const double s = std::sin(pose.yaw_rad);
  return cv::Point2f(
    static_cast<float>(pose.x + c * lateral_m - s * forward_m),
    static_cast<float>(pose.y + s * lateral_m + c * forward_m));
}

cv::Point2d mapMetersToCanvasPx(
  const OdomBevMosaicMeta & meta, const double map_x, const double map_y)
{
  return cv::Point2d(
    (map_x - meta.origin_map_x) / meta.meters_per_px,
    (meta.origin_map_y - map_y) / meta.meters_per_px);
}

cv::Rect boundingRectFromPoints(const std::vector<cv::Point2f> & points)
{
  if (points.empty()) {
    return {};
  }
  float min_x = points[0].x;
  float max_x = points[0].x;
  float min_y = points[0].y;
  float max_y = points[0].y;
  for (const auto & p : points) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
  }
  return cv::Rect(
    static_cast<int>(std::floor(min_x)), static_cast<int>(std::floor(min_y)),
    static_cast<int>(std::ceil(max_x - min_x)) + 1,
    static_cast<int>(std::ceil(max_y - min_y)) + 1);
}

}  // namespace

OdomBevMosaicAccumulator::OdomBevMosaicAccumulator(OdomBevMosaicParams params)
: params_(std::move(params))
{
  reset();
}

void OdomBevMosaicAccumulator::reset()
{
  canvas_.release();
  weight_map_.release();
  meta_ = OdomBevMosaicMeta{};
  meta_.meters_per_px = params_.meters_per_px;
  initialized_ = false;
}

cv::Point2d OdomBevMosaicAccumulator::mapBodyToCanvasPx(
  const double lateral_m, const double forward_m) const
{
  const cv::Point2d px(
    (lateral_m / meta_.meters_per_px) + params_.bev_width_px * 0.5,
    params_.bev_height_px - (forward_m / meta_.meters_per_px));
  return px;
}

cv::Mat OdomBevMosaicAccumulator::computeBevToCanvasAffine(const Pose2d & pose_map) const
{
  Pose2d pose = pose_map;
  if (params_.pose_lateral_offset_m != 0.0 || params_.pose_forward_offset_m != 0.0 ||
    params_.pose_yaw_offset_rad != 0.0)
  {
    Pose2d offset;
    offset.x = params_.pose_lateral_offset_m;
    offset.y = params_.pose_forward_offset_m;
    offset.yaw_rad = params_.pose_yaw_offset_rad;
    pose = pose.compose(offset);
  }

  const double mpp = meta_.meters_per_px;
  const float bev_cx = static_cast<float>(params_.bev_width_px) * 0.5f;
  const float bev_h = static_cast<float>(params_.bev_height_px);

  const std::array<cv::Point2f, 3> bev_pts = {
    cv::Point2f(bev_cx, bev_h),
    cv::Point2f(bev_cx, bev_h - static_cast<float>(1.0 / mpp)),
    cv::Point2f(bev_cx + static_cast<float>(1.0 / mpp), bev_h),
  };

  std::array<cv::Point2f, 3> canvas_pts;
  const std::array<std::pair<double, double>, 3> body_m = {{
    {0.0, 0.0},
    {0.0, 1.0},
    {1.0, 0.0},
  }};
  for (size_t i = 0; i < bev_pts.size(); ++i) {
    const cv::Point2f map_pt = bodyMetersToMap(pose, body_m[i].first, body_m[i].second);
    const cv::Point2d canvas = mapMetersToCanvasPx(meta_, map_pt.x, map_pt.y);
    canvas_pts[i] = cv::Point2f(static_cast<float>(canvas.x), static_cast<float>(canvas.y));
  }

  return cv::getAffineTransform(bev_pts.data(), canvas_pts.data());
}

cv::Mat OdomBevMosaicAccumulator::buildValidityMask(const cv::Mat & bev_bgr) const
{
  cv::Mat mask(bev_bgr.rows, bev_bgr.cols, CV_8U, cv::Scalar(255));
  cv::Mat gray;
  if (bev_bgr.channels() == 1) {
    gray = bev_bgr;
  } else {
    cv::cvtColor(bev_bgr, gray, cv::COLOR_BGR2GRAY);
  }
  cv::Mat valid = gray > params_.mask_gray_threshold;
  cv::bitwise_and(mask, valid, mask);
  return mask;
}

void OdomBevMosaicAccumulator::ensureCanvasContains(const cv::Rect & required_px)
{
  if (required_px.width <= 0 || required_px.height <= 0) {
    return;
  }

  const int pad = static_cast<int>(std::ceil(params_.canvas_padding_m / meta_.meters_per_px));
  int req_x0 = required_px.x - pad;
  int req_y0 = required_px.y - pad;
  int req_x1 = required_px.x + required_px.width + pad;
  int req_y1 = required_px.y + required_px.height + pad;

  if (!initialized_) {
    if (req_x0 != 0 || req_y0 != 0) {
      meta_.origin_map_x -= static_cast<double>(req_x0) * meta_.meters_per_px;
      meta_.origin_map_y += static_cast<double>(req_y0) * meta_.meters_per_px;
    }
    canvas_ = cv::Mat(req_y1 - req_y0, req_x1 - req_x0, CV_8UC3, cv::Scalar(0, 0, 0));
    weight_map_ = cv::Mat(req_y1 - req_y0, req_x1 - req_x0, CV_32F, cv::Scalar(0.f));
    meta_.canvas_bounds_px = cv::Rect(0, 0, req_x1 - req_x0, req_y1 - req_y0);
    initialized_ = true;
    return;
  }

  int cur_x0 = 0;
  int cur_y0 = 0;
  int cur_x1 = canvas_.cols;
  int cur_y1 = canvas_.rows;

  const int new_x0 = std::min(cur_x0, req_x0);
  const int new_y0 = std::min(cur_y0, req_y0);
  const int new_x1 = std::max(cur_x1, req_x1);
  const int new_y1 = std::max(cur_y1, req_y1);

  if (new_x0 == cur_x0 && new_y0 == cur_y0 && new_x1 == cur_x1 && new_y1 == cur_y1) {
    return;
  }

  if (new_x0 != cur_x0 || new_y0 != cur_y0) {
    meta_.origin_map_x -= static_cast<double>(cur_x0 - new_x0) * meta_.meters_per_px;
    meta_.origin_map_y += static_cast<double>(cur_y0 - new_y0) * meta_.meters_per_px;
  }

  cv::Mat new_canvas(new_y1 - new_y0, new_x1 - new_x0, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat new_weight(new_y1 - new_y0, new_x1 - new_x0, CV_32F, cv::Scalar(0.f));
  const cv::Point offset(cur_x0 - new_x0, cur_y0 - new_y0);
  canvas_.copyTo(new_canvas(cv::Rect(offset.x, offset.y, canvas_.cols, canvas_.rows)));
  weight_map_.copyTo(new_weight(cv::Rect(offset.x, offset.y, weight_map_.cols, weight_map_.rows)));
  canvas_ = std::move(new_canvas);
  weight_map_ = std::move(new_weight);
  meta_.canvas_bounds_px = cv::Rect(0, 0, new_x1 - new_x0, new_y1 - new_y0);
}

void OdomBevMosaicAccumulator::blendWarpedFrame(
  const cv::Mat & warped_bgr, const cv::Mat & warped_mask)
{
  for (int y = 0; y < canvas_.rows; ++y) {
    for (int x = 0; x < canvas_.cols; ++x) {
      if (warped_mask.at<uchar>(y, x) == 0) {
        continue;
      }
      const float w_new = 1.f;
      const float w_old = weight_map_.at<float>(y, x);
      const float w_sum = w_old + w_new;
      cv::Vec3b & dst = canvas_.at<cv::Vec3b>(y, x);
      const cv::Vec3b src = warped_bgr.at<cv::Vec3b>(y, x);
      dst[0] = static_cast<uchar>((dst[0] * w_old + src[0] * w_new) / w_sum);
      dst[1] = static_cast<uchar>((dst[1] * w_old + src[1] * w_new) / w_sum);
      dst[2] = static_cast<uchar>((dst[2] * w_old + src[2] * w_new) / w_sum);
      weight_map_.at<float>(y, x) = w_sum;
    }
  }
}

OdomBevMosaicFrameResult OdomBevMosaicAccumulator::accumulate(
  const cv::Mat & bev_bgr, const Pose2d & pose_map)
{
  OdomBevMosaicFrameResult result;
  if (bev_bgr.empty() || params_.bev_width_px <= 0 || params_.bev_height_px <= 0) {
    return result;
  }
  if (bev_bgr.cols != params_.bev_width_px || bev_bgr.rows != params_.bev_height_px) {
    return result;
  }

  if (!initialized_) {
    meta_.meters_per_px = params_.meters_per_px;
    const double half_w_m = params_.bev_width_px * 0.5 * meta_.meters_per_px;
    const double length_m = params_.bev_height_px * meta_.meters_per_px;
    meta_.origin_map_x = pose_map.x - half_w_m;
    meta_.origin_map_y = pose_map.y + length_m;
  }

  const cv::Mat affine = computeBevToCanvasAffine(pose_map);
  result.warp_affine_2x3 = affine.clone();

  const std::array<cv::Point2f, 4> bev_corners = {
    cv::Point2f(0.f, 0.f),
    cv::Point2f(static_cast<float>(params_.bev_width_px), 0.f),
    cv::Point2f(static_cast<float>(params_.bev_width_px), static_cast<float>(params_.bev_height_px)),
    cv::Point2f(0.f, static_cast<float>(params_.bev_height_px)),
  };
  std::vector<cv::Point2f> canvas_corners;
  canvas_corners.reserve(4);
  for (const auto & corner : bev_corners) {
    canvas_corners.push_back(cv::Point2f(
      static_cast<float>(
        affine.at<double>(0, 0) * corner.x + affine.at<double>(0, 1) * corner.y +
        affine.at<double>(0, 2)),
      static_cast<float>(
        affine.at<double>(1, 0) * corner.x + affine.at<double>(1, 1) * corner.y +
        affine.at<double>(1, 2))));
  }
  ensureCanvasContains(boundingRectFromPoints(canvas_corners));

  // Origin may have shifted while growing the canvas; recompute warp for the final canvas.
  const cv::Mat final_affine = computeBevToCanvasAffine(pose_map);
  result.warp_affine_2x3 = final_affine.clone();

  cv::Mat warped_bgr;
  cv::warpAffine(
    bev_bgr, warped_bgr, final_affine, canvas_.size(), cv::INTER_LINEAR, cv::BORDER_CONSTANT,
    cv::Scalar(0, 0, 0));

  const cv::Mat validity = buildValidityMask(bev_bgr);
  cv::Mat warped_mask;
  cv::warpAffine(validity, warped_mask, final_affine, canvas_.size(), cv::INTER_NEAREST);

  blendWarpedFrame(warped_bgr, warped_mask);

  result.accepted = true;
  ++meta_.frames_accumulated;
  return result;
}

}  // namespace lie_lane_detection
