#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>

namespace lie_lane_detection
{

IPMTransformer::IPMTransformer(const PipelineParams & params) : params_(params)
{
  updateParams(params);
}

void IPMTransformer::updateParams(const PipelineParams & params)
{
  params_ = params;
  meters_per_px_ = params_.bev_resolution_m_per_px;
  bev_width_px_ = static_cast<int>(params_.bev_width_m / meters_per_px_);
  bev_height_px_ = static_cast<int>(params_.bev_length_m / meters_per_px_);
  homography_valid_ = false;
}

bool IPMTransformer::computeHomography(const sensor_msgs::msg::CameraInfo * /*camera_info*/)
{
  if (params_.ipm_src_points.size() < 8 || params_.ipm_dst_points.size() < 8) {
    homography_valid_ = false;
    return false;
  }

  std::vector<cv::Point2f> src(4);
  std::vector<cv::Point2f> dst(4);
  for (int i = 0; i < 4; ++i) {
    src[i] = cv::Point2f(
      static_cast<float>(params_.ipm_src_points[2 * i]),
      static_cast<float>(params_.ipm_src_points[2 * i + 1]));
    const double dst_x_m = params_.ipm_dst_points[2 * i];
    const double dst_y_m = params_.ipm_dst_points[2 * i + 1];
    dst[i] = cv::Point2f(
      static_cast<float>(dst_x_m / meters_per_px_ + bev_width_px_ * 0.5f),
      static_cast<float>(bev_height_px_ - dst_y_m / meters_per_px_));
  }

  H_img2bev_ = cv::getPerspectiveTransform(src, dst);
  homography_valid_ = true;
  return true;
}

namespace
{

cv::Mat maskImageToIpmSrcRoi(const cv::Mat & image_bgr, const PipelineParams & params)
{
  if (image_bgr.empty()) {
    return image_bgr;
  }
  if (params.ipm_src_points.size() < 8) {
    return image_bgr.clone();
  }
  std::vector<cv::Point> poly(4);
  for (int i = 0; i < 4; ++i) {
    poly[static_cast<size_t>(i)] = cv::Point(
      static_cast<int>(std::lround(params.ipm_src_points[2 * i])),
      static_cast<int>(std::lround(params.ipm_src_points[2 * i + 1])));
  }
  cv::Mat mask = cv::Mat::zeros(image_bgr.size(), CV_8UC1);
  cv::fillPoly(mask, std::vector<std::vector<cv::Point>>{poly}, cv::Scalar(255));
  cv::Mat out = cv::Mat::zeros(image_bgr.size(), image_bgr.type());
  image_bgr.copyTo(out, mask);
  return out;
}

void metricDstMeters(const PipelineParams & params, std::array<std::pair<double, double>, 4> & out)
{
  if (params.ipm_dst_points.size() >= 8) {
    for (int i = 0; i < 4; ++i) {
      out[static_cast<size_t>(i)] = {
        params.ipm_dst_points[2 * i], params.ipm_dst_points[2 * i + 1]};
    }
    return;
  }
  const double half_w = 0.5 * params.bev_width_m;
  out = {{
    {-half_w, 0.0},
    {half_w, 0.0},
    {half_w, params.bev_length_m},
    {-half_w, params.bev_length_m},
  }};
}

std::vector<cv::Point2f> ipmMetricDstBevCorners(const PipelineParams & params)
{
  std::array<std::pair<double, double>, 4> dst_m;
  metricDstMeters(params, dst_m);

  const double mpp = params.bev_resolution_m_per_px;
  const int bev_w = static_cast<int>(params.bev_width_m / mpp);
  const int bev_h = static_cast<int>(params.bev_length_m / mpp);

  std::vector<cv::Point2f> corners(4);
  for (int i = 0; i < 4; ++i) {
    const double x_m = dst_m[static_cast<size_t>(i)].first;
    const double y_m = dst_m[static_cast<size_t>(i)].second;
    corners[static_cast<size_t>(i)] = cv::Point2f(
      static_cast<float>(x_m / mpp + bev_w * 0.5), static_cast<float>(bev_h - y_m / mpp));
  }
  return corners;
}

bool ipmMetricDstImageCorners(
  const cv::Mat & H_img2bev, const PipelineParams & params, std::vector<cv::Point2f> & corners_out)
{
  if (H_img2bev.empty() || H_img2bev.rows != 3 || H_img2bev.cols != 3) {
    return false;
  }
  const std::vector<cv::Point2f> bev_corners = ipmMetricDstBevCorners(params);
  cv::Mat H_bev2img;
  cv::invert(H_img2bev, H_bev2img, cv::DECOMP_LU);
  cv::perspectiveTransform(bev_corners, corners_out, H_bev2img);
  return corners_out.size() == 4;
}

bool ipmSrcBevCorners(
  const cv::Mat & H_img2bev,
  const PipelineParams & params,
  std::vector<cv::Point2f> & corners_out)
{
  if (H_img2bev.empty() || H_img2bev.rows != 3 || H_img2bev.cols != 3 ||
    params.ipm_src_points.size() < 8)
  {
    return false;
  }
  std::vector<cv::Point2f> src(4);
  for (int i = 0; i < 4; ++i) {
    src[static_cast<size_t>(i)] = cv::Point2f(
      static_cast<float>(params.ipm_src_points[2 * i]),
      static_cast<float>(params.ipm_src_points[2 * i + 1]));
  }
  cv::perspectiveTransform(src, corners_out, H_img2bev);
  return corners_out.size() == 4;
}

std::vector<cv::Point> toPointPoly(const std::vector<cv::Point2f> & pts)
{
  std::vector<cv::Point> poly(pts.size());
  for (size_t i = 0; i < pts.size(); ++i) {
    poly[i] =
      cv::Point(static_cast<int>(std::lround(pts[i].x)), static_cast<int>(std::lround(pts[i].y)));
  }
  return poly;
}

void drawCornerLabels(cv::Mat & image_bgr, const std::vector<cv::Point> & corners, const cv::Scalar & color)
{
  static const char * kLabels[] = {"BL", "BR", "TR", "TL"};
  for (int i = 0; i < 4 && i < static_cast<int>(corners.size()); ++i) {
    cv::circle(image_bgr, corners[static_cast<size_t>(i)], 4, color, -1, cv::LINE_AA);
    cv::putText(
      image_bgr, kLabels[i],
      corners[static_cast<size_t>(i)] + cv::Point(5, -5),
      cv::FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv::LINE_AA);
  }
}

void drawIpmMetricDstOnBev(
  cv::Mat & bev_bgr, const PipelineParams & params, const cv::Scalar & color, int thickness)
{
  if (bev_bgr.empty()) {
    return;
  }
  const std::vector<cv::Point> poly = toPointPoly(ipmMetricDstBevCorners(params));
  cv::polylines(bev_bgr, std::vector<std::vector<cv::Point>>{poly}, true, color, thickness);
}

void drawIpmSrcRoiOnBev(
  cv::Mat & bev_bgr, const cv::Mat & H_img2bev, const PipelineParams & params,
  const cv::Scalar & color, int thickness)
{
  if (bev_bgr.empty()) {
    return;
  }
  std::vector<cv::Point2f> corners;
  if (!ipmSrcBevCorners(H_img2bev, params, corners)) {
    return;
  }
  const std::vector<cv::Point> poly = toPointPoly(corners);
  cv::polylines(bev_bgr, std::vector<std::vector<cv::Point>>{poly}, true, color, thickness);
  drawCornerLabels(bev_bgr, poly, color);
}

}  // namespace

cv::Mat IPMTransformer::warpToBev(const cv::Mat & image, cv::Mat * H_out) const
{
  if (!homography_valid_) {
    if (H_out) {
      *H_out = cv::Mat();
    }
    return cv::Mat();
  }
  const cv::Mat masked = maskImageToIpmSrcRoi(image, params_);
  cv::Mat bev;
  cv::warpPerspective(
    masked, bev, H_img2bev_, cv::Size(bev_width_px_, bev_height_px_), cv::INTER_LINEAR,
    cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  if (H_out) {
    *H_out = H_img2bev_.clone();
  }
  return bev;
}

void drawIpmSrcRoi(
  cv::Mat & image_bgr, const PipelineParams & params, const cv::Scalar & color, int thickness)
{
  if (image_bgr.empty() || params.ipm_src_points.size() < 8) {
    return;
  }
  std::vector<cv::Point> poly(4);
  for (int i = 0; i < 4; ++i) {
    poly[static_cast<size_t>(i)] = cv::Point(
      static_cast<int>(std::lround(params.ipm_src_points[2 * i])),
      static_cast<int>(std::lround(params.ipm_src_points[2 * i + 1])));
  }
  cv::polylines(image_bgr, std::vector<std::vector<cv::Point>>{poly}, true, color, thickness);
  drawCornerLabels(image_bgr, poly, color);
}

void drawIpmRoiOnBev(cv::Mat & bev_bgr, const cv::Mat & H_img2bev, const PipelineParams & params)
{
  drawIpmMetricDstOnBev(bev_bgr, params, cv::Scalar(255, 0, 255), 2);
  drawIpmSrcRoiOnBev(bev_bgr, H_img2bev, params, cv::Scalar(0, 255, 255), 2);
}

void drawIpmMetricDstOnImage(
  cv::Mat & image_bgr, const cv::Mat & H_img2bev, const PipelineParams & params,
  const cv::Scalar & color, int thickness)
{
  if (image_bgr.empty()) {
    return;
  }
  std::vector<cv::Point2f> img_corners;
  if (!ipmMetricDstImageCorners(H_img2bev, params, img_corners)) {
    return;
  }
  const std::vector<cv::Point> poly = toPointPoly(img_corners);
  cv::polylines(image_bgr, std::vector<std::vector<cv::Point>>{poly}, true, color, thickness);
}

}  // namespace lie_lane_detection
