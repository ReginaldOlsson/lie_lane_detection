#include "lie_lane_detection/ipm_transformer.hpp"

#include <opencv2/imgproc.hpp>

namespace lie_lane_detection
{

IPMTransformer::IPMTransformer(const PipelineParams & params)
: params_(params)
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

cv::Mat IPMTransformer::warpToBev(const cv::Mat & image, cv::Mat * H_out) const
{
  if (!homography_valid_) {
    if (H_out) {
      *H_out = cv::Mat();
    }
    return cv::Mat();
  }
  cv::Mat bev;
  cv::warpPerspective(
    image, bev, H_img2bev_, cv::Size(bev_width_px_, bev_height_px_),
    cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  if (H_out) {
    *H_out = H_img2bev_.clone();
  }
  return bev;
}

}  // namespace lie_lane_detection
