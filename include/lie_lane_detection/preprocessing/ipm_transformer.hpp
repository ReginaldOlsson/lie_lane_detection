#pragma once

#include <opencv2/core.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

class IPMTransformer
{
public:
  explicit IPMTransformer(const PipelineParams & params);

  void updateParams(const PipelineParams & params);

  bool computeHomography(const sensor_msgs::msg::CameraInfo * camera_info = nullptr);

  cv::Mat warpToBev(const cv::Mat & image, cv::Mat * H_out = nullptr) const;

  int bevWidthPx() const {return bev_width_px_;}
  int bevHeightPx() const {return bev_height_px_;}
  double metersPerPixel() const {return meters_per_px_;}

  const cv::Mat & homography() const {return H_img2bev_;}

private:
  PipelineParams params_;
  cv::Mat H_img2bev_;
  int bev_width_px_{0};
  int bev_height_px_{0};
  double meters_per_px_{0.05};
  bool homography_valid_{false};
};

}  // namespace lie_lane_detection
