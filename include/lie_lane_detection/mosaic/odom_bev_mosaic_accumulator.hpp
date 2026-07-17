#pragma once

#include <opencv2/core.hpp>

#include "lie_lane_detection/mosaic/pose2d.hpp"

namespace lie_lane_detection
{

struct OdomBevMosaicParams
{
  double meters_per_px{0.05};
  int bev_width_px{0};
  int bev_height_px{0};
  uchar mask_gray_threshold{25};
  /// Extra margin (meters) when growing the global canvas.
  double canvas_padding_m{5.0};
  /// Optional body-frame offset applied before warping (meters / radians).
  double pose_lateral_offset_m{0.0};
  double pose_forward_offset_m{0.0};
  double pose_yaw_offset_rad{0.0};
};

struct OdomBevMosaicMeta
{
  double origin_map_x{0.0};
  double origin_map_y{0.0};
  double meters_per_px{0.05};
  int frames_accumulated{0};
  int frames_skipped{0};
  cv::Rect canvas_bounds_px;
};

struct OdomBevMosaicFrameResult
{
  bool accepted{false};
  cv::Mat warp_affine_2x3;
};

/// Global ortho mosaic: place metric BEV tiles using map-frame odometry.
class OdomBevMosaicAccumulator
{
public:
  explicit OdomBevMosaicAccumulator(OdomBevMosaicParams params = OdomBevMosaicParams{});

  void setParams(const OdomBevMosaicParams & params) {params_ = params;}
  void reset();

  OdomBevMosaicFrameResult accumulate(const cv::Mat & bev_bgr, const Pose2d & pose_map);

  const cv::Mat & canvas() const {return canvas_;}
  const OdomBevMosaicMeta & meta() const {return meta_;}
  bool initialized() const {return initialized_;}

  /// Map a metric ground point in the BEV body frame to canvas pixel coordinates.
  cv::Point2d mapBodyToCanvasPx(double lateral_m, double forward_m) const;

  /// Build 2×3 affine that maps BEV pixels into the current canvas.
  cv::Mat computeBevToCanvasAffine(const Pose2d & pose_map) const;

private:
  void ensureCanvasContains(const cv::Rect & required_px);
  void blendWarpedFrame(const cv::Mat & warped_bgr, const cv::Mat & warped_mask);
  cv::Mat buildValidityMask(const cv::Mat & bev_bgr) const;

  OdomBevMosaicParams params_;
  OdomBevMosaicMeta meta_;
  cv::Mat canvas_;
  cv::Mat weight_map_;
  bool initialized_{false};
};

}  // namespace lie_lane_detection
