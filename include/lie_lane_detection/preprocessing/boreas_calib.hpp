#pragma once

#include <filesystem>
#include <string>

#include <opencv2/core.hpp>

#include "lie_lane_detection/core/types.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"

namespace lie_lane_detection
{

/// Boreas rectified-camera calibration (see dataset calib/ folder).
struct BoreasCalib
{
  cv::Mat P{cv::Mat::eye(3, 4, CV_64F)};              // 3×4 projection matrix
  cv::Mat T_camera_lidar{cv::Mat::eye(4, 4, CV_64F)};  // lidar → camera
  int image_width{0};
  int image_height{0};
};

/// Load P_camera.txt (+ optional camera0_intrinsics.yaml for image size).
bool loadBoreasCalib(const std::filesystem::path & calib_dir, BoreasCalib & out);

/// Project a lidar-frame ground point (x forward, y left, z=0) to rectified image (u,v).
bool boreasLidarGroundToImage(
  const BoreasCalib & calib, double x_fwd_m, double y_left_m, double & u, double & v);

/// Inverse: rectified pixel (u,v) → lidar ground (x forward, y left) via ray–z=0 intersection.
bool boreasImageToLidarGround(
  const BoreasCalib & calib,
  double u, double v,
  double & x_fwd_m,   double & y_left_m);

/// Build metric IPM from Boreas extrinsics + P (auto ground quad in lidar frame).
bool configureBoreasGroundIpm(
  const BoreasCalib & calib,
  PipelineParams & params,
  cv::Mat & H_img2bev_out,
  IPMTransformer * ipm_out = nullptr);

/// Manual axis-aligned src rectangle (image px) → dst trapezoid on camera XZ plane.
/// Requires P_camera (intrinsics); auto-sizes bev_width_m / bev_length_m.
bool configureBoreasManualIpmSrc(
  const BoreasCalib & calib,
  PipelineParams & params,
  cv::Mat & H_img2bev_out,
  IPMTransformer * ipm_out = nullptr);

}  // namespace lie_lane_detection
