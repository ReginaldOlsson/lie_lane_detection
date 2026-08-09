#include "lie_lane_detection/preprocessing/boreas_calib.hpp"

#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>

namespace lie_lane_detection
{

namespace
{

/// Boreas manual IPM: axis-aligned src rectangle in rectified image pixels.
void boreasDefaultSrcRectangle(const BoreasCalib & calib, PipelineParams & params)
{
  constexpr double kInsetPx = 500.0;
  constexpr double kTopV = 1047.0 + 100.0;
  const double u_min = kInsetPx;
  const double u_max = static_cast<double>(calib.image_width) - kInsetPx;
  const double v_top = kTopV;
  const double v_bottom = static_cast<double>(calib.image_height);
  params.ipm_src_points = {
    u_min, v_bottom, u_max, v_bottom, u_max, v_top, u_min, v_top,
  };
}

bool parseYamlImageSize(const std::filesystem::path & yaml_path, int & w, int & h)
{
  std::ifstream in(yaml_path);
  if (!in) {
    return false;
  }
  std::string line;
  std::regex size_re(R"(image_size:\s*\[([0-9.]+),\s*([0-9.]+)\])");
  while (std::getline(in, line)) {
    std::smatch m;
    if (std::regex_search(line, m, size_re) && m.size() >= 3) {
      w = static_cast<int>(std::lround(std::stod(m[1].str())));
      h = static_cast<int>(std::lround(std::stod(m[2].str())));
      return w > 0 && h > 0;
    }
  }
  return false;
}

bool readMatrixRows(const std::filesystem::path & path, int rows, int cols, cv::Mat & out)
{
  std::ifstream in(path);
  if (!in) {
    return false;
  }
  out = cv::Mat(rows, cols, CV_64F);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      double v = 0.0;
      if (!(in >> v)) {
        return false;
      }
      out.at<double>(r, c) = v;
    }
  }
  return true;
}

cv::Vec3d projectLidarGround(const BoreasCalib & calib, double x_fwd_m, double y_left_m)
{
  // Boreas: T_camera_lidar maps lidar-frame homogeneous points into the camera frame;
  // P (3×4) projects camera-frame homogeneous coords to rectified pixels.
  const cv::Mat p_lidar = (cv::Mat_<double>(4, 1) << x_fwd_m, y_left_m, 0.0, 1.0);
  const cv::Mat p_cam = calib.T_camera_lidar * p_lidar;
  const cv::Mat uvw = calib.P * p_cam;
  const double w = uvw.at<double>(2);
  if (std::abs(w) < 1e-9) {
    return cv::Vec3d(0, 0, 0);
  }
  return cv::Vec3d(uvw.at<double>(0) / w, uvw.at<double>(1) / w, w);
}

struct GroundCorner
{
  double x_fwd_m{0.0};
  double y_left_m{0.0};
};

bool findLateralRangeAtX(
  const BoreasCalib & calib, double x_fwd_m, double & y_left_m, double & y_right_m)
{
  double y_lo = std::numeric_limits<double>::max();
  double y_hi = std::numeric_limits<double>::lowest();
  bool any = false;
  for (double y = -12.0; y <= 12.0; y += 0.05) {
    const cv::Vec3d uv = projectLidarGround(calib, x_fwd_m, y);
    if (uv[2] <= 0.0) {
      continue;
    }
    if (uv[0] < 0.0 || uv[0] >= calib.image_width || uv[1] < 0.0 || uv[1] >= calib.image_height) {
      continue;
    }
    any = true;
    y_lo = std::min(y_lo, y);
    y_hi = std::max(y_hi, y);
  }
  if (!any || y_hi - y_lo < 0.5) {
    return false;
  }
  // Boreas lidar: +y is left; keep a margin inside the visible interval.
  const double margin = 0.15 * (y_hi - y_lo);
  y_left_m = y_hi - margin;
  y_right_m = y_lo + margin;
  return y_left_m > y_right_m;
}

bool findNearForwardX(const BoreasCalib & calib, double & x_near_m)
{
  for (double x = 1.0; x <= 20.0; x += 0.5) {
    double yl = 0.0;
    double yr = 0.0;
    if (findLateralRangeAtX(calib, x, yl, yr)) {
      x_near_m = x;
      return true;
    }
  }
  return false;
}

bool findFarForwardX(
  const BoreasCalib & calib, double x_near_m, double x_target_m, double & x_far_m)
{
  x_far_m = x_near_m;
  for (double x = x_near_m + 1.0; x <= x_target_m; x += 1.0) {
    double yl = 0.0;
    double yr = 0.0;
    if (!findLateralRangeAtX(calib, x, yl, yr)) {
      break;
    }
    x_far_m = x;
  }
  return x_far_m > x_near_m + 1.0;
}

bool buildVisibleGroundQuad(
  const BoreasCalib & calib, double length_m, GroundCorner & bl, GroundCorner & br,
  GroundCorner & tr, GroundCorner & tl)
{
  double x_near = 0.0;
  if (!findNearForwardX(calib, x_near)) {
    return false;
  }
  double x_far = x_near;
  if (!findFarForwardX(calib, x_near, length_m, x_far)) {
    return false;
  }

  double yl_near = 0.0;
  double yr_near = 0.0;
  double yl_far = 0.0;
  double yr_far = 0.0;
  if (
    !findLateralRangeAtX(calib, x_near, yl_near, yr_near) ||
    !findLateralRangeAtX(calib, x_far, yl_far, yr_far)) {
    return false;
  }

  bl = {x_near, yl_near};
  br = {x_near, yr_near};
  tr = {x_far, yr_far};
  tl = {x_far, yl_far};
  return true;
}

}  // namespace

bool loadBoreasCalib(const std::filesystem::path & calib_dir, BoreasCalib & out)
{
  const auto p_path = calib_dir / "P_camera.txt";
  const auto t_path = calib_dir / "T_camera_lidar.txt";
  cv::Mat p_full;
  if (!readMatrixRows(p_path, 4, 4, p_full)) {
    return false;
  }
  out.P = p_full.rowRange(0, 3).clone();

  if (!readMatrixRows(t_path, 4, 4, out.T_camera_lidar)) {
    return false;
  }

  out.image_width = 2448;
  out.image_height = 2048;
  parseYamlImageSize(calib_dir / "camera0_intrinsics.yaml", out.image_width, out.image_height);
  return true;
}

bool boreasLidarGroundToImage(
  const BoreasCalib & calib, double x_fwd_m, double y_left_m, double & u, double & v)
{
  const cv::Vec3d uv = projectLidarGround(calib, x_fwd_m, y_left_m);
  if (uv[2] <= 0.0) {
    return false;
  }
  u = uv[0];
  v = uv[1];
  return std::isfinite(u) && std::isfinite(v);
}

bool boreasImageToLidarGround(
  const BoreasCalib & calib, double u, double v, double & x_fwd_m, double & y_left_m)
{
  // Solve P * T * [x, y, 0, 1]^T = λ [u, v, 1]^T for ground (x, y).
  const cv::Mat M = calib.P * calib.T_camera_lidar;
  if (M.rows != 3 || M.cols != 4) {
    return false;
  }

  const double m00 = M.at<double>(0, 0);
  const double m01 = M.at<double>(0, 1);
  const double m03 = M.at<double>(0, 3);
  const double m10 = M.at<double>(1, 0);
  const double m11 = M.at<double>(1, 1);
  const double m13 = M.at<double>(1, 3);
  const double m20 = M.at<double>(2, 0);
  const double m21 = M.at<double>(2, 1);
  const double m23 = M.at<double>(2, 3);

  const double a11 = u * m20 - m00;
  const double a12 = u * m21 - m01;
  const double b1 = m03 - u * m23;
  const double a21 = v * m20 - m10;
  const double a22 = v * m21 - m11;
  const double b2 = m13 - v * m23;

  const double det = a11 * a22 - a12 * a21;
  if (std::abs(det) < 1e-12) {
    return false;
  }

  x_fwd_m = (b1 * a22 - a12 * b2) / det;
  y_left_m = (a11 * b2 - b1 * a21) / det;
  return std::isfinite(x_fwd_m) && std::isfinite(y_left_m);
}

namespace
{

bool cameraRayDirection(const BoreasCalib & calib, double u, double v, cv::Vec3d & dir_out)
{
  const cv::Mat k = calib.P.colRange(0, 3).clone();
  cv::Mat k_inv;
  if (!cv::invert(k, k_inv)) {
    return false;
  }
  const cv::Mat d = k_inv * (cv::Mat_<double>(3, 1) << u, v, 1.0);
  dir_out = cv::Vec3d(d.at<double>(0), d.at<double>(1), d.at<double>(2));
  return dir_out[2] > 1e-9;
}

double cameraXZPlaneYFromReference(
  const BoreasCalib & calib, double u_ref, double v_ref, double z_ref_m)
{
  cv::Vec3d d;
  if (!cameraRayDirection(calib, u_ref, v_ref, d)) {
    return 1.0;
  }
  const double t = z_ref_m / d[2];
  return t * d[1];
}

bool boreasImageToCameraXZPlane(
  const BoreasCalib & calib, double u, double v, double y_plane_m, double & x_cam_m,
  double & z_cam_m)
{
  cv::Vec3d d;
  if (!cameraRayDirection(calib, u, v, d)) {
    return false;
  }
  if (std::abs(d[1]) < 1e-9) {
    return false;
  }
  const double t = y_plane_m / d[1];
  if (t <= 0.0) {
    return false;
  }
  x_cam_m = t * d[0];
  z_cam_m = t * d[2];
  return z_cam_m > 0.0 && std::isfinite(x_cam_m) && std::isfinite(z_cam_m);
}

}  // namespace

bool configureBoreasGroundIpm(
  const BoreasCalib & calib, PipelineParams & params, cv::Mat & H_img2bev_out,
  IPMTransformer * ipm_out)
{
  updateIpmDstFromBevExtent(params);

  const double half_w = 0.5 * params.bev_width_m;

  GroundCorner bl;
  GroundCorner br;
  GroundCorner tr;
  GroundCorner tl;
  if (!buildVisibleGroundQuad(calib, params.bev_length_m, bl, br, tr, tl)) {
    return false;
  }

  const std::array<GroundCorner, 4> ground = {bl, br, tr, tl};
  // Metric BEV corners (left/right, near/far) — symmetric output grid.
  const std::array<std::pair<double, double>, 4> dst_metric = {{
    {-half_w, 0.0},
    {half_w, 0.0},
    {half_w, params.bev_length_m / 2},
    {-half_w, params.bev_length_m / 2},
  }};

  std::vector<cv::Point2f> src(4);
  params.ipm_src_points.clear();
  params.ipm_dst_points.clear();

  for (int i = 0; i < 4; ++i) {
    const double x_m = ground[static_cast<size_t>(i)].x_fwd_m;
    const double y_m = ground[static_cast<size_t>(i)].y_left_m;
    double u = 0.0;
    double v = 0.0;
    if (!boreasLidarGroundToImage(calib, x_m, y_m, u, v)) {
      return false;
    }
    src[static_cast<size_t>(i)] = cv::Point2f(static_cast<float>(u), static_cast<float>(v));
    params.ipm_src_points.push_back(u);
    params.ipm_src_points.push_back(v);

    const double dst_x_m = dst_metric[static_cast<size_t>(i)].first;
    const double dst_y_m = dst_metric[static_cast<size_t>(i)].second;
    params.ipm_dst_points.push_back(dst_x_m);
    params.ipm_dst_points.push_back(dst_y_m);
  }

  // Sanity: all corners must land inside the rectified image.
  for (const auto & p : src) {
    if (
      p.x < 0.0f || p.y < 0.0f || p.x > static_cast<float>(calib.image_width) ||
      p.y > static_cast<float>(calib.image_height)) {
      return false;
    }
  }

  params.use_manual_ipm = true;

  IPMTransformer ipm_local(params);
  IPMTransformer * ipm = ipm_out != nullptr ? ipm_out : &ipm_local;
  ipm->updateParams(params);
  if (!ipm->computeHomography(nullptr)) {
    return false;
  }
  H_img2bev_out = ipm->homography().clone();
  return true;
}

bool configureBoreasManualIpmSrc(
  const BoreasCalib & calib, PipelineParams & params, cv::Mat & H_img2bev_out,
  IPMTransformer * ipm_out)
{
  // Axis-aligned src rectangle; dst is a ground-plane trapezoid from calibration geometry.
  boreasDefaultSrcRectangle(calib, params);
  params.use_manual_ipm = true;

  struct DstCorner
  {
    double x_m{0.0};
    double y_m{0.0};
  };
  std::array<DstCorner, 4> dst_corners;

  // Camera XZ plane (Y = y_plane): anchored using principal column cx and bottom ROI row.
  const double cx = calib.P.at<double>(0, 2);
  const double v_bottom = params.ipm_src_points[1];
  constexpr double kNearDepthM = 1.0;
  const double y_plane = cameraXZPlaneYFromReference(calib, cx, v_bottom, kNearDepthM);

  for (int i = 0; i < 4; ++i) {
    const double u = params.ipm_src_points[2 * i];
    const double v = params.ipm_src_points[2 * i + 1];
    double x_cam = 0.0;
    double z_cam = 0.0;
    if (!boreasImageToCameraXZPlane(calib, u, v, y_plane, x_cam, z_cam)) {
      return false;
    }
    dst_corners[static_cast<size_t>(i)].x_m = -x_cam;
    dst_corners[static_cast<size_t>(i)].y_m = z_cam;
  }

  double x_lat_min = dst_corners[0].x_m;
  double x_lat_max = dst_corners[0].x_m;
  double y_fwd_min = dst_corners[0].y_m;
  double y_fwd_max = dst_corners[0].y_m;
  for (const auto & c : dst_corners) {
    x_lat_min = std::min(x_lat_min, c.x_m);
    x_lat_max = std::max(x_lat_max, c.x_m);
    y_fwd_min = std::min(y_fwd_min, c.y_m);
    y_fwd_max = std::max(y_fwd_max, c.y_m);
  }

  const double lat_span = x_lat_max - x_lat_min;
  const double fwd_span = y_fwd_max - y_fwd_min;
  if (lat_span < 0.5 || fwd_span < 0.5) {
    return false;
  }

  const double lat_center = 0.5 * (x_lat_max + x_lat_min);
  constexpr double kMarginRatio = 0.02;
  params.bev_width_m = lat_span * (1.0 + kMarginRatio);
  params.bev_length_m = fwd_span * (1.0 + kMarginRatio);

  // BEV resolution: ~1 px per src ROI px (capped), so IPM detail matches the rectangle size.
  const double roi_w_px = std::max(1.0, params.ipm_src_points[2] - params.ipm_src_points[0]);
  const double roi_h_px = std::max(1.0, params.ipm_src_points[1] - params.ipm_src_points[5]);
  constexpr int kMaxBevPx = 2048;
  constexpr int kMinBevPx = 320;
  const double target_w_px = std::min(roi_w_px, static_cast<double>(kMaxBevPx));
  const double target_h_px = std::min(roi_h_px, static_cast<double>(kMaxBevPx));
  const double mpp_from_roi_w = params.bev_width_m / target_w_px;
  const double mpp_from_roi_h = params.bev_length_m / target_h_px;
  const double mpp_from_roi = std::min(mpp_from_roi_w, mpp_from_roi_h);
  const double max_span_m = std::max(params.bev_width_m, params.bev_length_m);
  const double mpp_min_dim = max_span_m / static_cast<double>(kMinBevPx);
  params.bev_resolution_m_per_px =
    std::min({params.bev_resolution_m_per_px, mpp_from_roi, mpp_min_dim});

  params.ipm_dst_points.clear();
  params.ipm_dst_points.reserve(8);
  for (const auto & c : dst_corners) {
    params.ipm_dst_points.push_back(c.x_m - lat_center);
    params.ipm_dst_points.push_back(c.y_m - y_fwd_min);
  }

  IPMTransformer ipm_local(params);
  IPMTransformer * ipm = ipm_out != nullptr ? ipm_out : &ipm_local;
  ipm->updateParams(params);
  if (!ipm->computeHomography(nullptr)) {
    return false;
  }
  H_img2bev_out = ipm->homography().clone();
  return true;
}

}  // namespace lie_lane_detection
