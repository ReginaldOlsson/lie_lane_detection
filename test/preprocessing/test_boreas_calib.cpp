#include <cmath>

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

#include "lie_lane_detection/preprocessing/boreas_calib.hpp"

namespace lie_lane_detection
{
namespace
{

BoreasCalib syntheticBoreasCalib()
{
  BoreasCalib calib;
  calib.P = (cv::Mat_<double>(3, 4) <<
    1440.1585693359375, 0.0, 1218.227729041253, 0.0,
    0.0, 1446.9215087890625, 1045.272153129241, 0.0,
    0.0, 0.0, 1.0, 0.0);
  calib.T_camera_lidar = (cv::Mat_<double>(4, 4) <<
    0.729304194569723, -0.684189392429244, -0.000516788457270013, -0.0231487194000858,
    -0.0128793477606882, -0.0129733964677743, -0.999832892730255, -0.229363448544232,
    0.68406835490634, 0.729188978435215, -0.018273465581034, -0.635346228374209,
    0.0, 0.0, 0.0, 1.0);
  calib.image_width = 2448;
  calib.image_height = 2048;
  return calib;
}

}  // namespace

TEST(BoreasCalib, imageGroundRoundTrip)
{
  const BoreasCalib calib = syntheticBoreasCalib();

  const std::array<std::pair<double, double>, 4> samples = {{
    {10.0, -3.0},
    {10.0, 3.0},
    {40.0, -2.0},
    {40.0, 2.0},
  }};

  for (const auto & [x_fwd, y_left] : samples) {
    double u = 0.0;
    double v = 0.0;
    ASSERT_TRUE(boreasLidarGroundToImage(calib, x_fwd, y_left, u, v));

    double x_back = 0.0;
    double y_back = 0.0;
    ASSERT_TRUE(boreasImageToLidarGround(calib, u, v, x_back, y_back))
      << "failed at (" << u << "," << v << ")";

    EXPECT_NEAR(x_back, x_fwd, 1e-3) << "x_fwd mismatch";
    EXPECT_NEAR(y_back, y_left, 1e-3) << "y_left mismatch";
  }
}

TEST(BoreasCalib, configureManualIpmProducesHomography)
{
  const BoreasCalib calib = syntheticBoreasCalib();
  PipelineParams params;
  params.bev_resolution_m_per_px = 0.05;

  cv::Mat H;
  IPMTransformer ipm(params);
  ASSERT_TRUE(configureBoreasManualIpmSrc(calib, params, H, &ipm));

  ASSERT_EQ(params.ipm_src_points.size(), 8u);
  ASSERT_EQ(params.ipm_dst_points.size(), 8u);
  EXPECT_GT(params.bev_width_m, 0.5);
  EXPECT_GT(params.bev_length_m, 0.5);
  EXPECT_EQ(H.rows, 3);
  EXPECT_EQ(H.cols, 3);
  EXPECT_FALSE(H.empty());

  ipm.updateParams(params);
  const int bev_w = ipm.bevWidthPx();
  const int bev_h = ipm.bevHeightPx();
  const double roi_w = params.ipm_src_points[2] - params.ipm_src_points[0];
  EXPECT_GE(bev_w, static_cast<int>(roi_w * 0.85));
  EXPECT_LE(bev_w, 2048);
  EXPECT_GE(bev_h, 300);
}

}  // namespace lie_lane_detection
