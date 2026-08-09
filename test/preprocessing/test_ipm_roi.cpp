#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"

#include <gtest/gtest.h>

namespace lie_lane_detection
{
namespace
{

TEST(IpmRoiParams, updateIpmDstFromBevExtent)
{
  PipelineParams params;
  params.bev_width_m = 20.0;
  params.bev_length_m = 60.0;
  updateIpmDstFromBevExtent(params);
  ASSERT_EQ(params.ipm_dst_points.size(), 8u);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[0], -10.0);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[1], 0.0);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[2], 10.0);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[3], 0.0);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[4], 10.0);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[5], 60.0);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[6], -10.0);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[7], 60.0);
}

TEST(IpmRoiParams, configureAutoIpmRoiUsesCustomRatios)
{
  PipelineParams params;
  params.bev_width_m = 18.0;
  params.bev_length_m = 50.0;
  params.ipm_bottom_x_min_ratio = 0.02;
  params.ipm_bottom_x_max_ratio = 0.98;
  params.ipm_bottom_y_ratio = 0.95;
  params.ipm_top_y_offset_ratio = 0.10;
  params.ipm_top_y_min_ratio = 0.30;
  params.ipm_top_y_max_ratio = 0.70;

  const int cols = 640;
  const int rows = 480;
  const double vp_x = 0.5 * cols;
  const double vp_y = 0.25 * rows;

  ASSERT_TRUE(configureAutoIpmRoi(params, cols, rows, vp_x, vp_y));
  ASSERT_EQ(params.ipm_src_points.size(), 8u);
  EXPECT_NEAR(params.ipm_src_points[0], 0.02 * cols, 1.0);
  EXPECT_NEAR(params.ipm_src_points[2], 0.98 * cols, 1.0);
  EXPECT_NEAR(params.ipm_src_points[1], 0.95 * rows, 1.0);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[0], -9.0);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[2], 9.0);
  EXPECT_DOUBLE_EQ(params.ipm_dst_points[5], 50.0);
}

}  // namespace
}  // namespace lie_lane_detection
