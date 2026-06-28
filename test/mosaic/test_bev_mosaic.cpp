#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "lie_lane_detection/mosaic/bev_mosaic_accumulator.hpp"
#include "lie_lane_detection/mosaic/bev_registration.hpp"

namespace lie_lane_detection
{
namespace
{

cv::Mat makeStripedBev(int cols, int rows)
{
  cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(40, 40, 40));
  for (int x = 40; x < cols; x += 80) {
    cv::line(img, cv::Point(x, 0), cv::Point(x, rows - 1), cv::Scalar(220, 220, 220), 3);
  }
  for (int y = 30; y < rows; y += 60) {
    cv::line(img, cv::Point(0, y), cv::Point(cols - 1, y), cv::Scalar(180, 180, 180), 2);
  }
  return img;
}

TEST(BevMosaicAccumulator, firstFrameInitializesCanvas)
{
  BevMosaicAccumulator acc;
  const cv::Mat frame = makeStripedBev(240, 400);
  const auto result = acc.accumulate(frame);
  EXPECT_TRUE(result.motion.valid);
  EXPECT_TRUE(acc.initialized());
  ASSERT_FALSE(acc.canvas().empty());
  EXPECT_EQ(acc.canvas().cols, frame.cols);
  EXPECT_EQ(acc.canvas().rows, frame.rows);
}

TEST(BevRegistration, identicalImagesGiveIdentity)
{
  BevRegistrationParams params;
  params.method = BevRegistrationMethod::ECC_THEN_ORB;
  const cv::Mat frame = makeStripedBev(200, 300);
  const auto reg = estimateBevFrameMotion(frame, frame, params);
  EXPECT_NEAR(reg.dx_px, 0.0, 1.0);
  EXPECT_NEAR(reg.dy_px, 0.0, 1.0);
  EXPECT_NEAR(reg.yaw_rad, 0.0, 0.05);
}

}  // namespace
}  // namespace lie_lane_detection
