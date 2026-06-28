#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "lie_lane_detection/mosaic/bev_mosaic_accumulator.hpp"

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
  EXPECT_TRUE(result.valid);
  EXPECT_TRUE(acc.initialized());
  ASSERT_FALSE(acc.canvas().empty());
  EXPECT_EQ(acc.canvas().cols, frame.cols);
  EXPECT_EQ(acc.canvas().rows, frame.rows);
}

TEST(BevMosaicAccumulator, pureTranslationExpandsCanvas)
{
  BevMosaicParams params;
  params.registration_method = BevRegistrationMethod::ECC;
  params.min_ecc_correlation = 0.2;
  BevMosaicAccumulator acc(params);

  const cv::Mat frame0 = makeStripedBev(200, 300);
  cv::Mat frame1;
  const cv::Mat shift = (cv::Mat_<double>(2, 3) << 1.0, 0.0, -12.0, 0.0, 1.0, -8.0);
  cv::warpAffine(frame0, frame1, shift, frame0.size());

  acc.accumulate(frame0);
  const auto result = acc.accumulate(frame1);
  EXPECT_TRUE(result.valid);
  EXPECT_GE(acc.canvas().cols, frame0.cols);
  EXPECT_GE(acc.canvas().rows, frame0.rows);
}

}  // namespace
}  // namespace lie_lane_detection
