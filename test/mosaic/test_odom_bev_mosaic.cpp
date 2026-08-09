#include "lie_lane_detection/mosaic/odom_bev_mosaic_accumulator.hpp"
#include "lie_lane_detection/mosaic/pose_buffer.hpp"

#include <opencv2/imgproc.hpp>

#include <gtest/gtest.h>

#include <cmath>

namespace lie_lane_detection
{
namespace
{

cv::Mat makeStripedBev(const int cols, const int rows)
{
  cv::Mat img(rows, cols, CV_8UC3, cv::Scalar(40, 40, 40));
  for (int x = 20; x < cols; x += 40) {
    cv::line(img, cv::Point(x, 0), cv::Point(x, rows - 1), cv::Scalar(220, 220, 220), 2);
  }
  return img;
}

OdomBevMosaicParams defaultParams(const int w = 200, const int h = 300)
{
  OdomBevMosaicParams p;
  p.meters_per_px = 0.1;
  p.bev_width_px = w;
  p.bev_height_px = h;
  p.canvas_padding_m = 1.0;
  return p;
}

TEST(OdomBevMosaicAccumulator, straightLineAlignsVerticalStripes)
{
  OdomBevMosaicAccumulator acc(defaultParams());
  const cv::Mat frame = makeStripedBev(200, 300);

  for (int i = 0; i < 5; ++i) {
    Pose2d pose;
    pose.x = static_cast<double>(i) * 2.0;
    pose.y = 0.0;
    pose.yaw_rad = 0.0;
    const auto result = acc.accumulate(frame, pose);
    ASSERT_TRUE(result.accepted);
  }

  ASSERT_TRUE(acc.initialized());
  EXPECT_GT(acc.canvas().cols, frame.cols);
  EXPECT_EQ(acc.meta().frames_accumulated, 5);

  cv::Mat gray;
  cv::cvtColor(acc.canvas(), gray, cv::COLOR_BGR2GRAY);
  cv::Mat edges;
  cv::Sobel(gray, edges, CV_32F, 1, 0, 3);
  cv::Mat abs_edges;
  cv::convertScaleAbs(edges, abs_edges);
  const double edge_energy = cv::sum(abs_edges)[0];
  EXPECT_GT(edge_energy, 1000.0);
}

TEST(OdomBevMosaicAccumulator, turnExpandsCanvas)
{
  OdomBevMosaicAccumulator acc(defaultParams(160, 240));
  const cv::Mat frame = makeStripedBev(160, 240);

  Pose2d pose0;
  pose0.x = 0.0;
  pose0.y = 0.0;
  pose0.yaw_rad = 0.0;
  acc.accumulate(frame, pose0);

  Pose2d pose1;
  pose1.x = 0.0;
  pose1.y = 0.0;
  pose1.yaw_rad = M_PI / 2.0;
  acc.accumulate(frame, pose1);

  EXPECT_GT(acc.canvas().rows, frame.rows / 2);
  EXPECT_GT(acc.canvas().cols, frame.cols / 2);
  EXPECT_EQ(acc.meta().frames_accumulated, 2);
}

TEST(OdomBevMosaicAccumulator, rejectsWrongSize)
{
  OdomBevMosaicAccumulator acc(defaultParams());
  const cv::Mat wrong_size(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
  Pose2d pose;
  const auto result = acc.accumulate(wrong_size, pose);
  EXPECT_FALSE(result.accepted);
  EXPECT_FALSE(acc.initialized());
}

TEST(PoseBuffer, interpolatesBetweenSamples)
{
  PoseBuffer buffer;
  buffer.addSample(0, Pose2d{0.0, 0.0, 0.0});
  buffer.addSample(1000000000LL, Pose2d{10.0, 0.0, 0.0});

  const auto mid = buffer.lookup(500000000LL, 600000000LL);
  ASSERT_TRUE(mid.has_value());
  EXPECT_NEAR(mid->x, 5.0, 0.1);
}

TEST(TfPoseResolver, composesMapToBaseLink)
{
  TfPoseResolver resolver;

  geometry_msgs::msg::TransformStamped static_tf;
  static_tf.header.frame_id = "applanix";
  static_tf.child_frame_id = "base_link";
  static_tf.transform.translation.x = 1.0;
  static_tf.transform.translation.y = 2.0;
  static_tf.transform.rotation.w = 1.0;
  resolver.addStaticTransform(static_tf);

  geometry_msgs::msg::TransformStamped dynamic_tf;
  dynamic_tf.header.stamp.sec = 1;
  dynamic_tf.header.frame_id = "map";
  dynamic_tf.child_frame_id = "applanix";
  dynamic_tf.transform.translation.x = 10.0;
  dynamic_tf.transform.translation.y = 20.0;
  dynamic_tf.transform.rotation.w = 1.0;
  resolver.addDynamicTransform(dynamic_tf);

  const auto pose = resolver.lookup("map", "base_link", 1000000000LL, 100000000LL);
  ASSERT_TRUE(pose.has_value());
  EXPECT_NEAR(pose->x, 11.0, 1e-6);
  EXPECT_NEAR(pose->y, 22.0, 1e-6);
}

TEST(TfPoseResolver, missingPoseReturnsNullopt)
{
  TfPoseResolver resolver;
  const auto pose = resolver.lookup("map", "base_link", 0, 1000);
  EXPECT_FALSE(pose.has_value());
}

}  // namespace
}  // namespace lie_lane_detection
