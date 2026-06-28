#include <gtest/gtest.h>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "lie_lane_detection/motion/ego_motion_estimator.hpp"

TEST(EgoMotionEstimator, firstFrameInvalid)
{
  lie_lane_detection::EgoMotionEstimator estimator;
  cv::Mat frame(200, 320, CV_8UC3, cv::Scalar(30, 30, 30));
  const auto ego = estimator.update(frame);
  EXPECT_FALSE(ego.valid);
}

TEST(EgoMotionEstimator, detectsLateralShift)
{
  lie_lane_detection::EgoMotionEstimator estimator;
  cv::Mat frame0(240, 320, CV_8UC3, cv::Scalar(40, 40, 40));
  cv::Mat texture(120, 320, CV_8UC1);
  cv::randu(texture, 0, 255);
  cv::Mat road_roi = frame0(cv::Rect(0, 120, 320, 120));
  cv::cvtColor(texture, road_roi, cv::COLOR_GRAY2BGR);

  estimator.update(frame0);

  cv::Mat frame1 = frame0.clone();
  cv::Mat shifted_texture;
  cv::Mat shift = (cv::Mat_<double>(2, 3) << 1, 0, 12, 0, 1, 0);
  cv::warpAffine(texture, shifted_texture, shift, texture.size());
  cv::cvtColor(shifted_texture, frame1(cv::Rect(0, 120, 320, 120)), cv::COLOR_GRAY2BGR);

  const auto ego = estimator.update(frame1);
  EXPECT_TRUE(ego.valid);
  EXPECT_GT(ego.delta_image_x, 5.0);
}
