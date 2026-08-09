#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/tracking/lane_tracker.hpp"

#include <opencv2/imgproc.hpp>

#include <gtest/gtest.h>

TEST(LaneTracker, prepareBevGrayForTracking)
{
  lie_lane_detection::PipelineParams params;
  cv::Mat bgr(120, 80, CV_8UC3, cv::Scalar(30, 30, 30));
  cv::rectangle(bgr, cv::Rect(10, 10, 60, 80), cv::Scalar(200, 200, 200), cv::FILLED);

  const auto prep = lie_lane_detection::prepareBevGrayForTracking(bgr, params);
  EXPECT_EQ(prep.display_bgr.channels(), 3);
  EXPECT_EQ(prep.gray.channels(), 1);
  EXPECT_EQ(prep.gray.rows, bgr.rows);
  EXPECT_EQ(prep.gray.cols, bgr.cols);
}

TEST(LaneTracker, processFrameUsesGrayscaleInternally)
{
  lie_lane_detection::LaneTracker tracker;
  lie_lane_detection::PipelineParams params;
  cv::Mat bgr(200, 120, CV_8UC3, cv::Scalar(40, 40, 40));
  for (int x = 20; x < bgr.cols; x += 30) {
    cv::line(bgr, cv::Point(x, 0), cv::Point(x, bgr.rows - 1), cv::Scalar(220, 220, 220), 2);
  }

  const auto result = tracker.processFrame(bgr, params);
  EXPECT_EQ(result.edges.channels(), 1);
  EXPECT_EQ(result.overlay.channels(), 3);
}

TEST(LaneTracker, lateralStdGrowsWithDistance)
{
  lie_lane_detection::LaneTracker tracker;
  const double near_std = tracker.lateralStdAtY(50.0, 400.0);
  const double far_std = tracker.lateralStdAtY(350.0, 400.0);
  EXPECT_GT(far_std, near_std);
}

TEST(LaneTracker, narrowParamsShrinkVxBins)
{
  lie_lane_detection::PipelineParams params;
  params.se2_vx_min = 0.0;
  params.se2_vx_max = 240.0;
  params.se2_vx_bins = 30;

  lie_lane_detection::LaneTrack track;
  track.xi[0] = 120.0;
  lie_lane_detection::narrowParamsForTracks(params, {track}, 40.0);

  EXPECT_LT(params.se2_vx_max - params.se2_vx_min, 240.0);
  EXPECT_GE(params.se2_vx_bins, 9);
  EXPECT_LT(params.se2_vx_bins, 30);
}
