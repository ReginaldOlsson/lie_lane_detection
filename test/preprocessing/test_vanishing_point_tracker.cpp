#include <gtest/gtest.h>

#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"

TEST(VanishingPointTracker, rejectsLargeJump)
{
  lie_lane_detection::VanishingPointTracker tracker;
  lie_lane_detection::VanishingPointEstimate m0;
  m0.x = 320.0;
  m0.y = 150.0;
  m0.confidence = 0.8;
  m0.valid = true;
  tracker.update(m0);
  tracker.update(m0);

  lie_lane_detection::VanishingPointEstimate m1;
  m1.x = 400.0;
  m1.y = 150.0;
  m1.confidence = 0.3;
  m1.valid = true;
  const auto out = tracker.update(m1);
  EXPECT_NEAR(out.x, 320.0, 20.0);
}

TEST(VanishingPointTracker, smoothsValidMeasurements)
{
  lie_lane_detection::VanishingPointTracker tracker;
  lie_lane_detection::VanishingPointEstimate m0;
  m0.x = 300.0;
  m0.y = 140.0;
  m0.confidence = 0.9;
  m0.valid = true;
  tracker.update(m0);
  tracker.update(m0);

  lie_lane_detection::VanishingPointEstimate m1;
  m1.x = 310.0;
  m1.y = 145.0;
  m1.confidence = 0.8;
  m1.valid = true;
  const auto out = tracker.update(m1);
  EXPECT_GT(out.x, 300.0);
  EXPECT_LT(out.x, 310.0);
}
