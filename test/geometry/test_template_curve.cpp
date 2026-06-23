#include <gtest/gtest.h>

#include "lie_lane_detection/geometry/template_curve.hpp"

TEST(TemplateCurveTest, StraightLineDistance)
{
  lie_lane_detection::PipelineParams params;
  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 100.0, -50.0, 50.0);

  lie_lane_detection::XiVector xi = lie_lane_detection::XiVector::Zero();
  xi[0] = 10.0;

  const lie_lane_detection::Vec2 p(10.0, 50.0);
  const double d = curve.distanceToCurve(xi, p);
  EXPECT_LT(d, 1.0);
}

TEST(TemplateCurveTest, CurvedTemplateNonZeroKappa)
{
  lie_lane_detection::PipelineParams params;
  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 1.0, -1.0, 1.0);

  lie_lane_detection::XiVector xi = lie_lane_detection::XiVector::Zero();
  xi[3] = 0.05;

  const lie_lane_detection::Vec2 on_curve = curve.sample(xi, 0.5);
  const double d = curve.distanceToCurve(xi, on_curve);
  EXPECT_LT(d, 1e-3);
}
