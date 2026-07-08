#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <random>

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

// The analytic (y-inversion + Newton) distanceToCurve must match a dense
// brute-force reference across the parameter range the pipeline explores.
TEST(TemplateCurveTest, AnalyticMatchesDenseBruteForce)
{
  lie_lane_detection::PipelineParams params;
  lie_lane_detection::TemplateCurve curve(params);
  curve.setBevExtents(0.0, 200.0, -100.0, 100.0);

  auto denseDistance = [&](const lie_lane_detection::XiVector & xi,
    const lie_lane_detection::Vec2 & p) {
      double best = std::numeric_limits<double>::max();
      constexpr int kSteps = 4000;
      for (int i = 0; i <= kSteps; ++i) {
        const double t = static_cast<double>(i) / kSteps;
        best = std::min(best, (p - curve.sample(xi, t)).norm());
      }
      return best;
    };

  std::mt19937 rng(123);
  std::uniform_real_distribution<double> vx(-40, 40), vy(-40, 40), om(-0.3, 0.3);
  std::uniform_real_distribution<double> ka(-0.15, 0.15), si(-0.3, 0.3);
  std::uniform_real_distribution<double> px(-60, 60), py(-40, 240);

  double max_err = 0.0;
  for (int n = 0; n < 5000; ++n) {
    lie_lane_detection::XiVector xi;
    xi << vx(rng), vy(rng), om(rng), ka(rng), si(rng);
    const lie_lane_detection::Vec2 p(px(rng), py(rng));
    max_err = std::max(max_err, std::abs(denseDistance(xi, p) - curve.distanceToCurve(xi, p)));
  }
  EXPECT_LT(max_err, 0.2) << "analytic distanceToCurve deviates from dense reference";
}
