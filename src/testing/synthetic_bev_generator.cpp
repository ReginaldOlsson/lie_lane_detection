#include "lie_lane_detection/testing/synthetic_bev_generator.hpp"

#include "lie_lane_detection/geometry/template_curve.hpp"

#include <opencv2/imgproc.hpp>

#include <array>
#include <random>

namespace lie_lane_detection
{

SyntheticBEVGenerator::SyntheticBEVGenerator(const PipelineParams & params) : params_(params)
{
}

std::string SyntheticBEVGenerator::scenarioName(SyntheticScenario scenario)
{
  switch (scenario) {
    case SyntheticScenario::ParallelStraight3:
      return "01_parallel_straight_3lane";
    case SyntheticScenario::CurvedHighway:
      return "02_curved_highway";
    case SyntheticScenario::YMerge:
      return "03_y_merge";
    case SyntheticScenario::FourLanes:
      return "04_four_lanes";
    case SyntheticScenario::DashedLanes:
      return "05_dashed_lanes";
    case SyntheticScenario::PartialOcclusion:
      return "06_partial_occlusion";
    case SyntheticScenario::IpmStyleSingle:
      return "07_ipm_style_single_lane";
    case SyntheticScenario::SplitDiverge:
      return "08_split_diverge";
    case SyntheticScenario::CombinedChallenge:
      return "09_combined_challenge";
    default:
      return "unknown";
  }
}

cv::Mat SyntheticBEVGenerator::blankCanvas(const SyntheticSceneSpec & spec) const
{
  return cv::Mat(spec.height, spec.width, CV_8UC3, spec.road_color);
}

void SyntheticBEVGenerator::drawLane(
  cv::Mat & canvas, const XiVector & xi, const SyntheticSceneSpec & spec) const
{
  if (spec.dashed) {
    drawDashedLane(canvas, xi, spec);
    return;
  }

  TemplateCurve curve(params_);
  curve.setBevExtents(0.0, static_cast<double>(spec.height), 0.0, static_cast<double>(spec.width));
  const auto poly = curve.samplePolyline(xi, 100);
  for (size_t i = 1; i < poly.size(); ++i) {
    cv::line(
      canvas, cv::Point(static_cast<int>(poly[i - 1].x()), static_cast<int>(poly[i - 1].y())),
      cv::Point(static_cast<int>(poly[i].x()), static_cast<int>(poly[i].y())), spec.lane_color,
      spec.lane_thickness);
  }
}

void SyntheticBEVGenerator::drawDashedLane(
  cv::Mat & canvas, const XiVector & xi, const SyntheticSceneSpec & spec) const
{
  TemplateCurve curve(params_);
  curve.setBevExtents(0.0, static_cast<double>(spec.height), 0.0, static_cast<double>(spec.width));
  const auto poly = curve.samplePolyline(xi, 200);
  bool draw = true;
  int run = 0;
  for (size_t i = 1; i < poly.size(); ++i) {
    const cv::Point p0(static_cast<int>(poly[i - 1].x()), static_cast<int>(poly[i - 1].y()));
    const cv::Point p1(static_cast<int>(poly[i].x()), static_cast<int>(poly[i].y()));
    const int seg_len = std::max(1, static_cast<int>(cv::norm(p1 - p0)));
    run += seg_len;
    if (draw) {
      cv::line(canvas, p0, p1, spec.lane_color, spec.lane_thickness);
    }
    if (run >= (draw ? spec.dash_length : spec.dash_gap)) {
      run = 0;
      draw = !draw;
    }
  }
}

void SyntheticBEVGenerator::applyIpmMask(cv::Mat & canvas) const
{
  const int h = canvas.rows;
  const int w = canvas.cols;
  std::vector<cv::Point> left_poly = {
    cv::Point(0, h), cv::Point(0, h - 1), cv::Point(w / 4, h - 1), cv::Point(w / 4, h)};
  std::vector<cv::Point> right_poly = {
    cv::Point(w, h), cv::Point(w, h - 1), cv::Point(3 * w / 4, h - 1), cv::Point(3 * w / 4, h)};
  cv::fillConvexPoly(canvas, left_poly, cv::Scalar(0, 0, 0));
  cv::fillConvexPoly(canvas, right_poly, cv::Scalar(0, 0, 0));
}

void SyntheticBEVGenerator::applyNoise(cv::Mat & canvas, const SyntheticSceneSpec & spec) const
{
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  const int n = static_cast<int>(spec.width * spec.height * spec.noise_density);
  for (int i = 0; i < n; ++i) {
    const int x = static_cast<int>(uni(rng) * (spec.width - 1));
    const int y = static_cast<int>(uni(rng) * (spec.height - 1));
    const int v = static_cast<int>(uni(rng) * 80) + 120;
    canvas.at<cv::Vec3b>(y, x) = cv::Vec3b(v, v, v);
  }
}

void SyntheticBEVGenerator::drawOcclusionBar(cv::Mat & canvas, int y0, int y1, int x0, int x1) const
{
  cv::rectangle(canvas, cv::Point(x0, y0), cv::Point(x1, y1), cv::Scalar(25, 25, 25), cv::FILLED);
}

GeneratedScene SyntheticBEVGenerator::generate(
  SyntheticScenario scenario, const SyntheticSceneSpec & spec_in) const
{
  SyntheticSceneSpec spec = spec_in;
  if (spec.width <= 0) {
    spec.width = 240;
  }
  if (spec.height <= 0) {
    spec.height = 800;
  }

  GeneratedScene scene;
  scene.name = scenarioName(scenario);
  scene.image = blankCanvas(spec);

  switch (scenario) {
    case SyntheticScenario::ParallelStraight3: {
      scene.description = "Three parallel straight lane boundaries";
      const std::array<double, 3> vx = {spec.width * 0.25, spec.width * 0.5, spec.width * 0.75};
      for (double v : vx) {
        XiVector xi = XiVector::Zero();
        xi[0] = v;
        scene.ground_truth_xi.push_back(xi);
        drawLane(scene.image, xi, spec);
      }
      break;
    }
    case SyntheticScenario::CurvedHighway: {
      scene.description = "Two curved lane boundaries (kappa + slight omega)";
      XiVector left = XiVector::Zero();
      left[0] = spec.width * 0.3;
      left[2] = 0.12;
      left[3] = 0.08;
      XiVector right = XiVector::Zero();
      right[0] = spec.width * 0.7;
      right[2] = 0.12;
      right[3] = 0.08;
      scene.ground_truth_xi = {left, right};
      drawLane(scene.image, left, spec);
      drawLane(scene.image, right, spec);
      break;
    }
    case SyntheticScenario::YMerge: {
      scene.description = "Y-merge: inner boundaries converge toward far field";
      XiVector left_outer = XiVector::Zero();
      left_outer[0] = spec.width * 0.15;
      XiVector left_inner = XiVector::Zero();
      left_inner[0] = spec.width * 0.4;
      left_inner[4] = 0.1;
      XiVector right_inner = XiVector::Zero();
      right_inner[0] = spec.width * 0.6;
      right_inner[4] = -0.1;
      XiVector right_outer = XiVector::Zero();
      right_outer[0] = spec.width * 0.85;
      scene.ground_truth_xi = {left_outer, left_inner, right_inner, right_outer};
      for (const auto & xi : scene.ground_truth_xi) {
        drawLane(scene.image, xi, spec);
      }
      break;
    }
    case SyntheticScenario::FourLanes: {
      scene.description = "Four-lane highway (5 boundaries)";
      for (int i = 1; i <= 5; ++i) {
        XiVector xi = XiVector::Zero();
        xi[0] = spec.width * static_cast<double>(i) / 6.0;
        scene.ground_truth_xi.push_back(xi);
        drawLane(scene.image, xi, spec);
      }
      break;
    }
    case SyntheticScenario::DashedLanes: {
      scene.description = "Single lane with dashed left/right markings";
      spec.dashed = true;
      XiVector left = XiVector::Zero();
      left[0] = spec.width * 0.35;
      XiVector right = XiVector::Zero();
      right[0] = spec.width * 0.65;
      scene.ground_truth_xi = {left, right};
      drawLane(scene.image, left, spec);
      drawLane(scene.image, right, spec);
      // forward arrow
      const int cx = spec.width / 2;
      const int by = spec.height - 30;
      cv::arrowedLine(
        scene.image, cv::Point(cx, by), cv::Point(cx, by - 40), cv::Scalar(200, 200, 200), 2,
        cv::LINE_AA, 0, 0.35);
      break;
    }
    case SyntheticScenario::PartialOcclusion: {
      scene.description = "Three straight lanes with shadow/occlusion band";
      const std::array<double, 3> vx = {spec.width * 0.25, spec.width * 0.5, spec.width * 0.75};
      for (double v : vx) {
        XiVector xi = XiVector::Zero();
        xi[0] = v;
        scene.ground_truth_xi.push_back(xi);
        drawLane(scene.image, xi, spec);
      }
      drawOcclusionBar(scene.image, spec.height / 3, spec.height / 2, 0, spec.width);
      spec.add_noise = true;
      break;
    }
    case SyntheticScenario::IpmStyleSingle: {
      scene.description = "IPM-like single lane, dashed, black corner mask";
      spec.dashed = true;
      spec.ipm_black_corners = true;
      spec.width = 412;
      spec.height = 412;
      scene.image = blankCanvas(spec);
      XiVector left = XiVector::Zero();
      left[0] = spec.width * 0.28;
      XiVector right = XiVector::Zero();
      right[0] = spec.width * 0.72;
      scene.ground_truth_xi = {left, right};
      drawLane(scene.image, left, spec);
      drawLane(scene.image, right, spec);
      const int cx = spec.width / 2;
      cv::arrowedLine(
        scene.image, cv::Point(cx, spec.height - 25), cv::Point(cx, spec.height - 65),
        cv::Scalar(200, 200, 200), 2, cv::LINE_AA, 0, 0.35);
      applyIpmMask(scene.image);
      // side clutter (curbs)
      cv::rectangle(
        scene.image, cv::Point(0, 0), cv::Point(25, spec.height), cv::Scalar(15, 15, 15),
        cv::FILLED);
      cv::rectangle(
        scene.image, cv::Point(spec.width - 25, 0), cv::Point(spec.width, spec.height),
        cv::Scalar(15, 15, 15), cv::FILLED);
      break;
    }
    case SyntheticScenario::SplitDiverge: {
      scene.description = "Lane split: boundaries diverge from shared near field";
      XiVector left = XiVector::Zero();
      left[0] = spec.width * 0.45;
      left[4] = -0.12;
      XiVector right = XiVector::Zero();
      right[0] = spec.width * 0.55;
      right[4] = 0.12;
      scene.ground_truth_xi = {left, right};
      drawLane(scene.image, left, spec);
      drawLane(scene.image, right, spec);
      break;
    }
    case SyntheticScenario::CombinedChallenge: {
      scene.description = "Curved 3-lane road + occlusion + dashed center + noise";
      spec.dashed = false;
      for (int i = 1; i <= 3; ++i) {
        XiVector xi = XiVector::Zero();
        xi[0] = spec.width * static_cast<double>(i) / 4.0;
        xi[2] = 0.08;
        xi[3] = 0.05;
        scene.ground_truth_xi.push_back(xi);
        drawLane(scene.image, xi, spec);
      }
      // dashed center line overlay
      SyntheticSceneSpec dashed_spec = spec;
      dashed_spec.dashed = true;
      XiVector center = XiVector::Zero();
      center[0] = spec.width * 0.5;
      center[2] = 0.08;
      center[3] = 0.05;
      drawLane(scene.image, center, dashed_spec);
      drawOcclusionBar(
        scene.image, spec.height / 2, spec.height * 2 / 3, spec.width / 6, spec.width * 5 / 6);
      spec.add_noise = true;
      break;
    }
  }

  if (spec.add_noise) {
    applyNoise(scene.image, spec);
  }
  if (spec.ipm_black_corners && scenario != SyntheticScenario::IpmStyleSingle) {
    applyIpmMask(scene.image);
  }

  return scene;
}

std::vector<GeneratedScene> SyntheticBEVGenerator::generateAll(
  const SyntheticSceneSpec & spec, const std::vector<SyntheticScenario> & scenarios) const
{
  std::vector<SyntheticScenario> list = scenarios;
  if (list.empty()) {
    list = {
      SyntheticScenario::ParallelStraight3,
      SyntheticScenario::CurvedHighway,
      SyntheticScenario::YMerge,
      SyntheticScenario::FourLanes,
      SyntheticScenario::DashedLanes,
      SyntheticScenario::PartialOcclusion,
      SyntheticScenario::IpmStyleSingle,
      SyntheticScenario::SplitDiverge,
      SyntheticScenario::CombinedChallenge};
  }

  std::vector<GeneratedScene> out;
  out.reserve(list.size());
  for (const auto s : list) {
    out.push_back(generate(s, spec));
  }
  return out;
}

}  // namespace lie_lane_detection
