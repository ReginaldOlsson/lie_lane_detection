#pragma once

#include "lie_lane_detection/core/types.hpp"

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace lie_lane_detection
{

enum class SyntheticScenario {
  ParallelStraight3,
  CurvedHighway,
  YMerge,
  FourLanes,
  DashedLanes,
  PartialOcclusion,
  IpmStyleSingle,
  SplitDiverge,
  CombinedChallenge
};

struct SyntheticSceneSpec
{
  int width{240};
  int height{800};
  cv::Scalar road_color{40, 40, 40};
  cv::Scalar lane_color{220, 220, 220};
  int lane_thickness{3};
  bool dashed{false};
  int dash_length{18};
  int dash_gap{14};
  bool ipm_black_corners{false};
  bool add_noise{false};
  double noise_density{0.002};
};

struct GeneratedScene
{
  std::string name;
  std::string description;
  cv::Mat image;
  std::vector<XiVector> ground_truth_xi;
};

class SyntheticBEVGenerator
{
public:
  explicit SyntheticBEVGenerator(const PipelineParams & params = PipelineParams{});

  GeneratedScene generate(SyntheticScenario scenario, const SyntheticSceneSpec & spec = {}) const;

  std::vector<GeneratedScene> generateAll(
    const SyntheticSceneSpec & spec = {},
    const std::vector<SyntheticScenario> & scenarios = {}) const;

  static std::string scenarioName(SyntheticScenario scenario);

private:
  cv::Mat blankCanvas(const SyntheticSceneSpec & spec) const;

  void drawLane(cv::Mat & canvas, const XiVector & xi, const SyntheticSceneSpec & spec) const;

  void drawDashedLane(cv::Mat & canvas, const XiVector & xi, const SyntheticSceneSpec & spec) const;

  void applyIpmMask(cv::Mat & canvas) const;

  void applyNoise(cv::Mat & canvas, const SyntheticSceneSpec & spec) const;

  void drawOcclusionBar(cv::Mat & canvas, int y0, int y1, int x0, int x1) const;

  PipelineParams params_;
};

}  // namespace lie_lane_detection
