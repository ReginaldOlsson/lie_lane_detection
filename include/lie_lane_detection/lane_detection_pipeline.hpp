#pragma once

#include <opencv2/core.hpp>
#include <sensor_msgs/msg/camera_info.hpp>

#include "lie_lane_detection/edge_extractor.hpp"
#include "lie_lane_detection/ipm_transformer.hpp"
#include "lie_lane_detection/lie_hough_voter.hpp"
#include "lie_lane_detection/manifold_ransac.hpp"
#include "lie_lane_detection/merge_topology.hpp"
#include "lie_lane_detection/multi_lane_extractor.hpp"
#include "lie_lane_detection/template_curve.hpp"
#include "lie_lane_detection/types.hpp"

namespace lie_lane_detection
{

class LaneDetectionPipeline
{
public:
  explicit LaneDetectionPipeline(const PipelineParams & params);

  void updateParams(const PipelineParams & params);

  LaneDetectionResult detect(
    const cv::Mat & image_bgr,
    const sensor_msgs::msg::CameraInfo * camera_info = nullptr);

  const IPMTransformer & ipm() const {return ipm_;}

private:
  PipelineParams params_;
  IPMTransformer ipm_;
  EdgeExtractor edge_extractor_;
  TemplateCurve template_curve_;
  LieHoughVoter hough_voter_;
  ManifoldRansac ransac_;
  MultiLaneExtractor multi_lane_;
  MergeTopology merge_topology_;
};

}  // namespace lie_lane_detection
