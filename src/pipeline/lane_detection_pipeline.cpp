#include "lie_lane_detection/pipeline/lane_detection_pipeline.hpp"

#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include<opencv2/opencv.hpp>

namespace lie_lane_detection
{

LaneDetectionPipeline::LaneDetectionPipeline(const PipelineParams & params)
: params_(params),
  ipm_(params_),
  edge_extractor_(params_),
  template_curve_(params_),
  hough_voter_(params_, &template_curve_),
  ransac_(params_, &template_curve_),
  multi_lane_(params_),
  merge_topology_(params_, &template_curve_)
{
}

void LaneDetectionPipeline::updateParams(const PipelineParams & params)
{
  params_ = params;
  ipm_.updateParams(params_);
  edge_extractor_ = EdgeExtractor(params_);
  template_curve_ = TemplateCurve(params_);
  hough_voter_.setTemplateCurve(&template_curve_);
  ransac_.setTemplateCurve(&template_curve_);
  multi_lane_ = MultiLaneExtractor(params_);
  merge_topology_.setTemplateCurve(&template_curve_);
}

LaneDetectionResult LaneDetectionPipeline::detect(
  const cv::Mat & image_bgr,
  const sensor_msgs::msg::CameraInfo * camera_info)
{
  LaneDetectionResult result;

  ipm_.computeHomography(camera_info);
  cv::Mat bev = ipm_.warpToBev(image_bgr, &result.H_img2bev);
  if (bev.empty()) {
    return result;
  }

  cv::Mat bev_prepared = prepareBevImage(bev);
  PipelineParams bev_params = params_;
  configureParamsForBev(bev_params, bev_prepared.cols, bev_prepared.rows);
  cv::imshow("BEV Prepared", bev_prepared);
  cv::waitKey(1);
  const BevDetectionResult bev_result = detectLanesInBev(bev_prepared, bev_params);
  result.lanes = bev_result.lanes;
  result.merges = bev_result.merges;
  result.debug.bev = bev.clone();
  result.debug.edges = bev_result.edges;
  result.debug.hough_slice = bev_result.hough_slice;
  result.debug.overlay = bev_result.overlay;

  return result;
}

}  // namespace lie_lane_detection
