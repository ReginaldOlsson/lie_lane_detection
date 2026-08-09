#include "lie_lane_detection/pipeline/line_lane_detection_pipeline.hpp"

#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/pipeline/line_lane_detection_runner.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"

namespace lie_lane_detection
{

LineLaneDetectionPipeline::LineLaneDetectionPipeline(const PipelineParams & params)
: params_(params)
{
}

void LineLaneDetectionPipeline::updateParams(const PipelineParams & params)
{
  params_ = params;
}

LineLaneDetectionResult LineLaneDetectionPipeline::detect(
  const cv::Mat & image_bgr, const sensor_msgs::msg::CameraInfo * camera_info)
{
  LineLaneDetectionResult result;

  IPMTransformer ipm(params_);
  ipm.computeHomography(camera_info);
  cv::Mat bev = ipm.warpToBev(image_bgr, &result.H_img2bev);
  if (bev.empty()) {
    return result;
  }

  cv::Mat bev_prepared = prepareBevImage(bev);
  PipelineParams bev_params = params_;
  configureParamsForBev(bev_params, bev_prepared.cols, bev_prepared.rows);

  const BevDetectionResult bev_result = detectLanesInBevFromLines(bev_prepared, bev_params);
  result.lanes = bev_result.lanes;
  result.merges = bev_result.merges;
  result.line_segment_count = bev_result.line_segment_count;
  result.edge_point_count = bev_result.edge_point_count;
  result.elapsed_ms = bev_result.elapsed_ms;
  result.line_hough_ms = bev_result.line_hough_ms;
  result.lie_vote_ms = bev_result.lie_vote_ms;
  result.debug.bev = bev.clone();
  result.debug.edges = bev_result.edges;
  result.debug.hough_slice = bev_result.hough_slice;
  result.debug.overlay = bev_result.overlay;

  return result;
}

}  // namespace lie_lane_detection
