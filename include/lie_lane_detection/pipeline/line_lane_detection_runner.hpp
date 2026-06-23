#pragma once

#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"

namespace lie_lane_detection
{

/// Line-first pipeline: OpenCV HoughLinesP → Line-Lie-Hough → RANSAC refine on edges.
BevDetectionResult detectLanesInBevFromLines(const cv::Mat & bev_bgr, PipelineParams params);

}  // namespace lie_lane_detection
