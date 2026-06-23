#include <memory>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "lie_lane_detection/pipeline/line_lane_detection_pipeline.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"

namespace lie_lane_detection
{

PipelineParams loadLineParams(rclcpp::Node & node)
{
  PipelineParams p;
  p.bev_width_m = node.declare_parameter<double>("bev_width_m", 12.0);
  p.bev_length_m = node.declare_parameter<double>("bev_length_m", 40.0);
  p.bev_resolution_m_per_px = node.declare_parameter<double>("bev_resolution_m_per_px", 0.05);
  p.ipm_src_points = node.declare_parameter<std::vector<double>>(
    "ipm_src_points", {200.0, 700.0, 600.0, 700.0, 700.0, 400.0, 100.0, 400.0});
  p.ipm_dst_points = node.declare_parameter<std::vector<double>>(
    "ipm_dst_points", {-6.0, 0.0, 6.0, 0.0, 6.0, 40.0, -6.0, 40.0});
  p.use_steerable_filter = node.declare_parameter<bool>("use_steerable_filter", true);
  p.connect_dashed_edges = node.declare_parameter<bool>("connect_dashed_edges", true);
  p.edge_anisotropic_blur = node.declare_parameter<bool>("edge_anisotropic_blur", true);
  p.edge_thin = node.declare_parameter<bool>("edge_thin", true);
  p.edge_low_threshold = node.declare_parameter<double>("edge_low_threshold", 30.0);
  p.edge_high_threshold = node.declare_parameter<double>("edge_high_threshold", 90.0);
  p.top_k_peaks = node.declare_parameter<int>("top_k_peaks", 6);
  p.vote_threshold_px = node.declare_parameter<double>("vote_threshold_px", 4.0);
  p.inlier_threshold_px = node.declare_parameter<double>("inlier_threshold_px", 5.0);
  p.ransac_iterations = node.declare_parameter<int>("ransac_iterations", 100);
  p.min_inliers = node.declare_parameter<int>("min_inliers", 15);
  p.kappa_min = node.declare_parameter<double>("kappa_min", -0.2);
  p.kappa_max = node.declare_parameter<double>("kappa_max", 0.2);
  p.sigma_min = node.declare_parameter<double>("sigma_min", -0.15);
  p.sigma_max = node.declare_parameter<double>("sigma_max", 0.15);
  p.hough_hypothesis_merge_ratio =
    node.declare_parameter<double>("hough_hypothesis_merge_ratio", 0.85);
  p.min_inlier_y_coverage = node.declare_parameter<double>("min_inlier_y_coverage", 0.35);
  p.line_hough_threshold = node.declare_parameter<int>("line_hough_threshold", 25);
  p.line_min_length_px = node.declare_parameter<double>("line_min_length_px", 18.0);
  p.line_max_gap_px = node.declare_parameter<double>("line_max_gap_px", 12.0);
  p.line_angle_threshold_rad = node.declare_parameter<double>("line_angle_threshold_rad", 0.45);
  return p;
}

class LineLaneDetectorNode : public rclcpp::Node
{
public:
  LineLaneDetectorNode()
  : Node("line_lane_detector_node"),
    params_(loadLineParams(*this)),
    pipeline_(params_),
    frame_id_(declare_parameter<std::string>("frame_id", "base_link"))
  {
    const std::string image_topic = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    const std::string camera_info_topic =
      declare_parameter<std::string>("camera_info_topic", "/camera/camera_info");

    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>("/lanes_line/markers", 10);
    merge_marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes_line/merge_markers", 10);
    bev_pub_ = image_transport::create_publisher(this, "/lanes_line/debug/bev");
    overlay_pub_ = image_transport::create_publisher(this, "/lanes_line/debug/overlay");
    hough_pub_ = image_transport::create_publisher(this, "/lanes_line/debug/hough");
    lines_pub_ = image_transport::create_publisher(this, "/lanes_line/debug/lines");

    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::CameraInfo::SharedPtr msg) {
        latest_camera_info_ = msg;
      });

    image_sub_ = image_transport::create_subscription(
      this, image_topic,
      std::bind(&LineLaneDetectorNode::onImage, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    stats_pub_ = create_publisher<std_msgs::msg::String>("/lanes_line/stats", 10);
  }

private:
  void publishCvImage(
    const image_transport::Publisher & pub,
    const cv::Mat & mat,
    const std_msgs::msg::Header & header)
  {
    if (mat.empty()) {
      return;
    }
    cv_bridge::CvImage cv_image(header, mat.channels() == 1 ? "mono8" : "bgr8", mat);
    pub.publish(cv_image.toImageMsg());
  }

  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
      return;
    }

    const sensor_msgs::msg::CameraInfo * info_ptr =
      latest_camera_info_ ? latest_camera_info_.get() : nullptr;

    const LineLaneDetectionResult result = pipeline_.detect(cv_ptr->image, info_ptr);
    const auto stamp = msg->header.stamp;

    marker_pub_->publish(lanesToMarkers(result.lanes, frame_id_, stamp));
    merge_marker_pub_->publish(mergesToMarkers(result.merges, frame_id_, stamp));

    publishCvImage(bev_pub_, result.debug.bev, msg->header);
    publishCvImage(overlay_pub_, result.debug.overlay, msg->header);
    publishCvImage(hough_pub_, result.debug.hough_slice, msg->header);
    publishCvImage(lines_pub_, result.debug.edges, msg->header);

    std_msgs::msg::String stats;
    stats.data =
      "lines=" + std::to_string(result.line_segment_count) +
      " edges=" + std::to_string(result.edge_point_count) +
      " lanes=" + std::to_string(result.lanes.size()) +
      " ms=" + std::to_string(result.elapsed_ms) +
      " line_hough_ms=" + std::to_string(result.line_hough_ms) +
      " lie_vote_ms=" + std::to_string(result.lie_vote_ms);
    stats_pub_->publish(stats);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "Line pipeline: %zu lanes, %zu segments, %.1f ms (hough %.1f, vote %.1f)",
      result.lanes.size(), result.line_segment_count, result.elapsed_ms,
      result.line_hough_ms, result.lie_vote_ms);
  }

  PipelineParams params_;
  LineLaneDetectionPipeline pipeline_;
  std::string frame_id_;
  sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr merge_marker_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr stats_pub_;
  image_transport::Publisher bev_pub_;
  image_transport::Publisher overlay_pub_;
  image_transport::Publisher hough_pub_;
  image_transport::Publisher lines_pub_;
};

}  // namespace lie_lane_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lie_lane_detection::LineLaneDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
