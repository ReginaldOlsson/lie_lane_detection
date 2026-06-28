#include <chrono>
#include <memory>
#include <sstream>
#include <string>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include "lie_lane_detection/nodes/node_params.hpp"
#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"

namespace lie_lane_detection
{

class BevLaneDetectorNode : public rclcpp::Node
{
public:
  BevLaneDetectorNode()
  : Node("bev_lane_detector_node"),
    params_(loadDetectionParams(*this)),
    frame_id_(declare_parameter<std::string>("frame_id", "camera_front"))
  {
    const std::string bev_topic = declare_parameter<std::string>("bev_topic", "/ipm/bev");

    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/detect/markers", 10);
    merge_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/detect/merge_markers", 10);
    stats_pub_ = create_publisher<std_msgs::msg::String>("/lanes/detect/stats", 10);
    overlay_pub_ = image_transport::create_publisher(this, "/lanes/detect/overlay");
    edges_pub_ = image_transport::create_publisher(this, "/lanes/detect/edges");

    bev_sub_ = image_transport::create_subscription(
      this, bev_topic,
      std::bind(&BevLaneDetectorNode::onBevImage, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    RCLCPP_INFO(get_logger(), "bev_lane_detector_node on %s", bev_topic.c_str());
    RCLCPP_INFO(get_logger(), "  publish: /lanes/detect/{overlay,edges,markers,stats}");
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

  static std::string formatStats(const BevDetectionResult & result, double total_ms)
  {
    std::ostringstream oss;
    oss << "mode=bev"
        << " lanes=" << result.lanes.size()
        << " total_ms=" << total_ms
        << " detect_ms=" << result.elapsed_ms;
    return oss.str();
  }

  void onBevImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
      return;
    }

    PipelineParams bev_params = params_;
    configureParamsForBev(bev_params, cv_ptr->image.cols, cv_ptr->image.rows);
    configureParamsForPerspectiveIpm(bev_params);

    const auto t0 = std::chrono::steady_clock::now();
    const cv::Mat work_bev = prepareBevForDetection(cv_ptr->image, bev_params);
    const BevDetectionResult det = detectLanesInBev(work_bev, bev_params);
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    marker_pub_->publish(lanesToMarkers(det.lanes, frame_id_, msg->header.stamp));
    merge_pub_->publish(mergesToMarkers(det.merges, frame_id_, msg->header.stamp));
    publishCvImage(overlay_pub_, det.overlay, msg->header);
    publishCvImage(edges_pub_, det.edges, msg->header);

    std_msgs::msg::String stats;
    stats.data = formatStats(det, total_ms);
    stats_pub_->publish(stats);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "DETECT | %zu lanes | %.1f ms",
      det.lanes.size(), total_ms);
  }

  PipelineParams params_;
  std::string frame_id_;

  image_transport::Subscriber bev_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr merge_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr stats_pub_;
  image_transport::Publisher overlay_pub_;
  image_transport::Publisher edges_pub_;
};

}  // namespace lie_lane_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lie_lane_detection::BevLaneDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
