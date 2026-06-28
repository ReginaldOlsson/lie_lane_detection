#include <chrono>
#include <memory>
#include <sstream>
#include <string>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include "lie_lane_detection/mosaic/bev_mosaic_accumulator.hpp"

namespace lie_lane_detection
{
namespace
{

BevRegistrationMethod parseRegistrationMethod(const std::string & value)
{
  if (value == "ecc") {
    return BevRegistrationMethod::ECC;
  }
  if (value == "features") {
    return BevRegistrationMethod::FEATURES;
  }
  return BevRegistrationMethod::AUTO;
}

BevMosaicParams loadMosaicParams(rclcpp::Node & node)
{
  BevMosaicParams p;
  p.canvas_margin_px = node.declare_parameter<int>("canvas_margin_px", 400);
  p.min_ecc_correlation = node.declare_parameter<double>("min_ecc_correlation", 0.35);
  p.ecc_max_iterations = node.declare_parameter<int>("ecc_max_iterations", 50);
  p.ecc_epsilon = node.declare_parameter<double>("ecc_epsilon", 1e-5);
  p.orb_features = node.declare_parameter<int>("orb_features", 1200);
  p.feature_match_ratio = node.declare_parameter<double>("feature_match_ratio", 0.75);
  p.feature_min_inliers = node.declare_parameter<int>("feature_min_inliers", 12);
  p.max_step_translation_px = node.declare_parameter<double>("max_step_translation_px", 80.0);
  p.max_step_rotation_rad = node.declare_parameter<double>("max_step_rotation_rad", 0.12);
  p.use_constant_velocity_fallback =
    node.declare_parameter<bool>("use_constant_velocity_fallback", true);
  p.registration_method =
    parseRegistrationMethod(node.declare_parameter<std::string>("registration_method", "auto"));
  return p;
}

}  // namespace

class BevMosaicNode : public rclcpp::Node
{
public:
  BevMosaicNode()
  : Node("bev_mosaic_node"),
    params_(loadMosaicParams(*this)),
    accumulator_(params_),
    frame_id_(declare_parameter<std::string>("frame_id", "bev_mosaic"))
  {
    const std::string bev_topic = declare_parameter<std::string>("bev_topic", "/ipm/bev");
    const std::string canvas_topic =
      declare_parameter<std::string>("canvas_topic", "/ipm/mosaic/canvas");
    const std::string aligned_topic =
      declare_parameter<std::string>("aligned_topic", "/ipm/mosaic/aligned");
    const std::string pose_topic =
      declare_parameter<std::string>("pose_topic", "/ipm/mosaic/global_pose");
    const std::string stats_topic =
      declare_parameter<std::string>("stats_topic", "/ipm/mosaic/stats");
    reset_on_first_frame_ = declare_parameter<bool>("reset_on_first_frame", false);

    canvas_pub_ = image_transport::create_publisher(this, canvas_topic);
    aligned_pub_ = image_transport::create_publisher(this, aligned_topic);
    pose_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(pose_topic, 10);
    stats_pub_ = create_publisher<std_msgs::msg::String>(stats_topic, 10);

    bev_sub_ = image_transport::create_subscription(
      this, bev_topic,
      std::bind(&BevMosaicNode::onBevImage, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    RCLCPP_INFO(get_logger(), "bev_mosaic_node on %s", bev_topic.c_str());
    RCLCPP_INFO(
      get_logger(),
      "  publish: %s, %s, %s",
      canvas_topic.c_str(), aligned_topic.c_str(), pose_topic.c_str());
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

  void publishPose(const cv::Mat & global_2x3)
  {
    if (global_2x3.empty()) {
      return;
    }
    cv::Mat pose_3x3 = cv::Mat::eye(3, 3, CV_64F);
    cv::Mat converted;
    global_2x3.convertTo(converted, CV_64F);
    converted.copyTo(pose_3x3(cv::Rect(0, 0, 3, 2)));

    std_msgs::msg::Float64MultiArray msg;
    msg.layout.dim.resize(2);
    msg.layout.dim[0].label = "row";
    msg.layout.dim[0].size = 3;
    msg.layout.dim[0].stride = 9;
    msg.layout.dim[1].label = "col";
    msg.layout.dim[1].size = 3;
    msg.layout.dim[1].stride = 3;
    msg.data.resize(9);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        msg.data[static_cast<size_t>(r * 3 + c)] = pose_3x3.at<double>(r, c);
      }
    }
    pose_pub_->publish(msg);
  }

  static std::string formatStats(const BevRegistrationResult & reg, double ms)
  {
    std::ostringstream oss;
    oss << "valid=" << (reg.valid ? 1 : 0)
        << " ecc=" << (reg.used_ecc ? 1 : 0)
        << " features=" << (reg.used_features ? 1 : 0)
        << " fallback=" << (reg.used_fallback ? 1 : 0)
        << " corr=" << reg.correlation
        << " ms=" << ms;
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

    if (reset_on_first_frame_ && frame_index_ == 0) {
      accumulator_.reset();
    }

    const auto t0 = std::chrono::steady_clock::now();
    const BevRegistrationResult reg = accumulator_.accumulate(cv_ptr->image);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std_msgs::msg::Header header = msg->header;
    header.frame_id = frame_id_;

    publishCvImage(canvas_pub_, accumulator_.canvas(), header);
    publishCvImage(aligned_pub_, accumulator_.lastAlignedFrame(), header);
    publishPose(reg.global_affine_2x3);

    std_msgs::msg::String stats;
    stats.data = formatStats(reg, ms);
    stats_pub_->publish(stats);

    ++frame_index_;
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "MOSAIC frame=%d | %s",
      frame_index_, stats.data.c_str());
  }

  BevMosaicParams params_;
  BevMosaicAccumulator accumulator_;
  std::string frame_id_;
  bool reset_on_first_frame_{false};
  int frame_index_{0};

  image_transport::Subscriber bev_sub_;
  image_transport::Publisher canvas_pub_;
  image_transport::Publisher aligned_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pose_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr stats_pub_;
};

}  // namespace lie_lane_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lie_lane_detection::BevMosaicNode>());
  rclcpp::shutdown();
  return 0;
}
