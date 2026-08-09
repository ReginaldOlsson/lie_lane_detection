#include "lie_lane_detection/mosaic/bev_mosaic_accumulator.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <sstream>
#include <string>

namespace lie_lane_detection
{
namespace
{

BevRegistrationMethod parseRegistrationMethod(const std::string & value)
{
  if (value == "ecc") {
    return BevRegistrationMethod::ECC;
  }
  if (value == "orb" || value == "features") {
    return BevRegistrationMethod::ORB;
  }
  if (value == "orb_then_ecc") {
    return BevRegistrationMethod::ORB_THEN_ECC;
  }
  if (value == "ecc_then_orb") {
    return BevRegistrationMethod::ECC_THEN_ORB;
  }
  if (value == "auto") {
    return BevRegistrationMethod::ORB;
  }
  return BevRegistrationMethod::ORB;
}

BevOrbMatchMethod parseOrbMatchMethod(const std::string & value)
{
  if (value == "flann" || value == "flann_lsh") {
    return BevOrbMatchMethod::FLANN_LSH;
  }
  if (value == "lowe" || value == "lowe_bf") {
    return BevOrbMatchMethod::LOWE_BF;
  }
  return BevOrbMatchMethod::RADIUS_BF;
}

BevMosaicParams loadMosaicParams(rclcpp::Node & node)
{
  BevMosaicParams p;
  p.max_stored_frames =
    std::max(1, static_cast<int>(node.declare_parameter<int64_t>("max_stored_frames", 3)));
  p.use_constant_velocity_fallback =
    node.declare_parameter<bool>("use_constant_velocity_fallback", true);

  p.registration.method =
    parseRegistrationMethod(node.declare_parameter<std::string>("registration_method", "auto"));
  p.registration.min_ecc_correlation = node.declare_parameter<double>("min_ecc_correlation", 0.25);
  p.registration.ecc_max_iterations = node.declare_parameter<int>("ecc_max_iterations", 80);
  p.registration.ecc_epsilon = node.declare_parameter<double>("ecc_epsilon", 1e-5);
  p.registration.orb_max_features = node.declare_parameter<int>("orb_features", 1500);
  p.registration.orb_uniform_cap = node.declare_parameter<int>("orb_uniform_cap", 1500);
  p.registration.orb_extract_count = node.declare_parameter<int>("orb_extract_count", 8000);
  p.registration.orb_grid_cell_px = node.declare_parameter<int>("orb_grid_cell_px", 16);
  p.registration.orb_max_per_cell = node.declare_parameter<int>("orb_max_per_cell", 8);
  p.registration.orb_scale_factor = node.declare_parameter<double>("orb_scale_factor", 1.2);
  p.registration.orb_nlevels = node.declare_parameter<int>("orb_nlevels", 4);
  p.registration.orb_fast_threshold = node.declare_parameter<int>("orb_fast_threshold", 20);
  p.registration.orb_match_method =
    parseOrbMatchMethod(node.declare_parameter<std::string>("orb_match_method", "radius"));
  p.registration.orb_xiang_gao_ratio = node.declare_parameter<double>("orb_xiang_gao_ratio", 2.0);
  p.registration.orb_lowe_ratio = node.declare_parameter<double>("orb_lowe_ratio", 0.75);
  p.registration.orb_radius_match_px = node.declare_parameter<int>("orb_radius_match_px", 100);
  p.registration.orb_match_ratio = node.declare_parameter<double>("feature_match_ratio", 0.75);
  p.registration.orb_min_inliers = node.declare_parameter<int>("feature_min_inliers", 10);
  p.registration.orb_ransac_threshold = node.declare_parameter<double>("orb_ransac_threshold", 3.0);
  p.registration.max_step_translation_px =
    node.declare_parameter<double>("max_step_translation_px", 200.0);
  p.registration.max_step_rotation_rad =
    node.declare_parameter<double>("max_step_rotation_rad", 0.15);
  p.registration.mask_bottom_exclude_ratio =
    node.declare_parameter<double>("mask_bottom_exclude_ratio", 0.12);
  p.registration.mask_gray_threshold =
    static_cast<uchar>(node.declare_parameter<int>("mask_gray_threshold", 25));
  p.registration.use_coarse_translation_init =
    node.declare_parameter<bool>("use_coarse_translation_init", true);
  p.registration.coarse_max_dy_px = node.declare_parameter<int>("coarse_max_dy_px", 120);
  p.registration.coarse_dy_step_px = node.declare_parameter<int>("coarse_dy_step_px", 2);
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
    const std::string blend_topic =
      declare_parameter<std::string>("blend_topic", "/ipm/mosaic/blend");
    const std::string diff_topic =
      declare_parameter<std::string>("diff_topic", "/ipm/mosaic/abs_diff");
    const std::string pose_topic =
      declare_parameter<std::string>("pose_topic", "/ipm/mosaic/global_pose");
    const std::string stats_topic =
      declare_parameter<std::string>("stats_topic", "/ipm/mosaic/stats");
    reset_on_first_frame_ = declare_parameter<bool>("reset_on_first_frame", false);

    canvas_pub_ = image_transport::create_publisher(this, canvas_topic);
    aligned_pub_ = image_transport::create_publisher(this, aligned_topic);
    blend_pub_ = image_transport::create_publisher(this, blend_topic);
    diff_pub_ = image_transport::create_publisher(this, diff_topic);
    pose_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(pose_topic, 10);
    stats_pub_ = create_publisher<std_msgs::msg::String>(stats_topic, 10);

    bev_sub_ = image_transport::create_subscription(
      this, bev_topic, std::bind(&BevMosaicNode::onBevImage, this, std::placeholders::_1), "raw");

    RCLCPP_INFO(get_logger(), "bev_mosaic_node on %s", bev_topic.c_str());
    RCLCPP_INFO(
      get_logger(), "  publish: %s, %s, %s, %s, %s", canvas_topic.c_str(), aligned_topic.c_str(),
      blend_topic.c_str(), diff_topic.c_str(), pose_topic.c_str());
  }

private:
  void publishCvImage(
    const image_transport::Publisher & pub, const cv::Mat & mat,
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

  static std::string formatStats(const BevMosaicFrameResult & frame, double ms)
  {
    const auto & reg = frame.motion;
    std::ostringstream oss;
    oss << "valid=" << (reg.valid ? 1 : 0) << " method=" << reg.method_used
        << " fallback=" << (frame.used_fallback ? 1 : 0) << " stored=" << frame.stored_frames
        << " corr=" << reg.correlation << " inliers=" << reg.inlier_count << " dx=" << reg.dx_px
        << " dy=" << reg.dy_px << " yaw_deg=" << (reg.yaw_rad * 180.0 / CV_PI) << " ms=" << ms;
    return oss.str();
  }

  void onBevImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    RCLCPP_INFO(
      get_logger(), "MOSAIC frame=%d | received %s", frame_index_, msg->header.frame_id.c_str());
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
    const BevMosaicFrameResult frame = accumulator_.accumulate(cv_ptr->image);
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std_msgs::msg::Header header = msg->header;
    header.frame_id = frame_id_;

    publishCvImage(canvas_pub_, accumulator_.canvas(), header);
    publishCvImage(
      aligned_pub_, frame.motion.valid ? accumulator_.lastAlignedFrame() : cv::Mat(), header);
    publishCvImage(blend_pub_, frame.blend_with_previous, header);
    publishCvImage(diff_pub_, frame.abs_diff_with_previous, header);
    publishPose(frame.global_affine_2x3);

    std_msgs::msg::String stats;
    stats.data = formatStats(frame, ms);
    stats_pub_->publish(stats);

    ++frame_index_;
    RCLCPP_INFO(get_logger(), "MOSAIC frame=%d | %s", frame_index_, stats.data.c_str());
  }

  BevMosaicParams params_;
  BevMosaicAccumulator accumulator_;
  std::string frame_id_;
  bool reset_on_first_frame_{false};
  int frame_index_{0};

  image_transport::Subscriber bev_sub_;
  image_transport::Publisher canvas_pub_;
  image_transport::Publisher aligned_pub_;
  image_transport::Publisher blend_pub_;
  image_transport::Publisher diff_pub_;
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
