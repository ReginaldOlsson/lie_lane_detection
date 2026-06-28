#include <memory>
#include <string>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

#include "lie_lane_detection/motion/ego_motion_estimator.hpp"
#include "lie_lane_detection/nodes/node_params.hpp"
#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"

namespace lie_lane_detection
{

class FrontalIpmNode : public rclcpp::Node
{
public:
  FrontalIpmNode()
  : Node("frontal_ipm_node"),
    params_(loadIpmParams(*this)),
    frame_id_(declare_parameter<std::string>("frame_id", "camera_front"))
  {
    const std::string image_topic =
      declare_parameter<std::string>("image_topic", "/camera/image_raw");
    const std::string bev_topic = declare_parameter<std::string>("bev_topic", "/ipm/bev");
    const std::string roi_topic =
      declare_parameter<std::string>("roi_debug_topic", "/ipm/debug/roi");
    const std::string homography_topic =
      declare_parameter<std::string>("homography_topic", "/ipm/homography");

    bev_pub_ = image_transport::create_publisher(this, bev_topic);
    roi_pub_ = image_transport::create_publisher(this, roi_topic);
    homography_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(homography_topic, 10);

    image_sub_ = image_transport::create_subscription(
      this, image_topic,
      std::bind(&FrontalIpmNode::onImage, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    RCLCPP_INFO(
      get_logger(),
      "frontal_ipm_node: %s -> %s (%.0fx%.0f m @ %.3f m/px, bottom_exclude=%.0f px)",
      image_topic.c_str(), bev_topic.c_str(),
      params_.bev_width_m, params_.bev_length_m, params_.bev_resolution_m_per_px,
      params_.bev_bottom_exclude_px);
    RCLCPP_INFO(get_logger(), "  debug: %s, homography: %s", roi_topic.c_str(), homography_topic.c_str());
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

  void publishHomography(const cv::Mat & H, const std_msgs::msg::Header & header)
  {
    if (H.empty() || H.rows != 3 || H.cols != 3) {
      return;
    }
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
        msg.data[static_cast<size_t>(r * 3 + c)] = H.at<double>(r, c);
      }
    }
    msg.layout.data_offset = 0;
    (void)header;
    homography_pub_->publish(msg);
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

    ego_estimator_.update(cv_ptr->image);
    const FrontalHomographyResult hg =
      estimateFrontalHomography(cv_ptr->image, params_, &vp_tracker_, &ego_estimator_);
    if (!hg.valid || hg.bev.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Auto-IPM failed");
      return;
    }

    cv::Mat bev_out = hg.bev.clone();
    maskBevBottomExclude(bev_out, params_);

    publishCvImage(bev_pub_, bev_out, msg->header);
    publishCvImage(roi_pub_, hg.debug_roi, msg->header);
    publishHomography(hg.H_img2bev, msg->header);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "IPM %dx%d | vp_valid=%d fallback=%d",
      hg.bev.cols, hg.bev.rows,
      hg.vanishing_point.valid ? 1 : 0,
      hg.used_fallback_roi ? 1 : 0);
  }

  PipelineParams params_;
  std::string frame_id_;
  VanishingPointTracker vp_tracker_;
  EgoMotionEstimator ego_estimator_;

  image_transport::Subscriber image_sub_;
  image_transport::Publisher bev_pub_;
  image_transport::Publisher roi_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr homography_pub_;
};

}  // namespace lie_lane_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lie_lane_detection::FrontalIpmNode>());
  rclcpp::shutdown();
  return 0;
}
