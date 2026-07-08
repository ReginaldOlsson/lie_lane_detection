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
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"

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

    // Fixed-mount camera: the IPM homography is essentially constant, so we
    // estimate the vanishing point only during a short warmup and then freeze
    // the homography. Steady-state frames skip VP detection entirely and only
    // run cv::warpPerspective.
    freeze_ipm_ = declare_parameter<bool>("freeze_ipm", true);
    calibration_frames_ =
      static_cast<int>(declare_parameter<int>("ipm_calibration_frames", 20));
    max_calibration_frames_ =
      static_cast<int>(declare_parameter<int>("ipm_max_calibration_frames", 120));

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

    // Manual/static IPM: freeze immediately from configured src points, so VP
    // detection never runs.
    if (freeze_ipm_ && params_.use_manual_ipm && params_.ipm_src_points.size() >= 8) {
      if (tryFreeze(params_)) {
        RCLCPP_INFO(get_logger(), "  IPM frozen from manual ipm_src_points (no VP warmup).");
      }
    } else if (freeze_ipm_) {
      RCLCPP_INFO(
        get_logger(),
        "  IPM auto-calibration: freeze after %d valid VP frames (hard cap %d).",
        calibration_frames_, max_calibration_frames_);
    }
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
    msg.data.resize(11);
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c) {
        msg.data[static_cast<size_t>(r * 3 + c)] = H.at<double>(r, c);
      }
    }
    msg.data[9] = static_cast<double>(header.stamp.sec);
    msg.data[10] = static_cast<double>(header.stamp.nanosec);
    msg.layout.data_offset = 0;
    homography_pub_->publish(msg);
  }

  // Build a persistent IPMTransformer from converged parameters and cache the
  // homography. Returns true once the fast warp-only path is armed.
  bool tryFreeze(PipelineParams frozen)
  {
    if (frozen.ipm_src_points.size() < 8) {
      return false;
    }
    if (frozen.ipm_dst_points.size() < 8) {
      updateIpmDstFromBevExtent(frozen);
    }
    auto ipm = std::make_unique<IPMTransformer>(frozen);
    if (!ipm->computeHomography(nullptr)) {
      return false;
    }
    frozen_ipm_ = std::move(ipm);
    frozen_params_ = frozen;
    frozen_H_ = frozen_ipm_->homography().clone();
    ipm_frozen_ = true;
    return true;
  }

  // Steady-state path: only warpPerspective + road mask, no VP detection.
  void processFrozen(const cv::Mat & image, const std_msgs::msg::Header & header)
  {
    cv::Mat bev = frozen_ipm_->warpToBev(image);
    if (bev.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Frozen IPM warp failed");
      return;
    }
    bev = prepareBevImage(bev);
    maskBevBottomExclude(bev, params_);
    publishHomography(frozen_H_, header);
    publishCvImage(bev_pub_, bev, header);
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 5000, "IPM(frozen) %dx%d", bev.cols, bev.rows);
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

    if (freeze_ipm_ && ipm_frozen_ && frozen_ipm_) {
      processFrozen(cv_ptr->image, msg->header);
      return;
    }

    // Calibration path: run full VP-based auto-IPM.
    ego_estimator_.update(cv_ptr->image);
    const FrontalHomographyResult hg =
      estimateFrontalHomography(cv_ptr->image, params_, &vp_tracker_, &ego_estimator_);
    if (!hg.valid || hg.bev.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Auto-IPM failed");
      return;
    }

    cv::Mat bev_out = hg.bev.clone();
    maskBevBottomExclude(bev_out, params_);

    publishHomography(hg.H_img2bev, msg->header);
    publishCvImage(bev_pub_, bev_out, msg->header);
    publishCvImage(roi_pub_, hg.debug_roi, msg->header);

    if (freeze_ipm_) {
      ++total_calib_frames_;
      if (hg.vanishing_point.valid && !hg.used_fallback_roi) {
        ++vp_valid_count_;
      }
      const bool converged = vp_valid_count_ >= calibration_frames_;
      const bool timed_out = total_calib_frames_ >= max_calibration_frames_;
      if ((converged || timed_out) && tryFreeze(hg.params)) {
        RCLCPP_INFO(
          get_logger(),
          "IPM frozen after %d frames (%d valid VP, %s). VP detection now skipped.",
          total_calib_frames_, vp_valid_count_, converged ? "converged" : "timeout-fallback");
        return;
      }
    }

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 2000,
      "IPM(calibrating %d/%d) %dx%d | vp_valid=%d fallback=%d",
      vp_valid_count_, calibration_frames_,
      hg.bev.cols, hg.bev.rows,
      hg.vanishing_point.valid ? 1 : 0,
      hg.used_fallback_roi ? 1 : 0);
  }

  PipelineParams params_;
  std::string frame_id_;
  VanishingPointTracker vp_tracker_;
  EgoMotionEstimator ego_estimator_;

  // IPM freeze / calibration state.
  bool freeze_ipm_{true};
  int calibration_frames_{20};
  int max_calibration_frames_{120};
  bool ipm_frozen_{false};
  int vp_valid_count_{0};
  int total_calib_frames_{0};
  PipelineParams frozen_params_;
  std::unique_ptr<IPMTransformer> frozen_ipm_;
  cv::Mat frozen_H_;

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
