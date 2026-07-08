#include "lie_lane_detection/nodes/node_params.hpp"
#include "lie_lane_detection/pipeline/intensity_strip_lane_detector.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <chrono>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>

namespace lie_lane_detection
{
namespace
{

bool homographyFromMsg(
  const std_msgs::msg::Float64MultiArray & msg,
  cv::Mat & H_out,
  builtin_interfaces::msg::Time * stamp_out = nullptr)
{
  if (msg.data.size() != 9 && msg.data.size() != 11) {
    return false;
  }
  H_out = cv::Mat(3, 3, CV_64F);
  for (int r = 0; r < 3; ++r) {
    for (int c = 0; c < 3; ++c) {
      H_out.at<double>(r, c) = msg.data[static_cast<size_t>(r * 3 + c)];
    }
  }
  if (stamp_out != nullptr && msg.data.size() >= 11) {
    stamp_out->sec = static_cast<int32_t>(msg.data[9]);
    stamp_out->nanosec = static_cast<uint32_t>(msg.data[10]);
  }
  return true;
}

}  // namespace

class IntensityStripLaneDetectorNode : public rclcpp::Node
{
public:
  IntensityStripLaneDetectorNode()
  : Node("intensity_strip_lane_detector_node"),
    detect_params_(loadDetectionParams(*this)),
    strip_params_(loadIntensityStripParams(*this)),
    frame_id_(declare_parameter<std::string>("frame_id", "camera_front"))
  {
    const std::string bev_topic = declare_parameter<std::string>("bev_topic", "/ipm/bev");
    const std::string image_topic = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    const std::string homography_topic =
      declare_parameter<std::string>("homography_topic", "/ipm/homography");
    const std::string frontal_overlay_topic =
      declare_parameter<std::string>("frontal_overlay_topic", "/camera/image_raw/overlay");

    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/intensity/markers", 10);
    stats_pub_ = create_publisher<std_msgs::msg::String>("/lanes/intensity/stats", 10);
    overlay_pub_ = image_transport::create_publisher(this, "/lanes/intensity/overlay");
    projection_pub_ = image_transport::create_publisher(this, "/lanes/intensity/projection");
    frontal_overlay_pub_ = image_transport::create_publisher(this, frontal_overlay_topic);

    image_sub_ = image_transport::create_subscription(
      this, image_topic,
      std::bind(&IntensityStripLaneDetectorNode::onImage, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    homography_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      homography_topic, rclcpp::QoS(10),
      std::bind(&IntensityStripLaneDetectorNode::onHomography, this, std::placeholders::_1));

    bev_sub_ = image_transport::create_subscription(
      this, bev_topic,
      std::bind(&IntensityStripLaneDetectorNode::onBevImage, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    RCLCPP_INFO(get_logger(), "intensity_strip_lane_detector_node on %s", bev_topic.c_str());
    RCLCPP_INFO(
      get_logger(),
      "  publish: /lanes/intensity/{overlay,projection,markers,stats}");
    RCLCPP_INFO(get_logger(), "  frontal overlay: %s", frontal_overlay_topic.c_str());
    RCLCPP_INFO(
      get_logger(),
      "  strips: H=%d max_peaks=%d v_half_w=%d",
      strip_params_.num_horizontal_strips,
      strip_params_.max_peaks_per_strip,
      strip_params_.vertical_strip_half_width_px);
  }

private:
  using StampKey = std::pair<int32_t, uint32_t>;

  struct StampKeyHash
  {
    size_t operator()(const StampKey & key) const noexcept
    {
      const size_t h1 = std::hash<int32_t>{}(key.first);
      const size_t h2 = std::hash<uint32_t>{}(key.second);
      return h1 ^ (h2 + 0x9e3779b9 + (h1 << 6) + (h1 >> 2));
    }
  };

  struct PendingFrontalOverlay
  {
    std_msgs::msg::Header header;
    IntensityStripDetectionResult result;
  };

  static StampKey stampKey(const builtin_interfaces::msg::Time & stamp)
  {
    return {stamp.sec, stamp.nanosec};
  }

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

  void pruneCaches()
  {
    while (stamp_order_.size() > 8) {
      const StampKey old = stamp_order_.front();
      stamp_order_.pop_front();
      frontal_cache_.erase(old);
      homography_cache_.erase(old);
    }
  }

  void touchStamp(const StampKey & key)
  {
    for (auto it = stamp_order_.begin(); it != stamp_order_.end(); ++it) {
      if (*it == key) {
        stamp_order_.erase(it);
        break;
      }
    }
    stamp_order_.push_back(key);
    pruneCaches();
  }

  void tryPublishFrontalOverlay()
  {
    if (!pending_frontal_.has_value()) {
      return;
    }

    const StampKey key = stampKey(pending_frontal_->header.stamp);
    const auto img_it = frontal_cache_.find(key);
    const auto H_it = homography_cache_.find(key);
    if (img_it == frontal_cache_.end() || H_it == homography_cache_.end()) {
      return;
    }

    const cv::Mat frontal_overlay = drawFrontalIntensityStripOverlay(
      img_it->second, pending_frontal_->result, strip_params_, H_it->second);
    publishCvImage(frontal_overlay_pub_, frontal_overlay, pending_frontal_->header);
    pending_frontal_.reset();
  }

  void onImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge error (image): %s", e.what());
      return;
    }

    std::lock_guard<std::mutex> lock(sync_mutex_);
    const StampKey key = stampKey(msg->header.stamp);
    frontal_cache_[key] = cv_ptr->image.clone();
    touchStamp(key);
    tryPublishFrontalOverlay();
  }

  void onHomography(const std_msgs::msg::Float64MultiArray::ConstSharedPtr & msg)
  {
    cv::Mat H;
    builtin_interfaces::msg::Time stamp;
    if (!homographyFromMsg(*msg, H, &stamp)) {
      return;
    }

    std::lock_guard<std::mutex> lock(sync_mutex_);
    StampKey key;
    if (msg->data.size() >= 11) {
      key = stampKey(stamp);
    } else if (pending_frontal_.has_value()) {
      key = stampKey(pending_frontal_->header.stamp);
    } else {
      return;
    }

    homography_cache_[key] = H;
    touchStamp(key);
    tryPublishFrontalOverlay();
  }

  void onBevImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge error (bev): %s", e.what());
      return;
    }

    const auto t0 = std::chrono::steady_clock::now();
    const IntensityStripDetectionResult result =
      detectLanesIntensityStrips(cv_ptr->image, detect_params_, strip_params_);
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    publishCvImage(overlay_pub_, result.overlay, msg->header);
    publishCvImage(projection_pub_, result.projection_debug, msg->header);
    marker_pub_->publish(lanesToMarkers(result.lanes, frame_id_, msg->header.stamp));

    {
      std::lock_guard<std::mutex> lock(sync_mutex_);
      pending_frontal_ = PendingFrontalOverlay{msg->header, result};
      tryPublishFrontalOverlay();
      if (pending_frontal_.has_value()) {
        RCLCPP_DEBUG(
          get_logger(),
          "Waiting for image/homography stamp %d.%u",
          msg->header.stamp.sec, msg->header.stamp.nanosec);
      }
    }

    std::ostringstream stats;
    stats << "mode=intensity_strip"
          << " H=" << result.horizontal_strip_count
          << " V=" << result.vertical_strips.size()
          << " lanes=" << result.lanes.size()
          << " total_ms=" << total_ms
          << " detect_ms=" << result.elapsed_ms;

    std_msgs::msg::String stats_msg;
    stats_msg.data = stats.str();
    stats_pub_->publish(stats_msg);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "INTENSITY | %s", stats_msg.data.c_str());
  }

  PipelineParams detect_params_;
  IntensityStripParams strip_params_;
  std::string frame_id_;

  std::mutex sync_mutex_;
  std::deque<StampKey> stamp_order_;
  std::unordered_map<StampKey, cv::Mat, StampKeyHash> frontal_cache_;
  std::unordered_map<StampKey, cv::Mat, StampKeyHash> homography_cache_;
  std::optional<PendingFrontalOverlay> pending_frontal_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr homography_sub_;
  image_transport::Subscriber bev_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr stats_pub_;
  image_transport::Publisher overlay_pub_;
  image_transport::Publisher projection_pub_;
  image_transport::Publisher frontal_overlay_pub_;
};

}  // namespace lie_lane_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lie_lane_detection::IntensityStripLaneDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
