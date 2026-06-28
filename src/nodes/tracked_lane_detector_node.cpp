#include <chrono>
#include <memory>
#include <sstream>
#include <string>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "lie_lane_detection/nodes/node_params.hpp"
#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"
#include "lie_lane_detection/motion/ego_motion_estimator.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/tracking/lane_tracker.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"

namespace lie_lane_detection
{

PipelineParams loadTrackedParams(rclcpp::Node & node)
{
  return loadDetectionParams(node);
}

class TrackedLaneDetectorNode : public rclcpp::Node
{
public:
  TrackedLaneDetectorNode()
  : Node("tracked_lane_detector_node"),
    params_(loadTrackedParams(*this)),
    frame_id_(declare_parameter<std::string>("frame_id", "camera_front"))
  {
    const std::string image_topic = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    use_lane_tracking_ = declare_parameter<bool>("use_lane_tracking", false);
    const int track_main_interval = declare_parameter<int>("track_main_interval", 5);
    publish_raw_ = declare_parameter<bool>("publish_raw", true);
    publish_fps_ = declare_parameter<double>("expected_fps", 10.0);

    if (use_lane_tracking_) {
      LaneTrackerParams tp;
      tp.main_detect_interval = track_main_interval;
      tp.max_tracks = declare_parameter<int>("track_max_tracks", 6);
      tp.min_spawn_inlier_ratio = declare_parameter<double>("track_min_spawn_inlier", 0.45);
      tp.max_stripe_lateral_delta_px =
        declare_parameter<double>("track_max_stripe_delta_px", 5.0);
      tp.fixed_camera = declare_parameter<bool>("track_fixed_camera", true);
      tp.snap_on_main = declare_parameter<bool>("track_snap_on_main", true);
      lane_tracker_.setParams(tp);
    }

    detect_marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/detect/markers", 10);
    detect_merge_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/detect/merge_markers", 10);
    detect_stats_pub_ = create_publisher<std_msgs::msg::String>("/lanes/detect/stats", 10);
    detect_overlay_pub_ = image_transport::create_publisher(this, "/lanes/detect/overlay");
    detect_frontal_pub_ = image_transport::create_publisher(this, "/lanes/detect/frontal_overlay");
    detect_edges_pub_ = image_transport::create_publisher(this, "/lanes/detect/edges");

    if (use_lane_tracking_) {
      track_marker_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/track/markers", 10);
      track_merge_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/track/merge_markers", 10);
      track_stats_pub_ = create_publisher<std_msgs::msg::String>("/lanes/track/stats", 10);
      track_overlay_pub_ = image_transport::create_publisher(this, "/lanes/track/overlay");
      track_frontal_pub_ = image_transport::create_publisher(this, "/lanes/track/frontal_overlay");
      track_edges_pub_ = image_transport::create_publisher(this, "/lanes/track/edges");
    }

    if (use_lane_tracking_ && publish_raw_) {
      raw_marker_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/raw/markers", 10);
      raw_merge_pub_ =
        create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/raw/merge_markers", 10);
      raw_stats_pub_ = create_publisher<std_msgs::msg::String>("/lanes/raw/stats", 10);
      raw_overlay_pub_ = image_transport::create_publisher(this, "/lanes/raw/overlay");
      raw_frontal_pub_ = image_transport::create_publisher(this, "/lanes/raw/frontal_overlay");
      raw_edges_pub_ = image_transport::create_publisher(this, "/lanes/raw/edges");
    }

    bev_pub_ = image_transport::create_publisher(this, "/lanes/debug/bev");

    image_sub_ = image_transport::create_subscription(
      this, image_topic,
      std::bind(&TrackedLaneDetectorNode::onImage, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    RCLCPP_WARN(
      get_logger(),
      "tracked_lane_detector_node is deprecated; use frontal_ipm_node + bev_lane_detector_node");
    RCLCPP_INFO(
      get_logger(),
      "Lane detector on %s | tracking=%s | publish_raw=%s",
      image_topic.c_str(),
      use_lane_tracking_ ? "on" : "off",
      publish_raw_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  detect: /lanes/detect/{overlay,frontal_overlay,stats,markers}");
    if (use_lane_tracking_) {
      RCLCPP_INFO(get_logger(), "  track:  /lanes/track/{overlay,frontal_overlay,stats,markers}");
      if (publish_raw_) {
        RCLCPP_INFO(get_logger(), "  raw:    /lanes/raw/{overlay,frontal_overlay,stats,markers}");
      }
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

  static std::string formatRawStats(const BevDetectionResult & raw, double total_ms)
  {
    std::ostringstream oss;
    oss << "mode=raw"
        << " lanes=" << raw.lanes.size()
        << " total_ms=" << total_ms
        << " detect_ms=" << raw.elapsed_ms;
    return oss.str();
  }

  static std::string formatTrackStats(
    const TrackedFrameResult & tracked,
    double total_ms,
    const VanishingPointEstimate & vp,
    const EgoMotionEstimate & ego)
  {
    std::ostringstream oss;
    oss << "mode=" << (tracked.ran_main_detector ? "main" : "stripe")
        << " lanes=" << tracked.lanes.size()
        << " tracks=" << tracked.track_count
        << " total_ms=" << total_ms
        << " track_ms=" << tracked.total_ms
        << " predict_ms=" << tracked.predict_ms
        << " stripe_ms=" << tracked.stripe_ms
        << " main_ms=" << tracked.main_ms
        << " vp=(" << vp.x << "," << vp.y << ")"
        << " vp_filt=" << (vp.used_temporal_prior ? 1 : 0);
    if (ego.valid) {
      oss << " ego_dx=" << ego.delta_image_x
          << " ego_bev=" << ego.delta_bev_x
          << " ego_yaw=" << ego.delta_yaw_rad;
    }
    return oss.str();
  }

  void publishDetection(
    const BevDetectionResult & result,
    double detect_ms,
    const std_msgs::msg::Header & header,
    const cv::Mat & frontal_bgr,
    const cv::Mat & H_img2bev)
  {
    detect_marker_pub_->publish(lanesToMarkers(result.lanes, frame_id_, header.stamp));
    detect_merge_pub_->publish(mergesToMarkers(result.merges, frame_id_, header.stamp));
    publishCvImage(detect_overlay_pub_, result.overlay, header);
    publishCvImage(detect_edges_pub_, result.edges, header);
    const cv::Mat frontal = drawFrontalOverlay(frontal_bgr, result.lanes, result.merges, H_img2bev);
    publishCvImage(detect_frontal_pub_, frontal, header);

    std_msgs::msg::String stats;
    stats.data = formatRawStats(result, detect_ms);
    detect_stats_pub_->publish(stats);
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

    const auto t0 = std::chrono::steady_clock::now();
    EgoMotionEstimate ego = ego_estimator_.update(cv_ptr->image);
    const FrontalHomographyResult hg =
      estimateFrontalHomography(cv_ptr->image, params_, &vp_tracker_, &ego_estimator_);
    if (!hg.valid || hg.bev.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Auto-IPM failed");
      return;
    }

    if (use_lane_tracking_ && ego.valid) {
      ego.delta_bev_x = EgoMotionEstimator::imageDeltaToBevLateral(
        ego.delta_image_x, hg.params, hg.bev.cols);
      lane_tracker_.compensateEgoMotion(ego.delta_bev_x);
    }

    PipelineParams bev_params = hg.params;
    configureParamsForBev(bev_params, hg.bev.cols, hg.bev.rows);
    configureParamsForPerspectiveIpm(bev_params);
    const cv::Mat bev_display = prepareBevForDetection(hg.bev, bev_params);
    const auto stamp = msg->header.stamp;
    publishCvImage(bev_pub_, bev_display, msg->header);

    if (!use_lane_tracking_) {
      const auto t_det0 = std::chrono::steady_clock::now();
      const BevDetectionResult det = detectLanesInBev(bev_display, bev_params);
      const auto t_det1 = std::chrono::steady_clock::now();
      const double detect_ms =
        std::chrono::duration<double, std::milli>(t_det1 - t_det0).count();
      publishDetection(det, detect_ms, msg->header, cv_ptr->image, hg.H_img2bev);
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 1000,
        "DETECT | %zu lanes | %.1f ms",
        det.lanes.size(), detect_ms);
      return;
    }

    BevDetectionResult raw_result;
    if (publish_raw_) {
      const auto t_raw0 = std::chrono::steady_clock::now();
      raw_result = detectLanesInBev(bev_display, bev_params);
      const auto t_raw1 = std::chrono::steady_clock::now();
      raw_total_ms_ = std::chrono::duration<double, std::milli>(t_raw1 - t_raw0).count();
    }

    const TrackedFrameResult tracked =
      lane_tracker_.processFrame(bev_display, bev_params, video_frame_index_, publish_fps_);
    ++video_frame_index_;

    if (publish_raw_) {
      publishDetection(raw_result, raw_total_ms_, msg->header, cv_ptr->image, hg.H_img2bev);
      raw_marker_pub_->publish(lanesToMarkers(raw_result.lanes, frame_id_, stamp));
      raw_merge_pub_->publish(mergesToMarkers(raw_result.merges, frame_id_, stamp));
      publishCvImage(raw_overlay_pub_, raw_result.overlay, msg->header);
      publishCvImage(raw_edges_pub_, raw_result.edges, msg->header);
      const cv::Mat raw_frontal = drawFrontalOverlay(
        cv_ptr->image, raw_result.lanes, raw_result.merges, hg.H_img2bev);
      publishCvImage(raw_frontal_pub_, raw_frontal, msg->header);
      std_msgs::msg::String raw_stats;
      raw_stats.data = formatRawStats(raw_result, raw_total_ms_);
      raw_stats_pub_->publish(raw_stats);
    }

    track_marker_pub_->publish(lanesToMarkers(tracked.lanes, frame_id_, stamp));
    track_merge_pub_->publish(mergesToMarkers(tracked.merges, frame_id_, stamp));
    publishCvImage(track_overlay_pub_, tracked.overlay, msg->header);
    publishCvImage(track_edges_pub_, tracked.edges, msg->header);
    const cv::Mat track_frontal = drawFrontalOverlay(
      cv_ptr->image, tracked.lanes, tracked.merges, hg.H_img2bev);
    publishCvImage(track_frontal_pub_, track_frontal, msg->header);

    const auto t1 = std::chrono::steady_clock::now();
    const double track_total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std_msgs::msg::String track_stats;
    track_stats.data = formatTrackStats(tracked, track_total_ms, hg.vanishing_point, ego);
    track_stats_pub_->publish(track_stats);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "TRACK %s | %zu lanes | %.1f ms | RAW %zu lanes | %.1f ms",
      tracked.ran_main_detector ? "MAIN" : "STRIPE",
      tracked.lanes.size(), tracked.total_ms,
      publish_raw_ ? raw_result.lanes.size() : 0,
      publish_raw_ ? raw_total_ms_ : 0.0);
  }

  PipelineParams params_;
  bool use_lane_tracking_{false};
  LaneTracker lane_tracker_;
  VanishingPointTracker vp_tracker_;
  EgoMotionEstimator ego_estimator_;
  std::string frame_id_;
  bool publish_raw_{true};
  int video_frame_index_{0};
  double publish_fps_{10.0};
  double raw_total_ms_{0.0};

  image_transport::Subscriber image_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr detect_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr detect_merge_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr detect_stats_pub_;
  image_transport::Publisher detect_overlay_pub_;
  image_transport::Publisher detect_frontal_pub_;
  image_transport::Publisher detect_edges_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr track_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr track_merge_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr track_stats_pub_;
  image_transport::Publisher track_overlay_pub_;
  image_transport::Publisher track_frontal_pub_;
  image_transport::Publisher track_edges_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr raw_marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr raw_merge_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr raw_stats_pub_;
  image_transport::Publisher raw_overlay_pub_;
  image_transport::Publisher raw_frontal_pub_;
  image_transport::Publisher raw_edges_pub_;
  image_transport::Publisher bev_pub_;
};

}  // namespace lie_lane_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lie_lane_detection::TrackedLaneDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
