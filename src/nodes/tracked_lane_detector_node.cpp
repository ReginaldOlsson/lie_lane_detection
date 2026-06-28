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

#include "lie_lane_detection/preprocessing/auto_frontal_ipm.hpp"
#include "lie_lane_detection/motion/ego_motion_estimator.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/tracking/lane_tracker.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"

namespace lie_lane_detection
{

PipelineParams loadTrackedParams(rclcpp::Node & node)
{
  PipelineParams p;
  p.use_steerable_filter = node.declare_parameter<bool>("use_steerable_filter", true);
  p.connect_dashed_edges = node.declare_parameter<bool>("connect_dashed_edges", true);
  p.edge_anisotropic_blur = node.declare_parameter<bool>("edge_anisotropic_blur", true);
  p.edge_thin = node.declare_parameter<bool>("edge_thin", true);
  p.edge_low_threshold = node.declare_parameter<double>("edge_low_threshold", 25.0);
  p.edge_high_threshold = node.declare_parameter<double>("edge_high_threshold", 70.0);
  p.top_k_peaks = node.declare_parameter<int>("top_k_peaks", 10);
  p.max_lane_hypotheses = node.declare_parameter<int>("max_lane_hypotheses", 10);
  p.use_iterative_peeling = node.declare_parameter<bool>("use_iterative_peeling", true);
  p.vote_threshold_px = node.declare_parameter<double>("vote_threshold_px", 8.0);
  p.inlier_threshold_px = node.declare_parameter<double>("inlier_threshold_px", 10.0);
  p.min_inlier_ratio = node.declare_parameter<double>("min_inlier_ratio", 0.30);
  p.min_inliers = node.declare_parameter<int>("min_inliers", 15);
  p.kappa_min = node.declare_parameter<double>("kappa_min", -0.28);
  p.kappa_max = node.declare_parameter<double>("kappa_max", 0.28);
  p.sigma_min = node.declare_parameter<double>("sigma_min", -0.55);
  p.sigma_max = node.declare_parameter<double>("sigma_max", 0.55);
  p.hough_hypothesis_merge_ratio =
    node.declare_parameter<double>("hough_hypothesis_merge_ratio", 0.85);
  p.min_inlier_y_coverage = node.declare_parameter<double>("min_inlier_y_coverage", 0.30);
  return p;
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
    const int track_main_interval = declare_parameter<int>("track_main_interval", 5);
    publish_raw_ = declare_parameter<bool>("publish_raw", true);
    publish_fps_ = declare_parameter<double>("expected_fps", 10.0);

    LaneTrackerParams tp;
    tp.main_detect_interval = track_main_interval;
    tp.max_tracks = declare_parameter<int>("track_max_tracks", 6);
    tp.min_spawn_inlier_ratio = declare_parameter<double>("track_min_spawn_inlier", 0.45);
    tp.max_stripe_lateral_delta_px =
      declare_parameter<double>("track_max_stripe_delta_px", 5.0);
    tp.fixed_camera = declare_parameter<bool>("track_fixed_camera", true);
    tp.snap_on_main = declare_parameter<bool>("track_snap_on_main", true);
    lane_tracker_.setParams(tp);

    track_marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/track/markers", 10);
    track_merge_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/track/merge_markers", 10);
    track_stats_pub_ = create_publisher<std_msgs::msg::String>("/lanes/track/stats", 10);
    track_overlay_pub_ = image_transport::create_publisher(this, "/lanes/track/overlay");
    track_frontal_pub_ = image_transport::create_publisher(this, "/lanes/track/frontal_overlay");
    track_edges_pub_ = image_transport::create_publisher(this, "/lanes/track/edges");

    raw_marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/raw/markers", 10);
    raw_merge_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/raw/merge_markers", 10);
    raw_stats_pub_ = create_publisher<std_msgs::msg::String>("/lanes/raw/stats", 10);
    raw_overlay_pub_ = image_transport::create_publisher(this, "/lanes/raw/overlay");
    raw_frontal_pub_ = image_transport::create_publisher(this, "/lanes/raw/frontal_overlay");
    raw_edges_pub_ = image_transport::create_publisher(this, "/lanes/raw/edges");

    bev_pub_ = image_transport::create_publisher(this, "/lanes/debug/bev");

    image_sub_ = image_transport::create_subscription(
      this, image_topic,
      std::bind(&TrackedLaneDetectorNode::onImage, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    RCLCPP_INFO(
      get_logger(),
      "Lane detector on %s | track interval=%d | publish_raw=%s",
      image_topic.c_str(), track_main_interval, publish_raw_ ? "true" : "false");
    RCLCPP_INFO(get_logger(), "  track: /lanes/track/{overlay,frontal_overlay,stats,markers}");
    RCLCPP_INFO(get_logger(), "  raw:   /lanes/raw/{overlay,frontal_overlay,stats,markers}");
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

    if (ego.valid) {
      ego.delta_bev_x = EgoMotionEstimator::imageDeltaToBevLateral(
        ego.delta_image_x, hg.params, hg.bev.cols);
      lane_tracker_.compensateEgoMotion(ego.delta_bev_x);
    }

    PipelineParams bev_params = hg.params;
    configureParamsForBev(bev_params, hg.bev.cols, hg.bev.rows);
    configureParamsForPerspectiveIpm(bev_params);

    BevDetectionResult raw_result;
    if (publish_raw_) {
      const auto t_raw0 = std::chrono::steady_clock::now();
      raw_result = detectLanesInBev(hg.bev, bev_params);
      const auto t_raw1 = std::chrono::steady_clock::now();
      raw_total_ms_ = std::chrono::duration<double, std::milli>(t_raw1 - t_raw0).count();
    }

    const TrackedFrameResult tracked =
      lane_tracker_.processFrame(hg.bev, bev_params, video_frame_index_, publish_fps_);
    ++video_frame_index_;

    const auto stamp = msg->header.stamp;
    publishCvImage(bev_pub_, hg.bev, msg->header);

    if (publish_raw_) {
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
  LaneTracker lane_tracker_;
  VanishingPointTracker vp_tracker_;
  EgoMotionEstimator ego_estimator_;
  std::string frame_id_;
  bool publish_raw_{true};
  int video_frame_index_{0};
  double publish_fps_{10.0};
  double raw_total_ms_{0.0};

  image_transport::Subscriber image_sub_;
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
