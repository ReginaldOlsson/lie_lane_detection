#include "lie_lane_detection/geometry/template_curve.hpp"
#include "lie_lane_detection/nodes/node_params.hpp"
#include "lie_lane_detection/pipeline/detection_common.hpp"
#include "lie_lane_detection/pipeline/lane_detection_runner.hpp"
#include "lie_lane_detection/visualization/visualization.hpp"

#include <algorithm>
#include <cmath>

#include <cv_bridge/cv_bridge.hpp>
#include <image_transport/image_transport.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
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

class BevLaneDetectorNode : public rclcpp::Node
{
public:
  BevLaneDetectorNode()
  : Node("bev_lane_detector_node"),
    params_(loadDetectionParams(*this)),
    frame_id_(declare_parameter<std::string>("frame_id", "camera_front"))
  {
    const std::string bev_topic = declare_parameter<std::string>("bev_topic", "/ipm/bev");
    const std::string image_topic = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    const std::string image_transport =
      declare_parameter<std::string>("image_transport", "raw");
    const std::string homography_topic =
      declare_parameter<std::string>("homography_topic", "/ipm/homography");
    const std::string frontal_overlay_topic =
      declare_parameter<std::string>("frontal_overlay_topic", "/lanes/detect/frontal_overlay");

    // Optional cap on detection rate (0 = process as fast as the worker can).
    max_detect_rate_hz_ = declare_parameter<double>("detect_max_rate_hz", 0.0);

    marker_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/detect/markers", 10);
    merge_pub_ =
      create_publisher<visualization_msgs::msg::MarkerArray>("/lanes/detect/merge_markers", 10);
    stats_pub_ = create_publisher<std_msgs::msg::String>("/lanes/detect/stats", 10);
    overlay_pub_ = image_transport::create_publisher(this, "/lanes/detect/overlay");
    edges_pub_ = image_transport::create_publisher(this, "/lanes/detect/edges");
    filtered_pub_ = image_transport::create_publisher(this, "/lanes/detect/filtered");
    frontal_overlay_pub_ = image_transport::create_publisher(this, frontal_overlay_topic);

    image_sub_ = image_transport::create_subscription(
      this, image_topic,
      std::bind(&BevLaneDetectorNode::onImage, this, std::placeholders::_1),
      image_transport, rmw_qos_profile_sensor_data);

    homography_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
      homography_topic, rclcpp::QoS(10),
      std::bind(&BevLaneDetectorNode::onHomography, this, std::placeholders::_1));

    bev_sub_ = image_transport::create_subscription(
      this, bev_topic, std::bind(&BevLaneDetectorNode::onBevImage, this, std::placeholders::_1),
      "raw", rmw_qos_profile_sensor_data);

    RCLCPP_INFO(get_logger(), "bev_lane_detector_node on %s", bev_topic.c_str());
    RCLCPP_INFO(
      get_logger(),
      "  publish: /lanes/detect/{overlay,frontal_overlay,edges,filtered,markers,stats}");
    RCLCPP_INFO(get_logger(), "  frontal overlay: %s", frontal_overlay_topic.c_str());
    RCLCPP_INFO(
      get_logger(),
      "  detection input: IPM BGR (bottom mask only; no grayscale/Otsu preprocess)");

    // Detection runs on a dedicated worker so the ROS executor never blocks or
    // backlogs; the mailbox always keeps only the newest BEV frame.
    worker_ = std::thread(&BevLaneDetectorNode::detectionWorker, this);
  }

  ~BevLaneDetectorNode() override
  {
    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      running_ = false;
    }
    job_cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
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
    std::vector<LaneHypothesis> lanes;
    std::vector<MergeEvent> merges;
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

    const cv::Mat frontal_overlay = drawFrontalOverlay(
      img_it->second, pending_frontal_->lanes, pending_frontal_->merges, H_it->second);
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

  static std::string formatStats(
    const BevDetectionResult & result,
    double total_ms,
    double otsu_threshold)
  {
    std::ostringstream oss;
    oss << "mode=bev"
        << " lanes=" << result.lanes.size()
        << " total_ms=" << total_ms
        << " detect_ms=" << result.elapsed_ms
        << " edge_ms=" << result.edge_ms
        << " vote_ms=" << result.vote_ms
        << " fit_ms=" << result.fit_ms
        << " post_ms=" << result.post_ms
        << " edges=" << result.edge_point_count;
    if (otsu_threshold >= 0.0) {
      oss << " otsu_t=" << otsu_threshold;
    }
    return oss.str();
  }

  // Executor-thread callback: only decode + drop into the single-slot mailbox.
  // Any pending (unprocessed) frame is overwritten so the worker always runs on
  // the freshest BEV image and the executor never backlogs.
  void onBevImage(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge error (bev): %s", e.what());
      return;
    }

    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      // Clone: toCvShare aliases msg memory that is freed after this callback.
      pending_job_ = BevJob{cv_ptr->image.clone(), msg->header};
    }
    job_cv_.notify_one();
  }

  void detectionWorker()
  {
    using clock = std::chrono::steady_clock;
    auto last_start = clock::now() - std::chrono::hours(1);
    while (true) {
      BevJob job;
      {
        std::unique_lock<std::mutex> lock(job_mutex_);
        job_cv_.wait(lock, [this] {return !running_ || pending_job_.has_value();});
        if (!running_) {
          return;
        }
        job = std::move(*pending_job_);
        pending_job_.reset();
      }

      // Optional max-rate throttle (latest-frame-wins already caps to worker
      // throughput; this bounds CPU/GPU load when the source is faster).
      if (max_detect_rate_hz_ > 0.0) {
        const double min_interval_ms = 1000.0 / max_detect_rate_hz_;
        const double since_ms =
          std::chrono::duration<double, std::milli>(clock::now() - last_start).count();
        if (since_ms < min_interval_ms) {
          std::this_thread::sleep_for(
            std::chrono::duration<double, std::milli>(min_interval_ms - since_ms));
        }
      }
      last_start = clock::now();

      processBev(job.image, job.header);
    }
  }

  // Low-pass matched lanes across frames to reduce output jitter. Lanes are
  // matched to the previous frame by lateral offset (xi[0]); matched lanes are
  // blended (alpha*measurement + (1-alpha)*previous) and their polylines
  // regenerated. Worker-thread only state, so no locking required.
  void applyTemporalSmoothing(
    std::vector<LaneHypothesis> & lanes, const PipelineParams & p, int rows, int cols)
  {
    if (!p.use_output_temporal_smoothing) {
      prev_lanes_ = lanes;
      return;
    }
    const double alpha = std::clamp(p.temporal_smoothing_alpha, 0.0, 1.0);
    const double y_max = bevEffectiveYMax(rows, p);
    TemplateCurve curve(p);
    curve.setBevExtents(0.0, y_max, 0.0, static_cast<double>(cols));

    std::vector<bool> used(prev_lanes_.size(), false);
    for (auto & lane : lanes) {
      int best = -1;
      double best_d = p.temporal_match_max_vx_px;
      for (size_t j = 0; j < prev_lanes_.size(); ++j) {
        if (used[j]) {
          continue;
        }
        const double d = std::abs(lane.xi[0] - prev_lanes_[j].xi[0]);
        if (d < best_d) {
          best_d = d;
          best = static_cast<int>(j);
        }
      }
      if (best >= 0) {
        used[static_cast<size_t>(best)] = true;
        lane.xi = alpha * lane.xi + (1.0 - alpha) * prev_lanes_[static_cast<size_t>(best)].xi;
        lane.polyline = curve.samplePolyline(lane.xi);
      }
    }
    prev_lanes_ = lanes;
  }

  void processBev(const cv::Mat & bev_image, const std_msgs::msg::Header & header)
  {
    PipelineParams bev_params = params_;
    configureParamsForBev(bev_params, bev_image.cols, bev_image.rows);
    configureParamsForPerspectiveIpm(bev_params);

    // Run detection on the IPM BGR image directly. EdgeExtractor converts to
    // grayscale internally (CLAHE + Sobel/steerable); the old grayscale road-mask
    // preprocess path zeroed too much lane paint and produced sparse edges.
    const cv::Mat work_bev = prepareBevForDetection(bev_image, bev_params);

    const auto t0 = std::chrono::steady_clock::now();
    const BevDetectionResult det =
      detectLanesInBev(bev_image, bev_params, /*configure_params=*/false);
    const auto t1 = std::chrono::steady_clock::now();
    const double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::vector<LaneHypothesis> lanes = det.lanes;
    applyTemporalSmoothing(lanes, bev_params, bev_image.rows, bev_image.cols);

    const cv::Mat overlay = drawOverlay(work_bev, lanes, det.merges);

    marker_pub_->publish(lanesToMarkers(lanes, frame_id_, header.stamp));
    merge_pub_->publish(mergesToMarkers(det.merges, frame_id_, header.stamp));
    publishCvImage(overlay_pub_, overlay, header);
    publishCvImage(edges_pub_, det.edges, header);
    publishCvImage(filtered_pub_, work_bev, header);

    {
      std::lock_guard<std::mutex> lock(sync_mutex_);
      pending_frontal_ = PendingFrontalOverlay{header, lanes, det.merges};
      tryPublishFrontalOverlay();
    }

    std_msgs::msg::String stats;
    stats.data = formatStats(det, total_ms, -1.0);
    stats_pub_->publish(stats);

    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000,
      "DETECT | %zu lanes | %.1f ms | %s",
      det.lanes.size(), total_ms, stats.data.c_str());
  }

  PipelineParams params_;
  std::string frame_id_;
  double max_detect_rate_hz_{0.0};

  std::mutex sync_mutex_;
  std::deque<StampKey> stamp_order_;
  std::unordered_map<StampKey, cv::Mat, StampKeyHash> frontal_cache_;
  std::unordered_map<StampKey, cv::Mat, StampKeyHash> homography_cache_;
  std::optional<PendingFrontalOverlay> pending_frontal_;

  // Detection worker + single-slot latest-frame-wins mailbox.
  struct BevJob
  {
    cv::Mat image;
    std_msgs::msg::Header header;
  };
  std::thread worker_;
  std::mutex job_mutex_;
  std::condition_variable job_cv_;
  std::optional<BevJob> pending_job_;
  bool running_{true};

  // Previous frame's (smoothed) lanes for temporal low-pass; worker-only.
  std::vector<LaneHypothesis> prev_lanes_;

  image_transport::Subscriber image_sub_;
  rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr homography_sub_;
  image_transport::Subscriber bev_sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr merge_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr stats_pub_;
  image_transport::Publisher overlay_pub_;
  image_transport::Publisher edges_pub_;
  image_transport::Publisher filtered_pub_;
  image_transport::Publisher frontal_overlay_pub_;
};

}  // namespace lie_lane_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lie_lane_detection::BevLaneDetectorNode>());
  rclcpp::shutdown();
  return 0;
}
