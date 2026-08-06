#include "lie_lane_detection/preprocessing/boreas_calib.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace lie_lane_detection
{
namespace
{

cv::Mat makeScaleMatrix(double sx, double sy)
{
  return (cv::Mat_<double>(3, 3) << sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0);
}

cv::Mat makeHorizontalFlipMatrix(int width_px)
{
  // x' = (width - 1) - x
  return (cv::Mat_<double>(3, 3) << -1.0, 0.0, static_cast<double>(width_px - 1), 0.0, 1.0, 0.0,
    0.0, 0.0, 1.0);
}

int decodeFlagForReduce(int reduce)
{
  switch (reduce) {
    case 2:
      return cv::IMREAD_REDUCED_COLOR_2;
    case 4:
      return cv::IMREAD_REDUCED_COLOR_4;
    case 8:
      return cv::IMREAD_REDUCED_COLOR_8;
    default:
      return cv::IMREAD_COLOR;
  }
}

std::string compressedTopicFromBase(const std::string & image_topic)
{
  if (image_topic.size() >= 11 &&
    image_topic.compare(image_topic.size() - 11, 11, "/compressed") == 0)
  {
    return image_topic;
  }
  return image_topic + "/compressed";
}

/// Bilinear sample on IPM src quad (BL, BR, TR, TL) at normalized (s,t) in [0,1]^2.
cv::Point2f sampleSrcQuad(
  const std::array<cv::Point2f, 4> & quad, double s, double t)
{
  const cv::Point2f bottom = (1.0f - static_cast<float>(s)) * quad[0] +
    static_cast<float>(s) * quad[1];
  const cv::Point2f top = (1.0f - static_cast<float>(s)) * quad[3] +
    static_cast<float>(s) * quad[2];
  return (1.0f - static_cast<float>(t)) * bottom + static_cast<float>(t) * top;
}

/// Draw an NxN cell grid inside the src ROI so perspective warp shows distortions on BEV.
void drawSrcGridOnImage(
  cv::Mat & image_bgr,
  const std::array<cv::Point2f, 4> & src_quad_proc,
  int grid_n,
  const cv::Scalar & color = cv::Scalar(0, 255, 255),
  int thickness = 1)
{
  if (image_bgr.empty() || grid_n < 1) {
    return;
  }
  for (int i = 0; i <= grid_n; ++i) {
    const double u = static_cast<double>(i) / static_cast<double>(grid_n);
    std::vector<cv::Point> col_line;
    std::vector<cv::Point> row_line;
    col_line.reserve(static_cast<size_t>(grid_n + 1));
    row_line.reserve(static_cast<size_t>(grid_n + 1));
    for (int j = 0; j <= grid_n; ++j) {
      const double v = static_cast<double>(j) / static_cast<double>(grid_n);
      const cv::Point2f p_col = sampleSrcQuad(src_quad_proc, u, v);
      const cv::Point2f p_row = sampleSrcQuad(src_quad_proc, v, u);
      col_line.emplace_back(
        static_cast<int>(std::lround(p_col.x)), static_cast<int>(std::lround(p_col.y)));
      row_line.emplace_back(
        static_cast<int>(std::lround(p_row.x)), static_cast<int>(std::lround(p_row.y)));
    }
    cv::polylines(
      image_bgr, std::vector<std::vector<cv::Point>>{col_line}, false, color, thickness,
      cv::LINE_AA);
    cv::polylines(
      image_bgr, std::vector<std::vector<cv::Point>>{row_line}, false, color, thickness,
      cv::LINE_AA);
  }
}

}  // namespace

/// Fast Boreas IPM: compressed image + camera_info -> BEV.
/// Decodes JPEG at reduced resolution, warps a capped BEV, drops stale frames.
class BoreasIpmNode : public rclcpp::Node
{
public:
  BoreasIpmNode()
  : Node("boreas_ipm_node")
  {
    const std::string image_topic =
      declare_parameter<std::string>("image_topic", "/boreas/image");
    const std::string camera_info_topic =
      declare_parameter<std::string>("camera_info_topic", "/boreas/camera_info");
    const std::string image_transport =
      declare_parameter<std::string>("image_transport", "compressed");
    const std::string bev_topic = declare_parameter<std::string>("bev_topic", "/ipm/bev");
    calib_dir_ = declare_parameter<std::string>("boreas_calib_dir", "");
    use_ground_ipm_ = declare_parameter<bool>("boreas_use_ground_ipm", false);
    flip_horizontal_ = declare_parameter<bool>("flip_horizontal", true);
    frame_id_ = declare_parameter<std::string>("frame_id", "camera");

    // 1 = full JPEG decode, 2/4/8 = OpenCV reduced-color decode (much cheaper).
    decode_reduce_ = declare_parameter<int>("decode_reduce", 4);
    if (decode_reduce_ != 1 && decode_reduce_ != 2 && decode_reduce_ != 4 && decode_reduce_ != 8) {
      throw std::runtime_error("boreas_ipm_node: decode_reduce must be 1, 2, 4, or 8");
    }
    max_bev_width_px_ = declare_parameter<int>("max_bev_width_px", 640);
    max_bev_height_px_ = declare_parameter<int>("max_bev_height_px", 800);
    warp_interpolation_ = declare_parameter<int>("warp_interpolation", cv::INTER_LINEAR);
    draw_src_grid_ = declare_parameter<bool>("draw_src_grid", true);
    src_grid_n_ = declare_parameter<int>("src_grid_n", 5);
    if (src_grid_n_ < 1) {
      src_grid_n_ = 1;
    }

    params_.bev_width_m = declare_parameter<double>("bev_width_m", 20.0);
    params_.bev_length_m = declare_parameter<double>("bev_length_m", 60.0);
    params_.bev_resolution_m_per_px =
      declare_parameter<double>("bev_resolution_m_per_px", 0.05);

    if (calib_dir_.empty()) {
      throw std::runtime_error("boreas_ipm_node: boreas_calib_dir is required");
    }
    if (!loadBoreasCalib(calib_dir_, calib_)) {
      throw std::runtime_error("boreas_ipm_node: failed to load Boreas calib from " + calib_dir_);
    }

    const auto qos = rclcpp::SensorDataQoS();
    bev_pub_ = create_publisher<sensor_msgs::msg::Image>(bev_topic, qos);
    camera_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
      camera_info_topic, qos,
      std::bind(&BoreasIpmNode::onCameraInfo, this, std::placeholders::_1));

    if (image_transport == "compressed") {
      const std::string compressed_topic = compressedTopicFromBase(image_topic);
      compressed_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        compressed_topic, qos,
        std::bind(&BoreasIpmNode::onCompressed, this, std::placeholders::_1));
      RCLCPP_INFO(
        get_logger(),
        "boreas_ipm_node: %s + %s -> %s [decode_reduce=%d, max_bev=%dx%d, calib=%s]",
        compressed_topic.c_str(), camera_info_topic.c_str(), bev_topic.c_str(), decode_reduce_,
        max_bev_width_px_, max_bev_height_px_, calib_dir_.c_str());
    } else {
      raw_sub_ = create_subscription<sensor_msgs::msg::Image>(
        image_topic, qos, std::bind(&BoreasIpmNode::onRawImage, this, std::placeholders::_1));
      RCLCPP_INFO(
        get_logger(),
        "boreas_ipm_node: %s (raw) + %s -> %s [decode_reduce=%d, max_bev=%dx%d, calib=%s]",
        image_topic.c_str(), camera_info_topic.c_str(), bev_topic.c_str(), decode_reduce_,
        max_bev_width_px_, max_bev_height_px_, calib_dir_.c_str());
    }

    worker_ = std::thread(&BoreasIpmNode::workerLoop, this);
  }

  ~BoreasIpmNode() override
  {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      shutting_down_ = true;
    }
    queue_cv_.notify_all();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  struct Job
  {
    std_msgs::msg::Header header;
    std::vector<uint8_t> compressed;
    cv::Mat raw_bgr;  // used only for raw transport path
    bool is_compressed{true};
  };

  void onCameraInfo(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(ipm_mutex_);
    latest_camera_info_ = msg;
    if (!ipm_ready_) {
      tryInitIpmLocked(*msg);
    }
  }

  bool tryInitIpmLocked(const sensor_msgs::msg::CameraInfo & info)
  {
    if (info.width > 0 && info.height > 0) {
      calib_.image_width = static_cast<int>(info.width);
      calib_.image_height = static_cast<int>(info.height);
    }
    if (info.k[0] > 0.0 && info.k[4] > 0.0) {
      calib_.P = cv::Mat::zeros(3, 4, CV_64F);
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          calib_.P.at<double>(r, c) = info.k[static_cast<size_t>(r * 3 + c)];
        }
      }
    }

    PipelineParams params = params_;
    cv::Mat H_full;
    const bool ok = use_ground_ipm_
                      ? configureBoreasGroundIpm(calib_, params, H_full, nullptr)
                      : configureBoreasManualIpmSrc(calib_, params, H_full, nullptr);
    if (!ok) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000, "Boreas IPM configuration failed from camera_info");
      return false;
    }

    IPMTransformer ipm(params);
    if (!ipm.computeHomography(&info)) {
      return false;
    }
    H_full = ipm.homography().clone();
    const int full_bev_w = ipm.bevWidthPx();
    const int full_bev_h = ipm.bevHeightPx();

    // Map reduced-resolution pixels -> full-res pixels, then apply full H.
    const double reduce = static_cast<double>(std::max(1, decode_reduce_));
    cv::Mat H = H_full * makeScaleMatrix(reduce, reduce);

    // Cap BEV size for cheap warp + publish.
    double bev_scale = 1.0;
    if (max_bev_width_px_ > 0) {
      bev_scale = std::min(bev_scale, static_cast<double>(max_bev_width_px_) / full_bev_w);
    }
    if (max_bev_height_px_ > 0) {
      bev_scale = std::min(bev_scale, static_cast<double>(max_bev_height_px_) / full_bev_h);
    }
    bev_scale = std::max(0.05, std::min(1.0, bev_scale));
    H = makeScaleMatrix(bev_scale, bev_scale) * H;

    const int bev_w = std::max(1, static_cast<int>(std::lround(full_bev_w * bev_scale)));
    const int bev_h = std::max(1, static_cast<int>(std::lround(full_bev_h * bev_scale)));
    if (flip_horizontal_) {
      H = makeHorizontalFlipMatrix(bev_w) * H;
    }

    params_ = params;
    H_img2bev_ = H;
    bev_size_ = cv::Size(bev_w, bev_h);

    // Src quad in process (reduced) coordinates for debug grid drawing.
    src_quad_proc_ = {};
    if (params_.ipm_src_points.size() >= 8) {
      const float inv_reduce = 1.0f / static_cast<float>(std::max(1, decode_reduce_));
      for (int i = 0; i < 4; ++i) {
        src_quad_proc_[static_cast<size_t>(i)] = cv::Point2f(
          static_cast<float>(params_.ipm_src_points[2 * i]) * inv_reduce,
          static_cast<float>(params_.ipm_src_points[2 * i + 1]) * inv_reduce);
      }
      has_src_quad_ = true;
    } else {
      has_src_quad_ = false;
    }

    ipm_ready_ = true;
    RCLCPP_INFO(
      get_logger(),
      "Boreas IPM ready: full %dx%d, decode 1/%d -> BEV %dx%d (%.1fx%.1f m, was %dx%d @ %.4f m/px)",
      calib_.image_width, calib_.image_height, decode_reduce_, bev_w, bev_h, params_.bev_width_m,
      params_.bev_length_m, full_bev_w, full_bev_h, params_.bev_resolution_m_per_px);
    return true;
  }

  void enqueue(Job && job)
  {
    {
      std::lock_guard<std::mutex> lock(queue_mutex_);
      latest_job_ = std::move(job);
      has_job_ = true;
    }
    queue_cv_.notify_one();
  }

  void onCompressed(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
  {
    Job job;
    job.header = msg->header;
    job.compressed = msg->data;
    job.is_compressed = true;
    enqueue(std::move(job));
  }

  void onRawImage(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge error: %s", e.what());
      return;
    }
    Job job;
    job.header = msg->header;
    job.raw_bgr = cv_ptr->image.clone();
    job.is_compressed = false;
    enqueue(std::move(job));
  }

  void workerLoop()
  {
    while (rclcpp::ok()) {
      Job job;
      {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] {return shutting_down_ || has_job_;});
        if (shutting_down_) {
          return;
        }
        job = std::move(latest_job_);
        has_job_ = false;
      }
      processJob(job);
    }
  }

  void processJob(const Job & job)
  {
    const auto t0 = std::chrono::steady_clock::now();

    cv::Mat H;
    cv::Size bev_size;
    std::array<cv::Point2f, 4> src_quad{};
    bool draw_grid = false;
    int grid_n = 5;
    {
      std::lock_guard<std::mutex> lock(ipm_mutex_);
      if (!ipm_ready_) {
        if (!latest_camera_info_ || !tryInitIpmLocked(*latest_camera_info_)) {
          RCLCPP_WARN_THROTTLE(
            get_logger(), *get_clock(), 2000,
            "Dropping image: Boreas IPM not ready (need camera_info)");
          return;
        }
      }
      H = H_img2bev_;
      bev_size = bev_size_;
      draw_grid = draw_src_grid_ && has_src_quad_;
      src_quad = src_quad_proc_;
      grid_n = src_grid_n_;
    }

    cv::Mat bgr;
    if (job.is_compressed) {
      if (job.compressed.empty()) {
        return;
      }
      const cv::Mat buf(1, static_cast<int>(job.compressed.size()), CV_8UC1,
        const_cast<uint8_t *>(job.compressed.data()));
      bgr = cv::imdecode(buf, decodeFlagForReduce(decode_reduce_));
      if (bgr.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "JPEG decode failed");
        return;
      }
    } else {
      bgr = job.raw_bgr;
      if (decode_reduce_ > 1) {
        cv::resize(
          bgr, bgr,
          cv::Size(
            std::max(1, bgr.cols / decode_reduce_), std::max(1, bgr.rows / decode_reduce_)),
          0.0, 0.0, cv::INTER_AREA);
      }
    }

    if (draw_grid) {
      drawSrcGridOnImage(bgr, src_quad, grid_n);
    }

    cv::Mat bev;
    cv::warpPerspective(bgr, bev, H, bev_size, warp_interpolation_, cv::BORDER_CONSTANT);
    if (bev.empty()) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000, "Boreas IPM warp failed");
      return;
    }

    std_msgs::msg::Header header = job.header;
    if (header.frame_id.empty()) {
      header.frame_id = frame_id_;
    }
    cv_bridge::CvImage cv_image(header, "bgr8", bev);
    bev_pub_->publish(*cv_image.toImageMsg());

    const double ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    ++frames_done_;
    proc_ms_sum_ += ms;
    const auto now = std::chrono::steady_clock::now();
    if (std::chrono::duration<double>(now - last_stats_).count() >= 2.0) {
      const double elapsed = std::chrono::duration<double>(now - last_stats_).count();
      const double hz = static_cast<double>(frames_done_) / elapsed;
      const double avg_ms = frames_done_ > 0 ? proc_ms_sum_ / static_cast<double>(frames_done_) : 0.0;
      RCLCPP_INFO(
        get_logger(), "IPM throughput: %.1f Hz (avg %.1f ms/frame, in %dx%d -> out %dx%d)", hz,
        avg_ms, bgr.cols, bgr.rows, bev.cols, bev.rows);
      frames_done_ = 0;
      proc_ms_sum_ = 0.0;
      last_stats_ = now;
    }
  }

  std::string calib_dir_;
  std::string frame_id_;
  bool use_ground_ipm_{false};
  bool flip_horizontal_{true};
  int decode_reduce_{4};
  int max_bev_width_px_{640};
  int max_bev_height_px_{800};
  int warp_interpolation_{cv::INTER_LINEAR};
  bool draw_src_grid_{true};
  int src_grid_n_{5};

  BoreasCalib calib_;
  PipelineParams params_;

  std::mutex ipm_mutex_;
  bool ipm_ready_{false};
  bool has_src_quad_{false};
  std::array<cv::Point2f, 4> src_quad_proc_{};
  cv::Mat H_img2bev_;
  cv::Size bev_size_;
  sensor_msgs::msg::CameraInfo::SharedPtr latest_camera_info_;

  std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  Job latest_job_;
  bool has_job_{false};
  bool shutting_down_{false};
  std::thread worker_;

  std::size_t frames_done_{0};
  double proc_ms_sum_{0.0};
  std::chrono::steady_clock::time_point last_stats_{std::chrono::steady_clock::now()};

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr bev_pub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr compressed_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr raw_sub_;
};

}  // namespace lie_lane_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<lie_lane_detection::BoreasIpmNode>());
  } catch (const std::exception & e) {
    fprintf(stderr, "boreas_ipm_node failed: %s\n", e.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
