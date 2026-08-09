#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/image.hpp>

#include <chrono>
#include <memory>
#include <string>

namespace lie_lane_detection
{

class VideoImagePublisherNode : public rclcpp::Node
{
public:
  VideoImagePublisherNode() : Node("video_image_publisher")
  {
    const std::string video_path = declare_parameter<std::string>(
      "video_path", "/home/mosal/Downloads/truck_videos/truck_highway_7m40.webm");
    const std::string image_topic =
      declare_parameter<std::string>("image_topic", "/camera/image_raw");
    frame_id_ = declare_parameter<std::string>("frame_id", "camera_front");
    publish_rate_hz_ = declare_parameter<double>("publish_rate_hz", 10.0);
    loop_ = declare_parameter<bool>("loop", true);
    target_width_ = declare_parameter<int>("target_width", 640);

    cap_.open(video_path);
    if (!cap_.isOpened()) {
      RCLCPP_FATAL(get_logger(), "Failed to open video: %s", video_path.c_str());
      throw std::runtime_error("video open failed");
    }

    const double fps = cap_.get(cv::CAP_PROP_FPS);
    const int frames = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_COUNT));
    const int width = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
    const int height = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));

    image_pub_ = create_publisher<sensor_msgs::msg::Image>(image_topic, 10);

    const auto period = std::chrono::duration<double>(1.0 / std::max(publish_rate_hz_, 0.1));
    timer_ = create_wall_timer(
      std::chrono::duration_cast<std::chrono::nanoseconds>(period),
      std::bind(&VideoImagePublisherNode::onTimer, this));

    RCLCPP_INFO(
      get_logger(),
      "Publishing %s (%dx%d, ~%d frames @ %.1f src fps) on %s at %.1f Hz, target_width=%d",
      video_path.c_str(), width, height, frames, fps, image_topic.c_str(), publish_rate_hz_,
      target_width_);
  }

private:
  void onTimer()
  {
    cv::Mat frame;
    if (!cap_.read(frame) || frame.empty()) {
      if (!loop_) {
        RCLCPP_INFO(get_logger(), "End of video, stopping publisher");
        timer_->cancel();
        return;
      }
      cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
      frame_index_ = 0;
      if (!cap_.read(frame) || frame.empty()) {
        RCLCPP_WARN(get_logger(), "Failed to loop video");
        return;
      }
    }

    if (target_width_ > 0 && frame.cols != target_width_) {
      const int target_h = static_cast<int>(std::lround(
        static_cast<double>(frame.rows) * target_width_ / static_cast<double>(frame.cols)));
      cv::resize(frame, frame, cv::Size(target_width_, target_h));
    }

    std_msgs::msg::Header header;
    header.stamp = now();
    header.frame_id = frame_id_ + "_" + std::to_string(frame_index_);

    cv_bridge::CvImage cv_image(header, "bgr8", frame);
    image_pub_->publish(*cv_image.toImageMsg());

    if (frame_index_ % 30 == 0) {
      RCLCPP_INFO(get_logger(), "Published frame %d (%dx%d)", frame_index_, frame.cols, frame.rows);
    }
    ++frame_index_;
  }

  cv::VideoCapture cap_;
  std::string frame_id_;
  double publish_rate_hz_{10.0};
  bool loop_{true};
  int target_width_{640};
  int frame_index_{0};

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace lie_lane_detection

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<lie_lane_detection::VideoImagePublisherNode>());
  rclcpp::shutdown();
  return 0;
}
