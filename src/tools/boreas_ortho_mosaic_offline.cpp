// Build a global ortho mosaic from a Boreas rosbag using TF map->base_link and IPM BEV tiles.
//
// Usage example:
//   boreas_ortho_mosaic_offline --rosbag /path/to/boreas_bag
//     --calib-dir /path/to/boreas/calib --output /tmp/boreas_ortho.png

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <tf2_msgs/msg/tf_message.hpp>

#include "lie_lane_detection/mosaic/odom_bev_mosaic_accumulator.hpp"
#include "lie_lane_detection/mosaic/pose_buffer.hpp"
#include "lie_lane_detection/preprocessing/boreas_calib.hpp"
#include "lie_lane_detection/preprocessing/ipm_transformer.hpp"

namespace fs = std::filesystem;
namespace lie = lie_lane_detection;

namespace
{

struct CliOptions
{
  fs::path rosbag;
  fs::path calib_dir;
  fs::path output_png = "/tmp/boreas_ortho.png";
  fs::path output_meta;
  std::string image_topic = "/boreas/image/compressed";
  std::string parent_frame = "map";
  std::string child_frame = "base_link";
  int decimate = 5;
  int max_frames = 500;
  double max_pose_delta_ms = 50.0;
  double pose_yaw_offset_deg = 0.0;
  double pose_lateral_offset_m = 0.0;
  double pose_forward_offset_m = 0.0;
};

bool parseArgs(int argc, char ** argv, CliOptions & opts)
{
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--rosbag" && i + 1 < argc) {
      opts.rosbag = fs::path(argv[++i]);
    } else if (arg == "--calib-dir" && i + 1 < argc) {
      opts.calib_dir = fs::path(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      opts.output_png = fs::path(argv[++i]);
    } else if (arg == "--meta" && i + 1 < argc) {
      opts.output_meta = fs::path(argv[++i]);
    } else if (arg == "--image-topic" && i + 1 < argc) {
      opts.image_topic = argv[++i];
    } else if (arg == "--parent-frame" && i + 1 < argc) {
      opts.parent_frame = argv[++i];
    } else if (arg == "--child-frame" && i + 1 < argc) {
      opts.child_frame = argv[++i];
    } else if (arg == "--decimate" && i + 1 < argc) {
      opts.decimate = std::max(1, std::stoi(argv[++i]));
    } else if (arg == "--max-frames" && i + 1 < argc) {
      opts.max_frames = std::max(1, std::stoi(argv[++i]));
    } else if (arg == "--max-pose-delta-ms" && i + 1 < argc) {
      opts.max_pose_delta_ms = std::max(0.0, std::stod(argv[++i]));
    } else if (arg == "--pose-yaw-offset-deg" && i + 1 < argc) {
      opts.pose_yaw_offset_deg = std::stod(argv[++i]);
    } else if (arg == "--pose-lateral-offset-m" && i + 1 < argc) {
      opts.pose_lateral_offset_m = std::stod(argv[++i]);
    } else if (arg == "--pose-forward-offset-m" && i + 1 < argc) {
      opts.pose_forward_offset_m = std::stod(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout <<
        "Usage: boreas_ortho_mosaic_offline --rosbag PATH --calib-dir DIR [options]\n"
        "  --output PATH              Output PNG (default /tmp/boreas_ortho.png)\n"
        "  --meta PATH                Optional YAML metadata sidecar\n"
        "  --image-topic TOPIC        Default /boreas/image/compressed\n"
        "  --parent-frame FRAME       Default map\n"
        "  --child-frame FRAME        Default base_link (composed via /tf + /tf_static)\n"
        "  --decimate N               Process every Nth image (default 5)\n"
        "  --max-frames N             Cap processed frames (default 500)\n"
        "  --max-pose-delta-ms MS     Max TF lookup gap (default 50)\n"
        "  --pose-yaw-offset-deg D    Body-frame yaw tweak for alignment\n"
        "  --pose-lateral-offset-m M  Body-frame lateral tweak\n"
        "  --pose-forward-offset-m M  Body-frame forward tweak\n";
      return false;
    }
  }

  if (opts.rosbag.empty() || opts.calib_dir.empty()) {
    std::cerr << "Provide --rosbag and --calib-dir\n";
    return false;
  }
  if (!fs::exists(opts.rosbag)) {
    std::cerr << "Rosbag not found: " << opts.rosbag << "\n";
    return false;
  }
  if (!fs::is_directory(opts.calib_dir) || !fs::exists(opts.calib_dir / "P_camera.txt")) {
    std::cerr << "Boreas calib dir missing P_camera.txt: " << opts.calib_dir << "\n";
    return false;
  }
  if (opts.output_meta.empty()) {
    opts.output_meta = opts.output_png;
    opts.output_meta.replace_extension(".yaml");
  }
  return true;
}

int64_t stampToNs(const builtin_interfaces::msg::Time & stamp)
{
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + static_cast<int64_t>(stamp.nanosec);
}

void loadTfFromBag(const fs::path & bag_path, lie::TfPoseResolver & resolver)
{
  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = bag_path.string();
  storage_options.storage_id = "sqlite3";

  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";

  rosbag2_storage::StorageFilter filter;
  filter.topics = {"/tf", "/tf_static"};

  rosbag2_cpp::Reader reader;
  reader.open(storage_options, converter_options);
  reader.set_filter(filter);

  rclcpp::Serialization<tf2_msgs::msg::TFMessage> serialization;

  while (reader.has_next()) {
    const auto bag_message = reader.read_next();
    if (bag_message->topic_name != "/tf" && bag_message->topic_name != "/tf_static") {
      continue;
    }

    tf2_msgs::msg::TFMessage msg;
    rclcpp::SerializedMessage serialized(*bag_message->serialized_data);
    serialization.deserialize_message(&serialized, &msg);

    for (const auto & transform : msg.transforms) {
      if (bag_message->topic_name == "/tf_static") {
        resolver.addStaticTransform(transform);
      } else {
        resolver.addDynamicTransform(transform);
      }
    }
  }
}

bool writeMetaYaml(const fs::path & path, const lie::OdomBevMosaicMeta & meta)
{
  std::ofstream out(path);
  if (!out) {
    return false;
  }
  out << "origin_map_x: " << meta.origin_map_x << "\n";
  out << "origin_map_y: " << meta.origin_map_y << "\n";
  out << "meters_per_px: " << meta.meters_per_px << "\n";
  out << "frames_accumulated: " << meta.frames_accumulated << "\n";
  out << "frames_skipped: " << meta.frames_skipped << "\n";
  out << "canvas_width_px: " << meta.canvas_bounds_px.width << "\n";
  out << "canvas_height_px: " << meta.canvas_bounds_px.height << "\n";
  return true;
}

}  // namespace

int main(int argc, char ** argv)
{
  CliOptions opts;
  if (!parseArgs(argc, argv, opts)) {
    return opts.rosbag.empty() ? 1 : 0;
  }

  lie::BoreasCalib boreas;
  if (!lie::loadBoreasCalib(opts.calib_dir, boreas)) {
    std::cerr << "Failed to load Boreas calibration\n";
    return 1;
  }

  lie::PipelineParams params;
  params.bev_resolution_m_per_px = 0.05;
  params.bev_bottom_exclude_px = 100.0;
  cv::Mat H_img2bev;
  lie::IPMTransformer ipm(params);
  if (!lie::configureBoreasManualIpmSrc(boreas, params, H_img2bev, &ipm)) {
    std::cerr << "Failed to configure Boreas manual IPM\n";
    return 1;
  }

  lie::TfPoseResolver resolver;
  std::cout << "Loading TF from " << opts.rosbag << "...\n";
  loadTfFromBag(opts.rosbag, resolver);

  lie::OdomBevMosaicParams mosaic_params;
  mosaic_params.meters_per_px = ipm.metersPerPixel();
  mosaic_params.bev_width_px = ipm.bevWidthPx();
  mosaic_params.bev_height_px = ipm.bevHeightPx();
  mosaic_params.pose_yaw_offset_rad = opts.pose_yaw_offset_deg * M_PI / 180.0;
  mosaic_params.pose_lateral_offset_m = opts.pose_lateral_offset_m;
  mosaic_params.pose_forward_offset_m = opts.pose_forward_offset_m;
  lie::OdomBevMosaicAccumulator accumulator(mosaic_params);

  const int64_t max_pose_delta_ns =
    static_cast<int64_t>(opts.max_pose_delta_ms * 1e6);

  rosbag2_storage::StorageOptions storage_options;
  storage_options.uri = opts.rosbag.string();
  storage_options.storage_id = "sqlite3";

  rosbag2_cpp::ConverterOptions converter_options;
  converter_options.input_serialization_format = "cdr";
  converter_options.output_serialization_format = "cdr";

  rosbag2_cpp::Reader reader;
  reader.open(storage_options, converter_options);

  rosbag2_storage::StorageFilter image_filter;
  image_filter.topics = {opts.image_topic};
  reader.set_filter(image_filter);

  rclcpp::Serialization<sensor_msgs::msg::CompressedImage> image_serialization;

  int image_index = 0;
  int processed = 0;
  int skipped_no_pose = 0;
  int skipped_decode = 0;

  std::cout << "Processing images from " << opts.image_topic << " (decimate=" << opts.decimate
            << ", max_frames=" << opts.max_frames << ")...\n";

  while (reader.has_next() && processed < opts.max_frames) {
    const auto bag_message = reader.read_next();
    if (bag_message->topic_name != opts.image_topic) {
      continue;
    }

    if (image_index % opts.decimate != 0) {
      ++image_index;
      continue;
    }
    ++image_index;

    sensor_msgs::msg::CompressedImage image_msg;
    rclcpp::SerializedMessage serialized(*bag_message->serialized_data);
    image_serialization.deserialize_message(&serialized, &image_msg);

    cv_bridge::CvImageConstPtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(image_msg, "bgr8");
    } catch (const cv_bridge::Exception & e) {
      std::cerr << "cv_bridge error: " << e.what() << "\n";
      ++skipped_decode;
      continue;
    }

    const cv::Mat bev = ipm.warpToBev(cv_ptr->image);
    if (bev.empty()) {
      ++skipped_decode;
      continue;
    }

    const int64_t stamp_ns = stampToNs(image_msg.header.stamp);
    const auto pose = resolver.lookup(
      opts.parent_frame, opts.child_frame, stamp_ns, max_pose_delta_ns);
    if (!pose.has_value()) {
      ++skipped_no_pose;
      continue;
    }

    const auto result = accumulator.accumulate(bev, pose.value());
    if (result.accepted) {
      ++processed;
      if (processed % 20 == 0) {
        std::cout << "  placed " << processed << " frames\n";
      }
    }
  }

  if (!accumulator.initialized() || accumulator.canvas().empty()) {
    std::cerr << "No frames accumulated (skipped_pose=" << skipped_no_pose
              << ", skipped_decode=" << skipped_decode << ")\n";
    return 1;
  }

  fs::create_directories(opts.output_png.parent_path());
  if (!cv::imwrite(opts.output_png.string(), accumulator.canvas())) {
    std::cerr << "Failed to write " << opts.output_png << "\n";
    return 1;
  }

  lie::OdomBevMosaicMeta meta = accumulator.meta();
  meta.frames_skipped = skipped_no_pose + skipped_decode;
  if (!writeMetaYaml(opts.output_meta, meta)) {
    std::cerr << "Warning: failed to write metadata " << opts.output_meta << "\n";
  }

  std::cout << "Wrote " << opts.output_png << " (" << meta.canvas_bounds_px.width << "x"
            << meta.canvas_bounds_px.height << " px, " << meta.frames_accumulated
            << " frames, m/px=" << meta.meters_per_px << ")\n";
  std::cout << "Metadata: " << opts.output_meta << "\n";
  return 0;
}
