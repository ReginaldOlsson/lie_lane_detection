#pragma once

#include "lie_lane_detection/mosaic/pose2d.hpp"

#include <geometry_msgs/msg/transform_stamped.hpp>

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace lie_lane_detection
{

struct PoseSample
{
  int64_t stamp_ns{0};
  Pose2d pose;
};

/// Time-ordered 2D pose buffer with linear interpolation.
class PoseBuffer
{
public:
  void clear();

  void addSample(int64_t stamp_ns, const Pose2d & pose);

  /// Returns interpolated pose at stamp_ns, or nullopt if outside range or gap too large.
  std::optional<Pose2d> lookup(int64_t stamp_ns, int64_t max_delta_ns) const;

  bool empty() const { return samples_.empty(); }
  size_t size() const { return samples_.size(); }

private:
  std::vector<PoseSample> samples_;
};

/// Minimal TF resolver for offline rosbag playback (parent -> child composition).
class TfPoseResolver
{
public:
  void clear();

  void addStaticTransform(const geometry_msgs::msg::TransformStamped & transform);
  void addDynamicTransform(const geometry_msgs::msg::TransformStamped & transform);

  /// Lookup planar pose of child in parent frame at stamp_ns.
  std::optional<Pose2d> lookup(
    const std::string & parent_frame, const std::string & child_frame, int64_t stamp_ns,
    int64_t max_delta_ns) const;

private:
  struct Edge
  {
    std::string parent;
    std::string child;
    Pose2d pose;
    bool is_static{false};
  };

  struct DynamicEdge
  {
    std::string parent;
    std::string child;
    PoseBuffer buffer;
  };

  static Pose2d transformToPose2d(const geometry_msgs::msg::TransformStamped & transform);
  static std::string normalizeFrame(std::string frame);

  std::vector<Edge> static_edges_;
  std::vector<DynamicEdge> dynamic_edges_;

  std::optional<Pose2d> lookupStaticChain(
    const std::string & parent, const std::string & child) const;
};

}  // namespace lie_lane_detection
