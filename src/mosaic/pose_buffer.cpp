#include "lie_lane_detection/mosaic/pose_buffer.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

namespace lie_lane_detection
{
namespace
{

std::string stripSlash(std::string frame)
{
  if (!frame.empty() && frame.front() == '/') {
    frame.erase(frame.begin());
  }
  return frame;
}

}  // namespace

void PoseBuffer::clear()
{
  samples_.clear();
}

void PoseBuffer::addSample(const int64_t stamp_ns, const Pose2d & pose)
{
  if (!samples_.empty() && stamp_ns < samples_.back().stamp_ns) {
    samples_.push_back({stamp_ns, pose});
    std::sort(samples_.begin(), samples_.end(), [](const PoseSample & a, const PoseSample & b) {
      return a.stamp_ns < b.stamp_ns;
    });
    return;
  }
  samples_.push_back({stamp_ns, pose});
}

std::optional<Pose2d> PoseBuffer::lookup(const int64_t stamp_ns, const int64_t max_delta_ns) const
{
  if (samples_.empty()) {
    return std::nullopt;
  }
  if (samples_.size() == 1) {
    const int64_t dt = std::abs(stamp_ns - samples_.front().stamp_ns);
    if (dt > max_delta_ns) {
      return std::nullopt;
    }
    return samples_.front().pose;
  }

  auto it = std::lower_bound(
    samples_.begin(), samples_.end(), stamp_ns,
    [](const PoseSample & sample, const int64_t t) { return sample.stamp_ns < t; });

  if (it == samples_.begin()) {
    const int64_t dt = samples_.front().stamp_ns - stamp_ns;
    if (dt > max_delta_ns) {
      return std::nullopt;
    }
    return samples_.front().pose;
  }
  if (it == samples_.end()) {
    const int64_t dt = stamp_ns - samples_.back().stamp_ns;
    if (dt > max_delta_ns) {
      return std::nullopt;
    }
    return samples_.back().pose;
  }

  const PoseSample & hi = *it;
  const PoseSample & lo = *(it - 1);
  const int64_t dt_hi = hi.stamp_ns - stamp_ns;
  const int64_t dt_lo = stamp_ns - lo.stamp_ns;
  if (dt_hi > max_delta_ns && dt_lo > max_delta_ns) {
    return std::nullopt;
  }

  const double span = static_cast<double>(hi.stamp_ns - lo.stamp_ns);
  if (span <= 0.0) {
    return lo.pose;
  }
  const double alpha = static_cast<double>(stamp_ns - lo.stamp_ns) / span;

  Pose2d out;
  out.x = lo.pose.x + alpha * (hi.pose.x - lo.pose.x);
  out.y = lo.pose.y + alpha * (hi.pose.y - lo.pose.y);
  const double dyaw = normalizeAngleRad(hi.pose.yaw_rad - lo.pose.yaw_rad);
  out.yaw_rad = normalizeAngleRad(lo.pose.yaw_rad + alpha * dyaw);
  return out;
}

void TfPoseResolver::clear()
{
  static_edges_.clear();
  dynamic_edges_.clear();
}

std::string TfPoseResolver::normalizeFrame(std::string frame)
{
  return stripSlash(std::move(frame));
}

Pose2d TfPoseResolver::transformToPose2d(const geometry_msgs::msg::TransformStamped & transform)
{
  Pose2d pose;
  pose.x = transform.transform.translation.x;
  pose.y = transform.transform.translation.y;
  pose.yaw_rad = yawFromQuaternion(
    transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z,
    transform.transform.rotation.w);
  return pose;
}

void TfPoseResolver::addStaticTransform(const geometry_msgs::msg::TransformStamped & transform)
{
  Edge edge;
  edge.parent = normalizeFrame(transform.header.frame_id);
  edge.child = normalizeFrame(transform.child_frame_id);
  edge.pose = transformToPose2d(transform);
  edge.is_static = true;
  static_edges_.push_back(std::move(edge));
}

void TfPoseResolver::addDynamicTransform(const geometry_msgs::msg::TransformStamped & transform)
{
  const std::string parent = normalizeFrame(transform.header.frame_id);
  const std::string child = normalizeFrame(transform.child_frame_id);
  const int64_t stamp_ns = static_cast<int64_t>(transform.header.stamp.sec) * 1000000000LL +
                           static_cast<int64_t>(transform.header.stamp.nanosec);

  auto it = std::find_if(
    dynamic_edges_.begin(), dynamic_edges_.end(),
    [&](const DynamicEdge & edge) { return edge.parent == parent && edge.child == child; });
  if (it == dynamic_edges_.end()) {
    DynamicEdge edge;
    edge.parent = parent;
    edge.child = child;
    dynamic_edges_.push_back(std::move(edge));
    it = dynamic_edges_.end() - 1;
  }
  it->buffer.addSample(stamp_ns, transformToPose2d(transform));
}

std::optional<Pose2d> TfPoseResolver::lookupStaticChain(
  const std::string & parent, const std::string & child) const
{
  if (parent == child) {
    return Pose2d{};
  }

  std::unordered_map<std::string, Pose2d> pose_from_parent;
  pose_from_parent[parent] = Pose2d{};
  bool changed = true;
  while (changed) {
    changed = false;
    for (const auto & edge : static_edges_) {
      if (pose_from_parent.count(edge.parent) == 0) {
        continue;
      }
      if (pose_from_parent.count(edge.child) > 0) {
        continue;
      }
      pose_from_parent[edge.child] = pose_from_parent.at(edge.parent).compose(edge.pose);
      changed = true;
    }
  }

  const auto it = pose_from_parent.find(child);
  if (it == pose_from_parent.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<Pose2d> TfPoseResolver::lookup(
  const std::string & parent_frame, const std::string & child_frame, const int64_t stamp_ns,
  const int64_t max_delta_ns) const
{
  const std::string parent = normalizeFrame(parent_frame);
  const std::string child = normalizeFrame(child_frame);
  if (parent == child) {
    return Pose2d{};
  }

  // Direct dynamic edge parent -> child.
  for (const auto & dyn : dynamic_edges_) {
    if (dyn.parent == parent && dyn.child == child) {
      return dyn.buffer.lookup(stamp_ns, max_delta_ns);
    }
  }

  // parent -> mid (dynamic) + mid -> child (static chain).
  for (const auto & dyn : dynamic_edges_) {
    if (dyn.parent != parent) {
      continue;
    }
    const auto mid_to_child = lookupStaticChain(dyn.child, child);
    if (!mid_to_child.has_value()) {
      continue;
    }
    const auto parent_to_mid = dyn.buffer.lookup(stamp_ns, max_delta_ns);
    if (!parent_to_mid.has_value()) {
      continue;
    }
    return parent_to_mid->compose(mid_to_child.value());
  }

  // Static-only chain.
  return lookupStaticChain(parent, child);
}

}  // namespace lie_lane_detection
