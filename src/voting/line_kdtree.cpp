#include "lie_lane_detection/voting/line_kdtree.hpp"

#include <algorithm>
#include <cmath>

namespace lie_lane_detection
{

void LineKDTree::build(const std::vector<LineSegment> & lines)
{
  lines_ = lines;
  indices_.resize(lines_.size());
  for (size_t i = 0; i < indices_.size(); ++i) {
    indices_[i] = i;
  }
  nodes_.clear();
  if (!indices_.empty()) {
    buildRecursive(0, static_cast<int>(indices_.size()), 0);
  }
}

int LineKDTree::buildRecursive(int begin, int end, int depth)
{
  if (begin >= end) {
    return -1;
  }
  const int axis = depth % 2;
  const int mid = (begin + end) / 2;
  std::nth_element(
    indices_.begin() + begin, indices_.begin() + mid, indices_.begin() + end,
    [&](size_t a, size_t b) {
      const auto & la = lines_[a];
      const auto & lb = lines_[b];
      const double ka = axis == 0 ? la.mx : la.my;
      const double kb = axis == 0 ? lb.mx : lb.my;
      return ka < kb;
    });

  Node node;
  node.axis = axis;
  node.index = indices_[static_cast<size_t>(mid)];
  const int node_id = static_cast<int>(nodes_.size());
  nodes_.push_back(node);
  nodes_[static_cast<size_t>(node_id)].left = buildRecursive(begin, mid, depth + 1);
  nodes_[static_cast<size_t>(node_id)].right = buildRecursive(mid + 1, end, depth + 1);
  return node_id;
}

void LineKDTree::queryRecursive(
  int node, double qx, double qy, double radius_sq, std::vector<size_t> & out) const
{
  if (node < 0 || node >= static_cast<int>(nodes_.size())) {
    return;
  }
  const Node & n = nodes_[static_cast<size_t>(node)];
  const LineSegment & line = lines_[n.index];
  const double dx = line.mx - qx;
  const double dy = line.my - qy;
  if (dx * dx + dy * dy <= radius_sq) {
    out.push_back(n.index);
  }
  const double split_val = n.axis == 0 ? line.mx : line.my;
  const double q_val = n.axis == 0 ? qx : qy;
  const int first = q_val < split_val ? n.left : n.right;
  const int second = q_val < split_val ? n.right : n.left;
  queryRecursive(first, qx, qy, radius_sq, out);
  const double diff = q_val - split_val;
  if (diff * diff <= radius_sq) {
    queryRecursive(second, qx, qy, radius_sq, out);
  }
}

std::vector<size_t> LineKDTree::queryRadius(double qx, double qy, double radius) const
{
  std::vector<size_t> out;
  if (nodes_.empty()) {
    return out;
  }
  queryRecursive(0, qx, qy, radius * radius, out);
  return out;
}

}  // namespace lie_lane_detection
