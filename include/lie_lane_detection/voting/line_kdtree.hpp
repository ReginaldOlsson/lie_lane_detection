#pragma once

#include "lie_lane_detection/core/types.hpp"

#include <cstddef>
#include <vector>

namespace lie_lane_detection
{

/// Simple 2D KD-tree over line segment midpoints for localized queries.
class LineKDTree
{
public:
  void build(const std::vector<LineSegment> & lines);

  /// Return indices of lines within radius of (qx, qy).
  std::vector<size_t> queryRadius(double qx, double qy, double radius) const;

  const std::vector<LineSegment> & lines() const { return lines_; }

private:
  struct Node
  {
    int axis{0};
    size_t index{0};
    int left{-1};
    int right{-1};
  };

  int buildRecursive(int begin, int end, int depth);
  void queryRecursive(
    int node, double qx, double qy, double radius_sq, std::vector<size_t> & out) const;

  std::vector<LineSegment> lines_;
  std::vector<size_t> indices_;
  std::vector<Node> nodes_;
};

}  // namespace lie_lane_detection
