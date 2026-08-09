#pragma once

#include <cstddef>
#include <vector>

namespace lie_lane_detection
{

/// Dense SE(2) vote accumulator without parallel_reduce heap copies.
class DenseAccumulator
{
public:
  explicit DenseAccumulator(size_t cells);

  void reset();
  void add(size_t index, double weight);
  double get(size_t index) const;
  size_t size() const { return cells_.size(); }

  /// Merge thread-local partial accumulators in-place (no allocation).
  static void mergeInPlace(std::vector<double> & acc, const std::vector<double> & partial);

  const std::vector<double> & data() const { return cells_; }

private:
  std::vector<double> cells_;
};

}  // namespace lie_lane_detection
