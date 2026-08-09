#include "lie_lane_detection/voting/sparse_accumulator.hpp"

#include <algorithm>

namespace lie_lane_detection
{

DenseAccumulator::DenseAccumulator(size_t cells) : cells_(cells, 0.0)
{
}

void DenseAccumulator::reset()
{
  std::fill(cells_.begin(), cells_.end(), 0.0);
}

void DenseAccumulator::add(size_t index, double weight)
{
  if (index < cells_.size()) {
    cells_[index] += weight;
  }
}

double DenseAccumulator::get(size_t index) const
{
  return index < cells_.size() ? cells_[index] : 0.0;
}

void DenseAccumulator::mergeInPlace(std::vector<double> & acc, const std::vector<double> & partial)
{
  if (acc.size() != partial.size()) {
    return;
  }
  for (size_t i = 0; i < acc.size(); ++i) {
    acc[i] += partial[i];
  }
}

}  // namespace lie_lane_detection
