#pragma once

#include <cstddef>

namespace lie_lane_detection
{

/// Scalar batch helpers (4-wide unroll); AVX path can replace these later.
inline void accumulateWeightedBatch4(
  double * dst,
  const double * weights,
  size_t count)
{
  size_t i = 0;
  for (; i + 4 <= count; i += 4) {
    dst[0] += weights[i];
    dst[1] += weights[i + 1];
    dst[2] += weights[i + 2];
    dst[3] += weights[i + 3];
  }
  for (; i < count; ++i) {
    dst[i] += weights[i];
  }
}

}  // namespace lie_lane_detection
