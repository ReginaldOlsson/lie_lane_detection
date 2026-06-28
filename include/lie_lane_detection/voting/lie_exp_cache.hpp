#pragma once

#include <cstdint>
#include <cmath>
#include <unordered_map>

#include <sophus/se2.hpp>

#include "lie_lane_detection/core/types.hpp"

namespace lie_lane_detection
{

/// Cache SE(2) exponentials for repeated bin/hypothesis evaluations.
class LieExpCache
{
public:
  Sophus::SE2d get(const XiVector & xi);

  void clear();

private:
  struct Key
  {
    int64_t a{0};
    int64_t b{0};
    int64_t c{0};

    bool operator==(const Key & other) const
    {
      return a == other.a && b == other.b && c == other.c;
    }
  };

  struct KeyHash
  {
    size_t operator()(const Key & k) const
    {
      return static_cast<size_t>(k.a ^ (k.b << 16) ^ (k.c << 32));
    }
  };

  static Key quantizeSe2(const XiVector & xi, double scale = 1000.0);

  std::unordered_map<Key, Sophus::SE2d, KeyHash> cache_;
};

}  // namespace lie_lane_detection
