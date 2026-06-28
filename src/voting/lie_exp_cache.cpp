#include "lie_lane_detection/voting/lie_exp_cache.hpp"

namespace lie_lane_detection
{

LieExpCache::Key LieExpCache::quantizeSe2(const XiVector & xi, double scale)
{
  Key k;
  k.a = static_cast<int64_t>(std::lround(xi[0] * scale));
  k.b = static_cast<int64_t>(std::lround(xi[1] * scale));
  k.c = static_cast<int64_t>(std::lround(xi[2] * scale));
  return k;
}

Sophus::SE2d LieExpCache::get(const XiVector & xi)
{
  const Key key = quantizeSe2(xi);
  const auto it = cache_.find(key);
  if (it != cache_.end()) {
    return it->second;
  }
  const Sophus::SE2d g = xiToSE2(xi);
  cache_.emplace(key, g);
  return g;
}

void LieExpCache::clear()
{
  cache_.clear();
}

}  // namespace lie_lane_detection
