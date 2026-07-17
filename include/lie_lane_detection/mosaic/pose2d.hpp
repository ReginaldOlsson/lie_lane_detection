#pragma once

#include <cmath>

namespace lie_lane_detection
{

/// Planar pose in a fixed world frame (e.g. map): x/y in meters, yaw about +Z.
struct Pose2d
{
  double x{0.0};
  double y{0.0};
  double yaw_rad{0.0};

  Pose2d compose(const Pose2d & local) const
  {
    const double c = std::cos(yaw_rad);
    const double s = std::sin(yaw_rad);
    Pose2d out;
    out.x = x + c * local.x - s * local.y;
    out.y = y + s * local.x + c * local.y;
    out.yaw_rad = yaw_rad + local.yaw_rad;
    return out;
  }

  Pose2d inverse() const
  {
    const double c = std::cos(yaw_rad);
    const double s = std::sin(yaw_rad);
    Pose2d out;
    out.x = -c * x - s * y;
    out.y = s * x - c * y;
    out.yaw_rad = -yaw_rad;
    return out;
  }
};

inline double normalizeAngleRad(double angle)
{
  while (angle > M_PI) {
    angle -= 2.0 * M_PI;
  }
  while (angle < -M_PI) {
    angle += 2.0 * M_PI;
  }
  return angle;
}

inline double yawFromQuaternion(double qx, double qy, double qz, double qw)
{
  return std::atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz));
}

}  // namespace lie_lane_detection
