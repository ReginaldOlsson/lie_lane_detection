#!/usr/bin/env python3
"""Publish Spatz BirdEyeParam for Boreas IPM BEV (sim-time aware).

Defaults match configureBoreasManualIpmSrc() / boreas_ortho_mosaic_offline:
  1448 x 1294 px @ 0.00698125 m/px  =>  resolution = 143.240859 px/m
  vehicle at bottom-center of BEV.

RViz birdeye_display expects /ipm/params next to /ipm/bev (same namespace).
"""

from __future__ import annotations

import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from spatz_interfaces.msg import BirdEyeParam


class BoreasBirdEyeParamPublisher(Node):
  def __init__(self) -> None:
    super().__init__('boreas_birdeye_param_publisher')

    # use_sim_time is provided by the ROS runtime / launch; do not redeclare it.
    self.declare_parameter('params_topic', '/ipm/params')
    self.declare_parameter('frame_id', 'camera_lidar')
    self.declare_parameter('publish_rate_hz', 1.0)
    # BEV size (px) from Boreas manual IPM autosize
    self.declare_parameter('height', 1294)
    self.declare_parameter('width', 1448)
    # Spatz resolution is px/m (= 1 / meters_per_px)
    self.declare_parameter('resolution', 143.240859)
    # RViz BirdeyeDisplay convention: vehicle on near edge, centered
    self.declare_parameter('offset_x', -724.0)
    self.declare_parameter('offset_y', 0.0)
    self.declare_parameter('offset_z', 0.0)

    topic = self.get_parameter('params_topic').get_parameter_value().string_value
    rate_hz = float(self.get_parameter('publish_rate_hz').value)
    rate_hz = max(rate_hz, 0.1)

    qos = QoSProfile(
      reliability=ReliabilityPolicy.RELIABLE,
      durability=DurabilityPolicy.TRANSIENT_LOCAL,
      history=HistoryPolicy.KEEP_LAST,
      depth=1,
    )
    self._pub = self.create_publisher(BirdEyeParam, topic, qos)
    self._timer = self.create_timer(1.0 / rate_hz, self._on_timer)

    use_sim = False
    if self.has_parameter('use_sim_time'):
      use_sim = self.get_parameter('use_sim_time').get_parameter_value().bool_value

    self.get_logger().info(
      f'Publishing BirdEyeParam on {topic} '
      f'(use_sim_time={use_sim}, '
      f'{int(self.get_parameter("width").value)}x'
      f'{int(self.get_parameter("height").value)}, '
      f'res={float(self.get_parameter("resolution").value):.6f} px/m)'
    )

  def _on_timer(self) -> None:
    msg = BirdEyeParam()
    msg.header.stamp = self.get_clock().now().to_msg()
    msg.header.frame_id = self.get_parameter('frame_id').get_parameter_value().string_value
    msg.height = int(self.get_parameter('height').value)
    msg.width = int(self.get_parameter('width').value)
    msg.resolution = float(self.get_parameter('resolution').value)
    msg.offset = Point(
      x=float(self.get_parameter('offset_x').value),
      y=float(self.get_parameter('offset_y').value),
      z=float(self.get_parameter('offset_z').value),
    )
    self._pub.publish(msg)


def main() -> None:
  rclpy.init()
  node = BoreasBirdEyeParamPublisher()
  try:
    rclpy.spin(node)
  except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
    pass
  finally:
    node.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == '__main__':
  main()
