# Boreas ortho mosaic (offline)

Build a global bird's-eye ortho mosaic from a Boreas rosbag by placing metric IPM tiles using TF `map → base_link`.

## Prerequisites

- Boreas rosbag with `/boreas/image/compressed`, `/tf`, and `/tf_static`
- Packaged Boreas calibration (`P_camera.txt`, `T_camera_lidar.txt`)

The Boreas bag TF tree is:

```
map -> applanix          (dynamic /tf)
applanix -> base_link    (static /tf_static)
base_link -> {lidar, camera_lidar}
```

There is no direct `map -> base_link` transform; the tool composes the chain at each image timestamp.

## Build

```bash
cd /path/to/autoware
colcon build --packages-select lie_lane_detection
source install/setup.bash
```

## Usage

```bash
./build/lie_lane_detection/boreas_ortho_mosaic_offline \
  --rosbag /home/mosal/rosbags/boreas_bag \
  --calib-dir install/lie_lane_detection/share/lie_lane_detection/config/boreas/calib \
  --output /tmp/boreas_ortho.png \
  --decimate 10 \
  --max-frames 200
```

### Options

| Flag                      | Default                    | Meaning                           |
| ------------------------- | -------------------------- | --------------------------------- |
| `--rosbag`                | (required)                 | Rosbag2 directory                 |
| `--calib-dir`             | (required)                 | Boreas `calib/` folder            |
| `--output`                | `/tmp/boreas_ortho.png`    | Output mosaic PNG                 |
| `--meta`                  | `<output>.yaml`            | Canvas origin, scale, frame count |
| `--image-topic`           | `/boreas/image/compressed` | Compressed camera topic           |
| `--parent-frame`          | `map`                      | Pose parent frame                 |
| `--child-frame`           | `base_link`                | Pose child frame                  |
| `--decimate`              | `5`                        | Process every Nth image           |
| `--max-frames`            | `500`                      | Cap placed frames                 |
| `--max-pose-delta-ms`     | `50`                       | Max TF lookup gap                 |
| `--pose-yaw-offset-deg`   | `0`                        | Body-frame yaw tweak              |
| `--pose-lateral-offset-m` | `0`                        | Body-frame lateral tweak          |
| `--pose-forward-offset-m` | `0`                        | Body-frame forward tweak          |

## Alignment notes

- IPM BEV uses the Boreas manual camera-ground trapezoid (`configureBoreasManualIpmSrc`).
- BEV body frame: **x = lateral (right +)**, **y = forward**, origin at bottom-center (vehicle).
- If lane paint appears sheared between tiles, try small `--pose-yaw-offset-deg` tweaks or switch `--child-frame` to `camera_lidar` for camera-centric placement.

## Long sequences

Use `--decimate` and `--max-frames` to limit memory. The canvas grows with driven distance at the IPM resolution (typically ~0.05–0.15 m/px).
