# Automatic homography from frontal images

Forward-facing dashcam footage has lane markings that converge toward a **vanishing point** on the horizon. Lane detection in this package assumes a **bird's-eye view (BEV)** where lanes are roughly parallel vertical lines. Without IPM, overlays on raw frontal images look wrong because the Lie-Hough template does not match perspective geometry.

This module estimates IPM parameters automatically from a single frontal frame (or per-frame in video mode).

## Algorithm

1. **Edge detection** — Canny on the lower half of the image.
2. **Line segments** — `HoughLinesP`, filtered to near-vertical segments in the road region.
3. **Vanishing point** — Pairwise intersections of lane-like segments; vote in image space for the dominant VP above the horizon.
4. **IPM trapezoid** — Apex at the VP; bottom corners from `ipm_bottom_x_min_ratio` / `ipm_bottom_x_max_ratio`; top edge from rays through the VP (`ipm_top_y_*` clamps).
5. **Homography** — `ipm_src_points` / `ipm_dst_points` → `IPMTransformer::computeHomography()` → `warpPerspective`.

Metric BEV extent comes from `bev_width_m`, `bev_length_m`, and `bev_resolution_m_per_px` (caller / yaml). `updateIpmDstFromBevExtent()` rebuilds destination corners in meters.

If VP estimation fails (too few lines, low confidence), the code falls back to `setDefaultHighwayIpmRoi()`.

### Fixed-camera temporal smoothing

For video / online use, pass a persistent `VanishingPointTracker` to `estimateFrontalHomography()`.
The tracker assumes the dashcam is rigidly mounted, so VP should not jump frame-to-frame:

- Per-frame VP: Hough intersections + vote-histogram **centroid** (sub-pixel)
- Confidence: peak / (peak + second_peak)
- Temporal filter: 2D Kalman with low process noise
- Outlier gate: reject jumps > ~30 px (configurable); hold filtered VP on bad frames
- Debug: red circle = filtered VP, orange = raw frame VP when they differ

```cpp
VanishingPointTracker vp_tracker;
for (each frame) {
  auto hg = estimateFrontalHomography(frame, params, &vp_tracker);
}
```

## API

| Function                      | Purpose                                  |
| ----------------------------- | ---------------------------------------- |
| `estimateVanishingPoint()`    | VP only                                  |
| `configureAutoIpmRoi()`       | Build trapezoid from VP                  |
| `estimateFrontalHomography()` | Full pipeline: VP → H → BEV              |
| `warpFrontalAutoIpm()`        | Convenience warp with optional debug viz |

`FrontalHomographyResult` includes `H_img2bev`, `params.ipm_src_points`, `bev`, and `debug_roi` (trapezoid drawn on input).

## ROS two-node pipeline (recommended)

Split IPM from detection for independent tuning:

| Node                     | Input               | Output                                          |
| ------------------------ | ------------------- | ----------------------------------------------- |
| `frontal_ipm_node`       | `/camera/image_raw` | `/ipm/bev`, `/ipm/debug/roi`, `/ipm/homography` |
| `bev_lane_detector_node` | `/ipm/bev`          | `/lanes/detect/{overlay,edges,markers,stats}`   |

```bash
ros2 launch lie_lane_detection truck_bev_pipeline.launch.xml use_video_publisher:=true
```

Config: `config/ipm_converter.yaml` (metric extent + trapezoid ratios), `config/bev_lane_detector.yaml` (detection only).

**Truck defaults:** 20 m × 60 m BEV, bottom corners 2%–98% image width, `ipm_top_y_min_ratio: 0.32`.

Tune IPM without touching detection:

```bash
ros2 launch lie_lane_detection truck_bev_pipeline.launch.xml \
  bev_width_m:=24.0 bev_length_m:=70.0 use_video_publisher:=true
```

Compare `/ipm/debug/roi` (yellow trapezoid on camera) with `/ipm/bev` before adjusting detection thresholds.

`tracked_lane_detector_node` (monolithic IPM + detect) is **deprecated**; use the two-node chain above.

## Tools

```bash
# Dump homography + YAML snippet for lane_detector.yaml
estimate_frontal_homography --image frame.png --output /tmp/hg

# Single image with auto-IPM then lane detection
lane_detect_offline --image frame.png --auto-ipm --output /tmp/out

# Video: auto-IPM per sampled frame (default mode)
lane_detect_video_offline --video clip.mp4 --output /tmp/video --auto-ipm

# Outputs (auto-IPM):
#   edge_overlay.mp4 / line_overlay.mp4          — BEV with lanes
#   edge_frontal_overlay.mp4 / line_frontal_overlay.mp4 — original view with lanes reprojected

# Raw frontal (no warp) — poor overlays, for comparison only
lane_detect_video_offline --video clip.mp4 --output /tmp/video --frontal-raw
```

## Limitations

- Works best on **highway** scenes with clear lane markings and a visible horizon.
- VP can drift frame-to-frame; for production, temporal smoothing or a fixed homography from calibration is preferable.
- Does not replace camera intrinsics / ground-plane calibration when metric accuracy matters.
- Curved roads and heavy occlusion reduce VP confidence.

## Related

- Manual IPM ROI: `setDefaultHighwayIpmRoi()` in `lane_detection_runner.hpp`
- Lewis line-curve Hough: `docs/LEWIS_HOUGH_CURVES.md`
