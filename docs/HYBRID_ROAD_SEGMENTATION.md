# Hybrid road segmentation (geometry + tiny AI)

Classical Lie-Hough + RANSAC is strong on clean asphalt but struggles when **snow, slush, and glare** create dense false edges. The hybrid path adds a **small ONNX segmenter** that runs on the BEV image _before_ edge extraction.

## Architecture

```
Frontal image ──IPM──► BEV
                         │
                         ├─► TinyRoadSegNet (ONNX, OpenCV DNN) ──► drivable mask
                         │         road / lane / snow / ignore
                         │
                         └─► masked BEV ──► EdgeExtractor ──► Lie-Hough ──► lanes
```

The geometric pipeline is unchanged; AI only **gates which pixels may produce edges**.

## Model classes

| ID  | Name   | Use                            |
| --- | ------ | ------------------------------ |
| 0   | ignore | sky, vehicles, curbs, off-road |
| 1   | road   | drivable surface               |
| 2   | lane   | lane paint                     |
| 3   | snow   | snow, slush, glare clutter     |

**Drivable mask** = `(road ∨ lane) ∧ ¬snow` using configurable thresholds in `PipelineParams`.

## C++ API

- `RoadFeatureSegmenter` — load ONNX, run on BEV, return `drivable_mask` + debug BGR
- `detectLanesInBev(..., RoadFeatureSegmenter*)` — optional hybrid gate
- Debug outputs: `*_road_mask.png`, `road_mask` imshow window

## Quick start

```bash
# 1) Export a tiny ONNX (random weights — tests plumbing only)
cd scripts
python3 road_mask_tiny.py export --output ../models/road_mask_tiny.onnx

# 2) Optional: weak labels from existing BEV frames (bootstrap)
python3 road_mask_tiny.py bootstrap-heuristic --images /tmp/bev_frames --output /tmp/bev_labels

# 3) Train on BEV + label PNGs (class id per pixel, 0..3)
python3 road_mask_tiny.py train --images /data/bev --labels /data/labels --output ../models/road_mask_tiny.onnx

# 4) Run offline pipeline with hybrid gate
colcon build --packages-select lie_lane_detection
./build/lie_lane_detection/lane_detect_dataset_offline \
  --dataset /path/to/boreas --road-mask-model models/road_mask_tiny.onnx --show
```

## Training data for snow

1. Run the offline tool to dump `*_bev.png` frames.
2. Label snow regions (class 3) in any paint tool; export single-channel PNG masks.
3. Mark road=1, lane paint=2, everything else=0.
4. Fine-tune with `road_mask_tiny.py train`.

Snow-heavy sequences should be **oversampled** in training. The loss uses higher weight on class 3.

## Parameters (`PipelineParams` / yaml)

| Param                                  | Default   | Meaning                 |
| -------------------------------------- | --------- | ----------------------- |
| `use_road_feature_segmenter`           | false     | Enable hybrid gate      |
| `road_segmenter_onnx_path`             | ""        | ONNX file               |
| `road_segmenter_input_width/height`    | 256 / 128 | Model input size        |
| `road_segmenter_road_threshold`        | 0.45      | Min P(road)             |
| `road_segmenter_lane_threshold`        | 0.35      | Min P(lane)             |
| `road_segmenter_snow_reject_threshold` | 0.55      | Reject if P(snow) above |

## Next steps

- GPU backend (`DNN_TARGET_CUDA`) for real-time
- Temporal mask smoothing across frames
- Distillation from a larger segmentation model
- Frontal + BEV ensemble (snow visible earlier in frontal view)
