#!/usr/bin/env python3
"""Tiny BEV road/lane/snow segmenter for hybrid lie_lane_detection.

Classes (logit channel order):
  0 ignore  - sky, vehicles, curbs, off-road
  1 road    - drivable surface
  2 lane    - lane paint / markings
  3 snow    - snow, slush, glare clutter

Export ONNX for C++ OpenCV DNN:
  python3 road_mask_tiny.py export --output ../models/road_mask_tiny.onnx

Train on paired BEV + label PNGs (single-channel class ids 0..3):
  python3 road_mask_tiny.py train --images DIR --labels DIR --output model.onnx

Weak-label bootstrap from BEV RGB (no manual masks):
  python3 road_mask_tiny.py bootstrap-heuristic --images DIR --output labels_dir/
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Install torch: pip install torch torchvision") from exc


CLASS_NAMES = ("ignore", "road", "lane", "snow")
INPUT_W = 256
INPUT_H = 128
NUM_CLASSES = 4


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyRoadSegNet(nn.Module):
    """~180k params encoder-decoder for 256x128 BEV."""

    def __init__(self, num_classes: int = NUM_CLASSES) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, 24)
        self.enc2 = ConvBlock(24, 48)
        self.enc3 = ConvBlock(48, 64)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(64, 96)
        self.up2 = nn.ConvTranspose2d(96, 64, 2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 48, 2, stride=2)
        self.dec1 = ConvBlock(48 + 48, 48)
        self.up0 = nn.ConvTranspose2d(48, 24, 2, stride=2)
        self.dec0 = ConvBlock(24 + 24, 24)
        self.head = nn.Conv2d(24, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d2 = self.dec2(torch.cat([self.up2(b), e3], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e2], dim=1))
        d0 = self.dec0(torch.cat([self.up0(d1), e1], dim=1))
        return self.head(d0)


def export_onnx(output: Path, width: int, height: int) -> None:
    model = TinyRoadSegNet().eval()
    dummy = torch.randn(1, 3, height, width)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["bev_rgb"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
    )
    print(f"Exported {output} ({width}x{height}, {NUM_CLASSES} classes)")


def list_images(folder: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in exts)


def load_pair(
    img_path: Path, label_path: Path, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(img_path)
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    if label is None:
        raise FileNotFoundError(label_path)
    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    label = cv2.resize(label, (width, height), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb, label.astype(np.int64)


def bootstrap_heuristic(images_dir: Path, labels_dir: Path) -> None:
    """Cheap pseudo-labels from grayscale heuristics (bootstrap before real snow labels)."""
    labels_dir.mkdir(parents=True, exist_ok=True)
    for img_path in list_images(images_dir):
        bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        label = np.zeros((h, w), dtype=np.uint8)
        # Road band: lower half, darker than snow, above noise floor.
        road = (gray > 28) & (gray < 175)
        road[int(h * 0.55) :, :] = False
        label[road] = 1
        # Lane paint: bright thin structures on road.
        bright = cv2.morphologyEx(
            (gray > 165).astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)
        )
        label[bright > 0] = 2
        # Snow / glare: very bright, low local contrast.
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        local_std = cv2.absdiff(gray, blur)
        snow = (gray > 185) & (local_std < 18)
        label[snow] = 3
        # Bottom near field is usually road.
        label[int(h * 0.72) :, :] = np.maximum(label[int(h * 0.72) :, :], 1)
        out = labels_dir / f"{img_path.stem}_label.png"
        cv2.imwrite(str(out), label)
        print(f"heuristic label: {out.name}")


def train(images_dir: Path, labels_dir: Path, output: Path, epochs: int, lr: float) -> None:
    pairs: list[tuple[Path, Path]] = []
    for img_path in list_images(images_dir):
        label_path = labels_dir / f"{img_path.stem}_label.png"
        if not label_path.exists():
            label_path = labels_dir / f"{img_path.stem}.png"
        if label_path.exists():
            pairs.append((img_path, label_path))
    if not pairs:
        raise SystemExit("No image/label pairs found")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyRoadSegNet().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # Weight snow/ignore a bit higher to learn suppression.
    class_weights = torch.tensor([1.0, 1.0, 1.4, 1.8], device=device)

    for epoch in range(epochs):
        random.shuffle(pairs)
        losses = []
        for img_path, label_path in pairs:
            rgb, label = load_pair(img_path, label_path, INPUT_W, INPUT_H)
            x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            y = torch.from_numpy(label).unsqueeze(0).to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"epoch {epoch + 1}/{epochs}  loss={np.mean(losses):.4f}")

    model.eval()
    export_onnx(output, INPUT_W, INPUT_H)


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny road/lane/snow BEV segmenter")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="Export untrained/random ONNX for plumbing tests")
    p_export.add_argument("--output", type=Path, default=Path("models/road_mask_tiny.onnx"))
    p_export.add_argument("--width", type=int, default=INPUT_W)
    p_export.add_argument("--height", type=int, default=INPUT_H)

    p_boot = sub.add_parser("bootstrap-heuristic", help="Generate weak labels from BEV images")
    p_boot.add_argument("--images", type=Path, required=True)
    p_boot.add_argument("--output", type=Path, required=True)

    p_train = sub.add_parser("train", help="Train on image/label folders and export ONNX")
    p_train.add_argument("--images", type=Path, required=True)
    p_train.add_argument("--labels", type=Path, required=True)
    p_train.add_argument("--output", type=Path, default=Path("models/road_mask_tiny.onnx"))
    p_train.add_argument("--epochs", type=int, default=12)
    p_train.add_argument("--lr", type=float, default=3e-4)

    args = parser.parse_args()
    if args.cmd == "export":
        export_onnx(args.output, args.width, args.height)
    elif args.cmd == "bootstrap-heuristic":
        bootstrap_heuristic(args.images, args.output)
    elif args.cmd == "train":
        train(args.images, args.labels, args.output, args.epochs, args.lr)


if __name__ == "__main__":
    main()
