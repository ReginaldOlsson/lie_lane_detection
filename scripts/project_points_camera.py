#!/usr/bin/env python3
"""Visualize Boreas IPM: camera FOV, image-plane ROI (src), and ground footprint (dst).

Uses the same ray–ground math as boreas_calib.cpp (P_camera + T_camera_lidar).
Plot axes: X = lateral (-y_left), Y = forward (x_fwd), Z = height (lidar z).

Example:
  python3 scripts/project_points_camera.py \\
    --dataset /home/mosal/Downloads/boreas-2020-11-26-13-58 \\
    --output /tmp/boreas_ipm_debug.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Manual IPM: axis-aligned src rectangle (BL, BR, TR, TL) — matches boreas_calib.cpp
DEFAULT_SRC_PX = [
    (500.0, 2048.0),  # BL
    (1948.0, 2048.0),  # BR
    (1948.0, 1147.0),  # TR
    (500.0, 1147.0),  # TL
]
CORNER_NAMES = ["BL", "BR", "TR", "TL"]


def load_matrix(path: Path, rows: int, cols: int) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    return data.reshape(rows, cols)


def load_boreas_calib(calib_dir: Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    p_full = load_matrix(calib_dir / "P_camera.txt", 4, 4)
    p = p_full[:3, :4]
    t = load_matrix(calib_dir / "T_camera_lidar.txt", 4, 4)
    w, h = 2448, 2048
    yaml_path = calib_dir / "camera0_intrinsics.yaml"
    if yaml_path.exists():
        import re

        text = yaml_path.read_text()
        m = re.search(r"image_size:\s*\[([0-9.]+),\s*([0-9.]+)\]", text)
        if m:
            w = int(round(float(m.group(1))))
            h = int(round(float(m.group(2))))
    return p, t, w, h


def lidar_ground_to_image(
    p: np.ndarray, t: np.ndarray, x_fwd: float, y_left: float
) -> tuple[float, float] | None:
    p_lidar = np.array([x_fwd, y_left, 0.0, 1.0], dtype=np.float64)
    p_cam = t @ p_lidar
    uvw = p @ p_cam
    w = uvw[2]
    if abs(w) < 1e-9:
        return None
    return float(uvw[0] / w), float(uvw[1] / w)


def image_to_lidar_ground(
    p: np.ndarray, t: np.ndarray, u: float, v: float
) -> tuple[float, float] | None:
    """Same 2×2 solve as boreasImageToLidarGround in boreas_calib.cpp."""
    m = p @ t
    m00, m01, m03 = m[0, 0], m[0, 1], m[0, 3]
    m10, m11, m13 = m[1, 0], m[1, 1], m[1, 3]
    m20, m21, m23 = m[2, 0], m[2, 1], m[2, 3]

    a11 = u * m20 - m00
    a12 = u * m21 - m01
    b1 = m03 - u * m23
    a21 = v * m20 - m10
    a22 = v * m21 - m11
    b2 = m13 - v * m23
    det = a11 * a22 - a12 * a21
    if abs(det) < 1e-12:
        return None

    x_fwd = (b1 * a22 - a12 * b2) / det
    y_left = (a11 * b2 - b1 * a21) / det
    if not (np.isfinite(x_fwd) and np.isfinite(y_left)):
        return None
    return float(x_fwd), float(y_left)


def camera_center_lidar(t: np.ndarray) -> np.ndarray:
    t_inv = np.linalg.inv(t)
    c_h = t_inv @ np.array([0.0, 0.0, 0.0, 1.0])
    return c_h[:3]


def pixel_ray_lidar(
    p: np.ndarray, t: np.ndarray, u: float, v: float
) -> tuple[np.ndarray, np.ndarray]:
    """Ray origin (camera center) and unit direction in lidar frame."""
    k = p[:3, :3]
    t_inv = np.linalg.inv(t)
    c = camera_center_lidar(t)
    d_cam = np.linalg.inv(k) @ np.array([u, v, 1.0], dtype=np.float64)
    d_cam /= np.linalg.norm(d_cam)
    r_inv = t_inv[:3, :3]
    d_lidar = r_inv @ d_cam
    d_lidar /= np.linalg.norm(d_lidar)
    return c, d_lidar


def ray_ground_hit(origin: np.ndarray, direction: np.ndarray) -> tuple[np.ndarray, float] | None:
    if abs(direction[2]) < 1e-9:
        return None
    t = -origin[2] / direction[2]
    point = origin + t * direction
    point[2] = 0.0
    return point, t


def lidar_to_plot_xyz(x_fwd: float, y_left: float, z: float = 0.0) -> np.ndarray:
    """Map lidar (x_fwd, y_left, z) → plot (lateral, forward, height)."""
    return np.array([-y_left, x_fwd, z], dtype=np.float64)


def image_plane_points_lidar(
    p: np.ndarray,
    t: np.ndarray,
    image_w: int,
    image_h: int,
    depth_cam_m: float,
) -> np.ndarray:
    """Four image corners placed on the camera image plane at metric depth."""
    k = p[:3, :3]
    t_inv = np.linalg.inv(t)
    corners_uv = [(0, 0), (image_w, 0), (image_w, image_h), (0, image_h)]
    pts = []
    for u, v in corners_uv:
        d_cam = np.linalg.inv(k) @ np.array([u, v, 1.0], dtype=np.float64)
        d_cam /= d_cam[2]
        p_cam = depth_cam_m * d_cam
        p_h = t_inv @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
        pts.append(p_h[:3])
    return np.array(pts)


def sample_image_plane_mesh(
    p: np.ndarray,
    t: np.ndarray,
    image_bgr: np.ndarray,
    nu: int = 48,
    nv: int = 32,
    depth_cam_m: float = 2.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coarse image-plane grid in plot coordinates + RGB colors."""
    h, w = image_bgr.shape[:2]
    k = p[:3, :3]
    t_inv = np.linalg.inv(t)
    us = np.linspace(0, w - 1, nu)
    vs = np.linspace(0, h - 1, nv)
    xx = np.zeros((nv, nu))
    yy = np.zeros((nv, nu))
    zz = np.zeros((nv, nu))
    colors = np.zeros((nv, nu, 3))

    for j, v in enumerate(vs):
        for i, u in enumerate(us):
            d_cam = np.linalg.inv(k) @ np.array([u, v, 1.0], dtype=np.float64)
            d_cam /= d_cam[2]
            p_cam = depth_cam_m * d_cam
            p_lidar = (t_inv @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]
            plot_pt = lidar_to_plot_xyz(p_lidar[0], p_lidar[1], p_lidar[2])
            xx[j, i] = plot_pt[0]
            yy[j, i] = plot_pt[1]
            zz[j, i] = plot_pt[2]
            bgr = image_bgr[int(round(v)), int(round(u))]
            colors[j, i] = bgr[::-1] / 255.0
    return xx, yy, zz, colors


def rodrigues_rotate(
    q: np.ndarray, pivot: np.ndarray, axis_unit: np.ndarray, theta: float
) -> np.ndarray:
    v = q - pivot
    k = axis_unit / np.linalg.norm(axis_unit)
    kxv = np.cross(k, v)
    kd = np.dot(k, v)
    return pivot + v * np.cos(theta) + kxv * np.sin(theta) + k * (kd * (1.0 - np.cos(theta)))


def find_hinge_rotation_to_ground(
    q: np.ndarray, pivot: np.ndarray, axis_unit: np.ndarray
) -> tuple[float, np.ndarray] | None:
    def z_at(theta: float) -> float:
        return float(rodrigues_rotate(q, pivot, axis_unit, theta)[2])

    prev_theta = -np.pi
    prev_z = z_at(prev_theta)
    for i in range(1, 361):
        theta = -np.pi + 2.0 * np.pi * i / 360.0
        z = z_at(theta)
        if prev_z * z <= 0.0 and abs(z - prev_z) > 1e-9:
            lo, hi = prev_theta, theta
            for _ in range(60):
                mid = 0.5 * (lo + hi)
                if z_at(lo) * z_at(mid) <= 0.0:
                    hi = mid
                else:
                    lo = mid
            theta_out = 0.5 * (lo + hi)
            ground = rodrigues_rotate(q, pivot, axis_unit, theta_out)
            if abs(ground[2]) > 1e-6:
                ground[2] = 0.0
            return theta_out, ground
        prev_theta = theta
        prev_z = z
    return None


def image_pixel_to_lidar(
    p: np.ndarray, t: np.ndarray, u: float, v: float, depth_cam_m: float
) -> np.ndarray:
    k = p[:3, :3]
    t_inv = np.linalg.inv(t)
    d_cam = np.linalg.inv(k) @ np.array([u, v, 1.0], dtype=np.float64)
    d_cam /= d_cam[2]
    p_cam = depth_cam_m * d_cam
    return (t_inv @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]


def camera_ray_direction(p: np.ndarray, u: float, v: float) -> np.ndarray | None:
    k = p[:3, :3]
    d = np.linalg.inv(k) @ np.array([u, v, 1.0], dtype=np.float64)
    if d[2] <= 1e-9:
        return None
    return d


def camera_xz_plane_y(p: np.ndarray, u_ref: float, v_ref: float, z_ref_m: float = 1.0) -> float:
    d = camera_ray_direction(p, u_ref, v_ref)
    if d is None:
        return 1.0
    t = z_ref_m / d[2]
    return float(t * d[1])


def image_to_camera_xz(
    p: np.ndarray, u: float, v: float, y_plane_m: float
) -> tuple[float, float] | None:
    d = camera_ray_direction(p, u, v)
    if d is None or abs(d[1]) < 1e-9:
        return None
    t = y_plane_m / d[1]
    if t <= 0.0:
        return None
    x_cam = float(t * d[0])
    z_cam = float(t * d[2])
    if z_cam <= 0.0:
        return None
    return x_cam, z_cam


def compute_dst_camera_xz(
    src_px: list[tuple[float, float]],
    p: np.ndarray,
) -> tuple[list[dict], float, float, bool]:
    """Dst trapezoid from ray ∩ camera XZ plane (Y = y_plane), anchored at (cx, v_bottom)."""
    cx = float(p[0, 2])
    v_bottom = src_px[0][1]
    y_plane = camera_xz_plane_y(p, cx, v_bottom, z_ref_m=1.0)

    corners = []
    for name, (u, v) in zip(CORNER_NAMES, src_px):
        hit = image_to_camera_xz(p, u, v, y_plane)
        if hit is None:
            raise RuntimeError(f"XZ plane intersection failed for {name}")
        x_cam, z_cam = hit
        corners.append(
            {
                "name": name,
                "u": u,
                "v": v,
                "x_cam": x_cam,
                "y_cam": y_plane,
                "z_cam": z_cam,
                "dst_x": -x_cam,
                "dst_y": z_cam,
            }
        )

    x_vals = [c["dst_x"] for c in corners]
    y_vals = [c["dst_y"] for c in corners]
    lat_center = 0.5 * (min(x_vals) + max(x_vals))
    y_min = min(y_vals)
    for c in corners:
        c["dst_x_norm"] = c["dst_x"] - lat_center
        c["dst_y_norm"] = c["dst_y"] - y_min

    lat_span = max(x_vals) - min(x_vals)
    fwd_span = max(y_vals) - y_min
    forward_ok = min(corners[2]["dst_y"], corners[3]["dst_y"]) > max(
        corners[0]["dst_y"], corners[1]["dst_y"]
    )
    return corners, lat_span * 1.02, fwd_span * 1.02, forward_ok


def plot_scene(
    p: np.ndarray,
    t: np.ndarray,
    image_bgr: np.ndarray,
    image_w: int,
    image_h: int,
    src_px: list[tuple[float, float]],
    output: Path | None,
    show: bool,
) -> None:
    corners, bev_w, bev_l, forward_ok = compute_dst_camera_xz(src_px, p)
    c = camera_center_lidar(t)
    c_plot = lidar_to_plot_xyz(c[0], c[1], c[2])
    depth_cam = float(np.mean([corners[0]["z_cam"], corners[1]["z_cam"]]))

    fig = plt.figure(figsize=(16, 8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax2d = fig.add_subplot(1, 2, 2)

    # --- 2D camera image with src ROI ---
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    ax2d.imshow(img_rgb)
    roi = np.array(src_px + [src_px[0]])
    ax2d.plot(roi[:, 0], roi[:, 1], color="yellow", linewidth=2, label="src ROI")
    for c in corners:
        ax2d.scatter(c["u"], c["v"], c="cyan", s=40, zorder=5)
        ax2d.text(
            c["u"] + 20,
            c["v"],
            c["name"],
            color="white",
            fontsize=9,
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )
    ax2d.set_title("Rectified camera image + src trapezoid")
    ax2d.set_xlabel("u (px)")
    ax2d.set_ylabel("v (px)")
    ax2d.legend(loc="upper right")

    # --- 3D: image plane with texture ---
    xx, yy, zz, facecolors = sample_image_plane_mesh(p, t, image_bgr, depth_cam_m=depth_cam)
    ax3d.plot_surface(
        xx, yy, zz, rstride=1, cstride=1, facecolors=facecolors, shade=False, alpha=0.85
    )

    # Image-plane ROI corners at same depth
    plane_corners = []
    k = p[:3, :3]
    t_inv = np.linalg.inv(t)
    for u, v in src_px:
        d_cam = np.linalg.inv(k) @ np.array([u, v, 1.0], dtype=np.float64)
        d_cam /= d_cam[2]
        p_cam = depth_cam * d_cam
        p_lidar = (t_inv @ np.array([p_cam[0], p_cam[1], p_cam[2], 1.0]))[:3]
        plane_corners.append(lidar_to_plot_xyz(p_lidar[0], p_lidar[1], p_lidar[2]))
    plane_corners = np.array(plane_corners)
    roi_closed = np.vstack([plane_corners, plane_corners[0]])
    ax3d.plot(
        roi_closed[:, 0],
        roi_closed[:, 1],
        roi_closed[:, 2],
        color="yellow",
        linewidth=2.5,
        label="src ROI (image plane)",
    )

    # Ground plane
    gx = np.linspace(-8, 8, 10)
    gy = np.linspace(0, max(45, bev_l + 5), 12)
    gxx, gyy = np.meshgrid(gx, gy)
    gzz = np.zeros_like(gxx)
    ax3d.plot_surface(gxx, gyy, gzz, color="gray", alpha=0.15, shade=False)

    # Camera
    ax3d.scatter(
        c_plot[0], c_plot[1], c_plot[2], color="black", s=80, marker="^", label="camera center"
    )

    # Dst corners on camera XZ plane, drawn in lidar frame for 3D context.
    t_inv = np.linalg.inv(t)
    ground_plot = []
    for corner in corners:
        p_cam = np.array([corner["x_cam"], corner["y_cam"], corner["z_cam"], 1.0])
        p_lidar = (t_inv @ p_cam)[:3]
        g_plot = lidar_to_plot_xyz(p_lidar[0], p_lidar[1], p_lidar[2])
        ground_plot.append(g_plot)

        origin, direction = pixel_ray_lidar(p, t, corner["u"], corner["v"])
        ip = plane_corners[CORNER_NAMES.index(corner["name"])]
        ax3d.plot(
            [c_plot[0], ip[0], g_plot[0]],
            [c_plot[1], ip[1], g_plot[1]],
            [c_plot[2], ip[2], g_plot[2]],
            color="orange",
            linestyle="--",
            alpha=0.75,
            linewidth=1.2,
        )
        ax3d.scatter(g_plot[0], g_plot[1], g_plot[2], color="red", s=60, marker="X")
        ax3d.text(g_plot[0], g_plot[1], g_plot[2] + 0.3, corner["name"], color="red", fontsize=8)

    if len(ground_plot) == 4:
        ground_plot = np.array(ground_plot)

        # Full dst footprint on ground (BL → BR → TR → TL).
        gq = np.vstack([ground_plot, ground_plot[0]])
        ax3d.plot(
            gq[:, 0],
            gq[:, 1],
            gq[:, 2],
            color="magenta",
            linewidth=2.5,
            label="dst footprint (ground)",
        )
        verts = [list(zip(gq[:, 0], gq[:, 1], gq[:, 2]))]
        ax3d.add_collection3d(Poly3DCollection(verts, color="magenta", alpha=0.12))

        def draw_wall(
            i_bottom: int,
            i_top: int,
            color: str,
            label: str,
            alpha: float = 0.18,
        ) -> None:
            """Connect src bottom/top on image plane to matching dst corners on ground."""
            src_bot = plane_corners[i_bottom]
            src_top = plane_corners[i_top]
            dst_bot = ground_plot[i_bottom]
            dst_top = ground_plot[i_top]

            # Src vertical edge (e.g. BL–TL on image plane).
            ax3d.plot(
                [src_bot[0], src_top[0]],
                [src_bot[1], src_top[1]],
                [src_bot[2], src_top[2]],
                color=color,
                linewidth=2.5,
            )
            # Dst vertical edge (e.g. BL–TL on ground).
            ax3d.plot(
                [dst_bot[0], dst_top[0]],
                [dst_bot[1], dst_top[1]],
                [dst_bot[2], dst_top[2]],
                color=color,
                linewidth=2.5,
            )
            # Src bottom → dst bottom, src top → dst top.
            ax3d.plot(
                [src_bot[0], dst_bot[0]],
                [src_bot[1], dst_bot[1]],
                [src_bot[2], dst_bot[2]],
                color=color,
                linestyle="-",
                linewidth=2.0,
                alpha=0.9,
            )
            ax3d.plot(
                [src_top[0], dst_top[0]],
                [src_top[1], dst_top[1]],
                [src_top[2], dst_top[2]],
                color=color,
                linestyle="-",
                linewidth=2.0,
                alpha=0.9,
            )
            wall = np.array([src_bot, src_top, dst_top, dst_bot])
            ax3d.add_collection3d(
                Poly3DCollection(
                    [list(zip(wall[:, 0], wall[:, 1], wall[:, 2]))],
                    color=color,
                    alpha=alpha,
                )
            )
            ax3d.plot([], [], [], color=color, linewidth=2.5, label=label)

        # Left wall: BL (0) ↔ TL (3).
        draw_wall(0, 3, "cyan", "left wall BL–TL (src ↔ dst)")
        # Right wall: BR (1) ↔ TR (2).
        draw_wall(1, 2, "lime", "right wall BR–TR (src ↔ dst)")

        # Top dst edge TL–TR and bottom src edge BL–BR (emphasize near/far).
        ax3d.plot(
            [ground_plot[3, 0], ground_plot[2, 0]],
            [ground_plot[3, 1], ground_plot[2, 1]],
            [ground_plot[3, 2], ground_plot[2, 2]],
            color="red",
            linewidth=3.0,
            linestyle="-",
            label="dst top TL–TR",
        )
        ax3d.plot(
            [plane_corners[0, 0], plane_corners[1, 0]],
            [plane_corners[0, 1], plane_corners[1, 1]],
            [plane_corners[0, 2], plane_corners[1, 2]],
            color="gold",
            linewidth=3.0,
            linestyle="-",
            label="src bottom BL–BR",
        )

    ax3d.set_xlabel("lateral (-y_left) [m]")
    ax3d.set_ylabel("forward (x_fwd) [m]")
    ax3d.set_zlabel("height (z) [m]")
    title = "Boreas IPM geometry (camera XZ plane)"
    if not forward_ok:
        title += " — check top/bottom forward order"
    ax3d.set_title(title)
    ax3d.view_init(elev=22, azim=-58)
    ax3d.set_xlim(-6, 6)
    ax3d.set_ylim(-5, max(50, bev_l + 5))
    ax3d.set_zlim(-0.5, 3.0)

    handles, labels = ax3d.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax3d.legend(by_label.values(), by_label.keys(), loc="upper left", fontsize=8)

    # Text summary
    summary_lines = [
        f"BEV auto-size: {bev_w:.2f} m wide × {bev_l:.2f} m long",
        f"Forward order OK: {forward_ok}",
        "Bottom/top = ray ∩ camera XZ plane (Y = y_plane via cx, bottom row)",
        "Ground hits (x_cam, z_cam) → dst (-x_cam, z_cam):",
    ]
    for c in corners:
        summary_lines.append(
            f"  {c['name']}: ({c['x_cam']:.2f}, {c['z_cam']:.2f}) on Y={c['y_cam']:.3f} → "
            f"({c['dst_x_norm']:.2f}, {c['dst_y_norm']:.2f})"
        )
    fig.text(0.02, 0.02, "\n".join(summary_lines), fontsize=8, family="monospace", va="bottom")

    fig.tight_layout()
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Boreas IPM src/dst geometry")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("/home/mosal/Downloads/boreas-2020-11-26-13-58"),
        help="Boreas sequence root (calib/ + camera/)",
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Rectified camera frame (default: first PNG in camera/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/tmp/boreas_ipm_geometry.png"),
        help="Save figure to this path",
    )
    parser.add_argument("--show", action="store_true", help="Open interactive matplotlib window")
    args = parser.parse_args()

    calib_dir = args.dataset / "calib"
    p, t, image_w, image_h = load_boreas_calib(calib_dir)

    if args.image is None:
        camera_dir = args.dataset / "camera"
        images = sorted(camera_dir.glob("*.png"))
        if not images:
            raise SystemExit(f"No PNG images in {camera_dir}")
        image_path = images[0]
    else:
        image_path = args.image

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise SystemExit(f"Failed to read image: {image_path}")

    print(f"Calib: {calib_dir}")
    print(f"Image: {image_path} ({image_w}x{image_h})")
    print("Src px (BL, BR, TR, TL):")
    for name, (u, v) in zip(CORNER_NAMES, DEFAULT_SRC_PX):
        print(f"  {name}: ({u:.1f}, {v:.1f})")

    # Round-trip sanity on one ground point
    rt = lidar_ground_to_image(p, t, 10.0, -2.0)
    if rt:
        back = image_to_lidar_ground(p, t, rt[0], rt[1])
        print(f"Round-trip (10, -2) → ({rt[0]:.1f},{rt[1]:.1f}) → {back}")

    plot_scene(p, t, image_bgr, image_w, image_h, DEFAULT_SRC_PX, args.output, args.show)


if __name__ == "__main__":
    main()
