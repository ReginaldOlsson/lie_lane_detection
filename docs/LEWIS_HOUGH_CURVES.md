# Lewis et al. (IVCNZ 2016) — Parabolic Hough + RHT

Reference: Baker, Lewis, Mills — *Power line detection using a randomized Hough transform with an improved accumulator* (IVCNZ 2016). Local copy: `/home/mosal/Downloads/ivcnz_lewis_final_4.pdf`.

This note maps the paper's ideas to `lie_lane_detection` and records what we adopted vs. what remains future work.

---

## Paper pipeline (summary)

1. **Perspective warp** — straighten curved structures before voting (analogous to our IPM / BEV).
2. **Edge thinning** — skeletonize binary edges so each pixel votes once.
3. **Anisotropic Gaussian blur** — suppress clutter perpendicular to expected curve orientation (power lines ≈ horizontal in warped view; lane markings ≈ vertical in BEV).
4. **Parabolic Hough accumulator** — vote for `(a, b, c)` in `x = a y² + b y + c` (or equivalent parameterization).
5. **Randomized Hough Transform (RHT)** — repeated epochs: random minimal samples → fit → inlier test → **average similar hypotheses** → peel inlier pixels.
6. **Spurious peak rejection** — composite arcs that fit only a short segment of the image are discarded (their Fig. 3).

---

## Mapping to our Lie-Hough pipeline

| Lewis (2016) | Our implementation | Status |
|--------------|-------------------|--------|
| Perspective warp | `IPMTransformer` / `warpPerspectiveToBev` | Done |
| Anisotropic blur | `edge_anisotropic_blur` — `GaussianBlur(7×3)` on BEV gray | Done |
| Edge thinning | `edge_thin` — morphological skeleton (`thinBinaryEdges`) | Done |
| Curve family | Deformable template + SE(2): `xi = (vx, vy, ω, κ, σ)` | Generalization |
| Hough voting | `LieHoughVoter` Stage A (SE2) + Stage B (κ, σ) | Done |
| Hypothesis averaging | `hough_hypothesis_merge_ratio` — vote-weighted ξ average in Stage B | Done |
| RHT epochs + peel | `use_iterative_peeling` + `peelEdgesNearCurve` | Done |
| RANSAC refine | `ManifoldRansac` on manifold | Extension beyond paper |
| Short-arc rejection | `min_inlier_y_coverage` — inliers must span ≥35% of image height | Done |

---

## Parameters (YAML)

```yaml
edge_anisotropic_blur: true   # Lewis-style directional smoothing
edge_thin: true               # one-pixel-wide edge skeleton
hough_hypothesis_merge_ratio: 0.85   # merge bins within 85% of peak votes
min_inlier_y_coverage: 0.35          # reject local composite fits
```

---

## Differences and extensions

**Richer curve model.** Lewis fits parabolas in warped coordinates. We use a graded Lie-algebra template (see [MATHEMATICS.md](MATHEMATICS.md)): SE(2) pose plus quadratic curvature and linear merge shear. This handles Y-merges and diverging lanes that a single parabola cannot.

**Manifold RANSAC.** After Hough peaks, we refine on the SE(2) manifold with Gauss–Newton and Euclidean updates on κ/σ. The paper uses RHT averaging only; RANSAC improves sub-bin accuracy and inlier labeling.

**Transverse distance (not yet coded).** Lewis scores inliers with distance normal to the curve. We still use Euclidean point-to-curve distance in `TemplateCurve::nearestPoint`. For strong curvature (synthetic scene `02_curved_highway`), ω recovery can lag ground truth; normal-distance inliers are the planned fix (documented in MATHEMATICS.md).

**SE(2) hypothesis merge.** Vote-weighted averaging is applied in Stage B (κ, σ). Stage A SE(2) peaks still use NMS; merging nearby SE(2) bins by geodesic average is optional future work.

---

## Expected effects

- **Thinner edges** → fewer duplicate votes, sharper Hough peaks, less smearing on curves.
- **Anisotropic blur** → less horizontal road texture clutter in BEV.
- **Hypothesis merge** → stabler κ/σ estimates (RHT-style noise reduction).
- **Y-coverage gate** → fewer false lanes from short edge clusters (Lewis Fig. 3 class).

---

## Related reading

- [MATHEMATICS.md](MATHEMATICS.md) — Lie algebra, graded structure, refinement roadmap
- [README.md](../README.md) — full pipeline and tuning guide
