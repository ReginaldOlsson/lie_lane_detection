# Line-first → Lie-curve Hough (design note)

**Idea:** Run classical **line Hough** first (OpenCV `HoughLines` / `HoughLinesP`) to get many short line segments, then run a **second Lie-Hough** stage that groups lines into curve hypotheses — one deformable lane boundary per consistent bundle of segments.

This is complementary to the current **edge-pixel → Lie-Hough** pipeline documented in [README.md](../README.md) and [LEWIS_HOUGH_CURVES.md](LEWIS_HOUGH_CURVES.md).

---

## Motivation

| Problem with edge-only Lie-Hough | How line-first helps |
|----------------------------------|----------------------|
| Every edge pixel votes → heavy accumulator, duplicate votes on thick edges | Lines collapse local edge runs into one primitive |
| Dashed lanes produce fragmented edge clouds | Each dash ≈ one segment; grouping finds the underlying curve |
| ω (heading) is weakly constrained by point distance alone | Each segment carries a **tangent angle** — direct ω constraint |
| Clutter (cracks, shadows) adds spurious votes | Random short lines rarely form a **coherent bundle** under one ξ |
| Tight curves need many κ bins | Locally straight segments + global κ fit is a natural multi-scale split |

The second stage answers: *given hundreds of lines, which subsets are tangent samples of the same lane curve?*

---

## Proposed pipeline

```mermaid
flowchart LR
  A[BEV image] --> B[Edge extraction]
  B --> C[OpenCV HoughLinesP]
  C --> D[LineSegment set]
  D --> E[Line-Lie-Hough voter]
  E --> F[Curve hypotheses ξ]
  F --> G[Manifold RANSAC]
  G --> H[Peel assigned lines / edges]
  H --> E
```

**Stage 1 — Classical line Hough (OpenCV)**

```cpp
cv::HoughLinesP(edges, lines, rho=1, theta=CV_PI/180, threshold, minLineLength, maxLineGap);
```

In BEV, lane markings are ~vertical → segments have angle θ ≈ π/2 + ω_local and midpoints `(x̄, ȳ)` along the forward axis.

Output: `LineSegment { x1,y1,x2,y2, length, angle, mx, my }`.

**Stage 2 — Lie-Hough on lines (new module)**

Instead of voting from edge pixels, each line segment `L` votes for ξ if the deformable template passes near `L` **and** tangent direction matches:

1. **Position constraint:** distance from `L` midpoint (or line-to-curve distance) < `vote_threshold_px`
2. **Tangent constraint:** `|angle(L) − tangentAngle(γ at t*)| < angle_threshold` where `t*` is closest template parameter to the midpoint

Vote weight: `line.length` (longer dashes = stronger evidence), optionally × edge magnitude along the segment.

Accumulator structure can reuse the existing two-stage design:

- **Stage A:** SE(2) bins `(vx, vy, ω)` from line midpoints + angles
- **Stage B:** deformation bins `(κ, σ)` — which bundle of lines shares the same quadratic + shear?

**Stage 3 — Multi-curve extraction**

Same patterns as today:

- NMS on ξ peaks
- RHT-style hypothesis merge ([LEWIS_HOUGH_CURVES.md](LEWIS_HOUGH_CURVES.md))
- Iterative **peeling**: assign inlier lines to best curve, remove, repeat
- Optional: refine with `ManifoldRansac` on supporting lines **and** nearby edge pixels

---

## Mathematics: line as a local Lie sample

For template `γ(t; ξ)` in BEV (see [MATHEMATICS.md](MATHEMATICS.md)):

- Midpoint `p = (x̄, ȳ)` should lie near `γ(t*; ξ)` for some `t* ∈ [0,1]`
- Tangent `τ(t*; ξ) = dγ/dt` should align with segment direction `d = (x2−x1, y2−y1)/‖d‖`

**Vote score for line L and hypothesis ξ:**

\[
w(L) \cdot \mathbb{1}\big[\|p - \gamma(t^*)\| < \delta\big] \cdot \mathbb{1}\big[|\angle(d) - \angle(\tau(t^*))| < \delta_\theta\big]
\]

where `w(L) = length(L)` and `t* = argmin_t ‖p − γ(t)‖` (reuse `TemplateCurve::nearestPoint`).

**Curve from a set of lines:** A peak in the `(κ, σ)` accumulator means *many* segments agree on the same global deformation while permitting different local `(vx, vy, ω)` along the chain — exactly the “family of lines → one curve” question.

**Relation to parabolic Hough (Lewis):** Lewis fits `x = ay² + by + c` in warped coordinates. Each line segment is a local linearization of that parabola. Stage 2 is a **Lie-group generalization** of “which lines belong to the same parabola?”

---

## Grouping strategies (implementation options)

### A. Accumulator voting (recommended first)

Mirror `LieHoughVoter` but input `vector<LineSegment>`:

- Fewer primitives than edges (often 50–500 vs 10k+ pixels)
- Tangent term stabilizes ω on curved scenes (e.g. synthetic `02_curved_highway`)
- Reuse TBB `parallel_reduce`, NMS, RHT merge

### B. Line-bundle RANSAC

1. Sample 2–3 lines at random
2. Solve for ξ that best fits their midpoints + angles (minimal solver)
3. Count inlier lines
4. Peel and repeat

Good when line count is moderate; less smooth than dense voting.

### C. Graph clustering

- Nodes = line segments
- Edge weight = compatibility score `C(L_i, L_j | ξ)` under shared κ, σ
- Connected components / spectral clustering → curve groups
- Refine each group with Lie-Hough or RANSAC

Useful for merge/split topologies where accumulator peaks overlap.

---

## Hybrid mode (practical default)

```
edges ──┬── HoughLinesP ──► LineLieHough ──► seeds
        └── subsampled edges ──────────────► ManifoldRansac refine
```

- **Seeds** from line-Lie-Hough (robust grouping, better ω)
- **Refinement** on edge pixels (sub-pixel lateral accuracy)
- **Inliers** = lines OR edges within threshold (union or intersection)

Toggle via `PipelineParams::use_line_primitives` (future).

---

## Expected wins and risks

**Wins**

- Better ω recovery on curves (tangent data per segment)
- Natural dashed-lane handling
- Faster Stage-2 voting (fewer primitives)
- Spurious composite peaks reduced before κ/σ search

**Risks**

- `HoughLinesP` may miss faint or highly curved segments (`minLineLength` tradeoff)
- Very tight curves: segments span different local ω → need wider angle tolerance or smaller `minLineLength`
- Parallel lanes close together: similar θ → rely on **vx** separation + `min_lane_separation_px`
- Two extra hyperparameters: `hough_line_threshold`, `min_line_length`, `max_line_gap`

---

## Suggested new types / files

```cpp
// types.hpp
struct LineSegment {
  double x1, y1, x2, y2;
  double length;
  double angle;   // radians, [0, π)
  double mx, my;  // midpoint
};

// line_hough_extractor.hpp — Stage 1
class LineHoughExtractor {
  std::vector<LineSegment> extract(const cv::Mat& edge_image) const;
};

// line_lie_hough_voter.hpp — Stage 2
class LineLieHoughVoter {
  std::vector<LaneHypothesis> vote(const std::vector<LineSegment>& lines) const;
};
```

Wire into `detectLanesInBev()` as an alternate seed path before `ManifoldRansac`.

---

## Evaluation plan

When implemented, compare on:

| Metric | Edge-only | Line → Lie |
|--------|-----------|------------|
| `02_curved_highway` ω error | baseline ~0.07 | target < 0.04 |
| `05_dashed_lanes` recall | baseline | expect ↑ |
| Primitives / frame | ~18k edges | ~100–400 lines |
| Stage A+B time | baseline | expect ↓ |

---

## Related docs

- [LEWIS_HOUGH_CURVES.md](LEWIS_HOUGH_CURVES.md) — RHT, thinning, hypothesis merge
- [MATHEMATICS.md](MATHEMATICS.md) — ξ, template γ, transverse distance (future inlier metric)
