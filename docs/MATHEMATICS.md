# Mathematical context and refinements

This note connects the **Lie-theoretic lane model** used in `lie_lane_detection` to broader mathematics, including the lecture:

**[Graded Lie algebras and families of algebraic curves](https://www.youtube.com/watch?v=TlAnOMjybJg)** — Beth Romano (CIRM, 2023)  
*(timestamp ~11:00 / `t=654s`: SL₃ grading → PGL₂ acting on ℙ⁴, elliptic curves from Thorne’s construction)*

---

## Important: what that video is (and is not)

Romano’s talk is **pure arithmetic geometry**, not autonomous driving. The goal is to understand **rational points on curves** and **Selmer groups** by:

1. Building a **graded Lie algebra** from a reductive group `H` and an automorphism `θ`
2. Extracting a **representation** `G ↷ V` from the `θ = −1` eigenspace `H₁`
3. Constructing **families of algebraic curves** as fibers of a restricted map (Slodowy / Thorne slices)

Our package solves a **different problem** (2D lane boundaries in BEV), but several structural ideas rhyme with Romano’s pipeline.

| Romano (arithmetic geometry) | `lie_lane_detection` (BEV lanes) |
|------------------------------|----------------------------------|
| Graded Lie algebra `⊕ H_j` | Split parameters `xi = (SE₂ part, κ, σ)` |
| Representation `G ↷ H₁` | Hough accumulator over `(vx, vy, ω)` then `(κ, σ)` |
| Orbit counting (Bhargava) | **Vote counting** in parameter space |
| Family of curves as fibers | **One curve per hypothesis** `γ(t; xi)` |
| Stable grading / nice orbits | NMS + `min_inlier_ratio` + lane separation |
| `exp` / `log` on Lie groups | Sophus `SE2::exp` / `.log()` in RANSAC GN |
| Restrict map to graded piece `H₁` | Stage A votes with deformation presets on `H₁`-like slice |

The video is useful as **conceptual background** for why we parameterize curves through a **Lie group action** and refine in **tangent space**, not as a drop-in algorithm for IPM lane detection.

---

## Our mathematical model (formal summary)

### State space

We use a 5-vector:

\[
\xi = (v_x, v_y, \omega, \kappa, \sigma) \in \mathbb{R}^5
\]

- **SE(2) block** `(v_x, v_y, ω)` — via Sophus, a group element \(g = \exp(\hat{\xi}_{SE2}) \in SE(2)\)
- **Deformation block** `(κ, σ)` — extrinsic to SE(2), applied in template coordinates before the group warp

### Template and curve map

Normalized forward parameter \(t \in [0,1]\), \(y = y_{\min} + t \cdot (y_{\max} - y_{\min})\), \(y_n = (y - y_{\min})/(y_{\max} - y_{\min})\):

\[
\gamma_0(t) = \begin{pmatrix} \kappa \, y_n^2 \, L + \sigma \, y_n \, L \\ y \end{pmatrix}, \quad L = y_{\max} - y_{\min}
\]

\[
\gamma(t; \xi) = g(\xi) \cdot \gamma_0(t)
\]

This is a **semidirect product** picture: deform the spine, then apply a rigid BEV motion.

### Detection as inverse problem

Given edge points \(\{p_i\}\), find \(\xi\) such that many \(p_i\) lie near \(\gamma(\cdot; \xi)\).

We approximate this in three Lie-theoretic steps:

1. **Voting** — discrete search on a graded parameter space (coarse SE(2), fine \((\kappa,\sigma)\))
2. **RANSAC** — robust fit with minimal samples; preserve \((\kappa,\sigma)\) from seed during minimal solves
3. **Gauss–Newton** — update SE(2) via **left multiplication** `exp(δ) · g` (manifold-aware), update \((\kappa,\sigma)\) in \(\mathbb{R}^2\)

---

## Parallels with Romano’s talk (timestamp ~11:00)

At [t ≈ 654 s](https://www.youtube.com/watch?v=TlAnOMjybJg&t=654s), Romano works through **`sl_n` with a ℤ/2 grading**:

- Involution \(\theta(A) = -A^\*\) (anti-diagonal flip)
- Fixed group: \(SO(n)\); eigenspace \(H_1\): symmetric matrices
- For \(n=3\): \(G \cong PGL_2\) acting on \(\mathbb{P}^4\) — the **same representation** appearing in Bhargava–Shankar elliptic-curve work
- Thorne’s construction: intersect Slodowy slice with \(H_1\) → **family of elliptic curves**

### Analogy for lane detection

| Thorne / Romano | Our pipeline |
|-----------------|--------------|
| Start with large Lie algebra `sl_n` | Full 5D `xi` space |
| Grade into eigenspaces `H_j` | **Stage A:** fix \((\kappa,\sigma) \approx 0\), search SE(2) |
| Restrict quotient map to `H₁` | **Stage B:** fix SE(2) peak, search \((\kappa,\sigma)\) |
| Fibers = algebraic curves | Each `xi` defines one lane polyline |
| Orbit count in representation | Hough vote sum per bin |
| “Stable” grading → well-behaved orbits | Peaks separated by `hypothesisDistance` + NMS |

Our **multi-deformation Stage-A presets** (straight, ±κ, ±σ) play a role similar to **not restricting to κ = σ = 0 only** when the true curve lives in a larger graded piece — this was the bug we fixed when curved lanes under-detected.

---

## Geodesics and distance (what we already do)

**Hypothesis distance** (NMS) mixes SE(2) geodesic length with Euclidean \((\kappa,\sigma)\):

```cpp
d(ξ_a, ξ_b) = \| \log(g_a^{-1} g_b) \| + \lambda_k |\kappa_a - \kappa_b| + \lambda_s |\sigma_a - \sigma_b|
```

**RANSAC GN** updates SE(2) on the manifold:

```text
g ← exp(δ_SE2) · g
κ ← κ + δ_κ,   σ ← σ + δ_σ
```

This matches the standard recipe from Lie-group optimization (see also MERL TR2008-031 on learning on Lie groups for invariant detection).

**Not yet implemented:** a true product-manifold metric treating \((\kappa,\sigma)\) as a separate chart with its own connection, or **parallel transport** of tangent updates between peel iterations (Romano emphasizes parallel transport on SPD manifolds in *different* work — ReManNet — but the idea transfers conceptually).

---

## Refinement roadmap (inspired by the lecture)

These are ordered by impact vs. implementation cost for **curved / merge / diverge** synthetic scenes.

### 1. Formal graded search (documented + mostly done)

Treat detection as optimization on a **graded parameter space**:

- **Grade 0 / coarse:** \(G = SE(2)\) — `(vx, vy, ω)`
- **Grade 1 / fine:** deformation \(D = \mathbb{R}^2\) — `(κ, σ)`

Stage A + B in `LieHoughVoter` implement this. Optional: expose `grading_mode` in yaml (`coarse_fine` vs `joint_5d` for ablations).

### 2. Stable-orbit peak selection (medium)

Romano’s “stable grading” ⇒ orbits with **closed orbits and finite stabilizer**.

Practical translation for Hough:

- Require peak **contrast** vs. neighbors (not just max vote count)
- Reject peaks whose supporting edges have **high normal variance** (unstable symmetry)

*Benefit:* fewer spurious lanes on noisy real IPM (TuSimple runs).

### 3. Transverse / tubular distance (high value for curves)

Slodowy slices are **transverse** to orbits. For lanes:

- Inlier cost = distance to \(\gamma\) measured **normal to the curve**, not just Euclidean point-to-polyline
- For near-vertical BEV lanes, normal ≈ horizontal; for large \(|\omega|\), use Jacobian of \(\gamma(t;\xi)\)

*Benefit:* better **ω recovery** on `02_curved_highway` (current gap vs. GT ω = 0.12).

### 4. Orbit-centric RANSAC scoring (medium)

Bhargava counts **orbits** in a representation, not arbitrary tuples.

- Score a hypothesis by **unique arc length covered** on the template, not raw edge count
- Weight votes by edge magnitude × **alignment with curve tangent**

*Benefit:* dashed lanes and partial occlusion (`06`, `09`).

### 5. Lie logarithm for bounded κ, σ (low)

Map \(\kappa,\sigma\) through a bounded chart (e.g. \(\atanh\) scaled to `[kappa_min, kappa_max]`) so GN steps respect limits without hard `clamp`.

*Benefit:* smoother convergence on `08_split_diverge` (large |σ|).

### 6. 3D / metric lifting (future, ReManNet-adjacent)

Romano’s talk is 1D curves over \(\mathbb{Q}\); ReManNet uses **SPD / Lie algebra** for 3D lane consistency.

If we extend beyond BEV pixels:

- Lift polylines to **metric 3D** via camera + ground plane
- Use **tunnel IoU**-style loss for evaluation (not just vx error)

This is **iteration 2+** scope, not current package.

---

## Evaluation: curve scenes vs. GT

After `generate_test_images --detect`, see `curve_eval.txt` per scene:

| Scene | Geometry tested | Key parameters |
|-------|-----------------|----------------|
| `02_curved_highway` | Constant κ, ω | κ, ω |
| `03_y_merge` | Opposing σ | σ |
| `08_split_diverge` | Opposing σ | σ |
| `09_combined_challenge` | κ + ω + occlusion | κ, ω |

**Current weak point:** ω on `02` (heading / tilt) — aligns with needing **transverse distance** and finer ω bins in Stage B, not more κ bins.

A complementary approach — **classical line Hough first, then Lie-Hough to group segments into curves** — is outlined in [LINE_TO_CURVE_HOUGH.md](LINE_TO_CURVE_HOUGH.md). Line tangents directly constrain ω and may address this weak point.

---

## Suggested reading (lane-relevant Lie theory)

| Resource | Relevance |
|----------|-----------|
| [Romano — Graded Lie algebras…](https://www.youtube.com/watch?v=TlAnOMjybJg) | Graded search, orbits ↔ curve families (conceptual) |
| [Symmetry detection via Lie-algebra voting](https://github.com/FlyingGiraffe/symmetry_detection) | Closest paper analog to Lie-Hough |
| [MERL — Learning on Lie Groups](https://www.merl.com/publications/docs/TR2008-031.pdf) | `exp`/`log` detection & tracking |
| [Sophus SE(2) docs](https://github.com/strasdat/Sophus) | Implementation of our SE(2) block |
| [ReManNet (CVPR 2026)](https://arxiv.org/abs/2603.19776) | SPD manifold + log map for **3D** lanes (different group, same “work in Lie algebra” principle) |
| [LINE_TO_CURVE_HOUGH.md](LINE_TO_CURVE_HOUGH.md) | Line Hough → Lie-curve grouping (design note) |

---

## Summary

- The YouTube lecture explains **graded Lie algebras → representations → families of curves** in number theory.
- Our detector uses the **same structural pattern** at engineering scale: **graded search on SE(2) then deformation**, **orbit-like voting**, **manifold refinement**.
- The math refinements that matter most for **curves** here are: **transverse distance**, **stable peak selection**, and **orbit-aware scoring** — not importing Bhargava’s orbit-counting machinery wholesale.

For package usage and pipeline stages, see [README.md](../README.md).
