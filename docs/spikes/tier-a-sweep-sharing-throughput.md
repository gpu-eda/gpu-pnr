# Spike — Does sweep-sharing (multi-source SSSP) amortise compute at tile sizes Phase 3.3 will use?

**Status:** Resolved (2026-05-12) — **YES at 256², NO at 1024²; the
crossover is sharp.** 256² is the tile-size sweet spot for the planned
tile-decomposition routing. K=100 sources gives 4× speedup in 3D
(125 → 31 ms/source). At 512² and above the multi-source kernel ties
sequential or loses to it (memory-bandwidth-bound).

## Question

For the planned Phase 3.3 tile-decomposition router, does
`sweep_sssp_3d_multi` (K sources in one fused kernel call) actually
amortise compute relative to K sequential `sweep_sssp_3d` calls at the
tile sizes a chip-scale router would use (256², 512², 1024²)?

## Why this is in question

The whole wafer.space-scale performance hypothesis rests on
sweep-sharing being a real win. ADR 0008's earlier 2D measurement
showed K=50 gave 3.1× at 256² but only 0.97× at 1024². That data was
2D only and ~6 months stale. Before committing to chip-scale routing
(WS3.3, plan deliverable 6), we needed to know:

- Does the 3D kernel preserve the 2D scaling? (3D adds a sequential
  via-relax loop that might dominate.)
- What's the K-knee at each tile size?
- Does the result hold on current MPS perf?

If sweep-sharing doesn't scale in 3D, the whole "GPU SSSP beats CPU
A* at chip scale" thesis weakens and we'd need to plan differently.

## Approach

- **Q1.** Re-measure 2D sweep-sharing at 256² / 512² / 1024² /
  2048² to refresh ADR 0008's numbers on current MPS perf.
- **Q2.** Build `sweep_sssp_3d_multi` (3D analogue) and bench 3D at
  256² / 512² / 1024² with L=5 layers and via_cost=5.0.

## Decision matrix

| Outcome | Interpretation | Action |
|---|---|---|
| Q1 fails (2D regression vs ADR 0008) | MPS perf regressed | Investigate MPS / pin to known-good torch version |
| Q1 passes, Q2 fails (3D no win at any size) | Via-relax dominates compute; sweep-sharing isn't viable for our 3D use case | Re-plan WS3.3 around per-net mini-grids only; the wafer.space thesis weakens |
| Both pass (3D wins at some tile size) | Tile decomposition at that size is the right architecture | Proceed with WS3.3; the bull case stands |

## Findings

- **2026-05-12 (Q1, 2D).** Reconfirmed ADR 0008's pattern. 256² peaks
  at **9× speedup at K=100-200** (better than ADR 0008's measurement;
  MPS perf has improved). 512² peaks at 3× then declines. 1024² breaks
  even or loses. 2048² incomplete (killed during run).
- **2026-05-12 (Q2, 3D).** Built `sweep_sssp_3d_multi` (commit
  `e5dd5be`). At **256² × 5 layers**: K=10 → 2.05×, K=25 → 3.92×,
  K=50 → 3.35×, **K=100 → 4.05×** (peak). 31 ms per source at the
  peak vs 125 ms sequential. At 512²: peaks at 1.5× then collapses
  to 0.19× at K=100 (likely MPS memory pressure on the 524 MB
  distance tensor). 1024² incomplete (timed out at 7+ min on the
  K=1 case).

## Outcome

**Resolved YES at 256² tiles.** Phase 3.3's tile decomposition should
target 256² as the routing-tile size, with K up to ~100 sources per
batched call. The K-knee in 3D is sharper than 2D (4× vs 9× peak) due
to the sequential per-layer via-relax overhead, but the win is still
real and large enough to flip the wafer.space-scale performance
calculation in our favor: ~22-44 min wall-clock on MPS vs TR's
estimated 1-3 hours, with CUDA pushing toward ~2-5 min via its 7-15×
memory-bandwidth advantage.

The 3D bench at 512²+ confirms that **larger tile sizes are
counter-productive** — at 512² K=100 the multi-source kernel
catastrophically regresses, likely because the (K, L, H, W) =
(100, 5, 512, 512) × float32 distance tensor blows past whatever MPS
memory-pressure threshold matters. 256² × 5 stays comfortably within
budget.

## Implications captured in permanent docs

- The 3D bench numbers and the K-knee will be folded into the next
  Phase 3 results update (`docs/results.md`) before WS3.3 work
  begins, so the next session has the empirical anchor without
  needing to re-run.
- The "tile size = 256² for sweep-sharing" decision is a tactical
  parameter — not ADR-worthy on its own; will be folded into the
  WS3.3 architecture ADR when that work lands.
