# Plan — WS3.3 guide-constrained router implementation

**Status:** Proposed (2026-05-29). Supersedes the 8-slice fixed-tile +
K=100 + halo plan (in `git log` before this date), which
[ADR 0012](../adr/0012-tile-decomposition.md) Amendments 1–4 invalidated.

<!--
Status lifecycle: Proposed → Active → Closed (YYYY-MM-DD).
Companion to docs/plans/phase3-detailed-routing.md §WS3.3 (high-level
summary). This plan covers the build slicing only; design lives in ADR 0012.
-->

## Why this plan replaces the previous one

The original plan built a chip partitioned into **256² overlapping tiles**,
K=100-batched per tile, with halo reconciliation. ADR 0012's amendments
dismantled every pillar of that:

- **Amendment 1** — fixed tiles + K-batching are dead. Tier B showed
  `sweep_sssp_3d_multi` (K sources on one shared grid) collapses past 256²;
  the model becomes **per-net guide-constrained sub-grids**. No tiles, no
  halo, no owned-region assignment.
- **Amendment 3** — route on the **track pitch** (1120 DBU), not 200 DBU.
  This shrinks the median net ~31× and makes plain guide-bbox sweep (option
  A) viable for ~94% of nets.
- **Amendment 4** — the **batched small-grid sweep** (K *independent*
  grids in one call) is validated: 2.46–4.05× over sequential on MPS. This
  is the GPU-parallelism path, replacing K=100-on-one-grid.

## What is already built (the substrate)

| Piece | Where | State |
|---|---|---|
| Track-pitch chip grid | `build_chip_grid(pitch_dbu=1120)` (`scripts/_hazard3_io.py`) | ✓ shipped, 3 tests |
| Guide → sub-grid mapper | `guide_region` (`src/gpu_pnr/guides.py`) | ✓ shipped, validated both pitches |
| Batched small-grid kernel | `sweep_sssp_3d_batched` (`src/gpu_pnr/sweep.py`) | ✓ shipped, 5 tests, validated 2.46–4.05× |
| Per-net + batched prototypes | `scripts/track_pitch_sweep_prototype.py`, `scripts/batched_sweep_prototype.py` | ✓ measurement harnesses |

**The router is what remains** — composing these into a chip-scale
guide-constrained pass with cross-net conflict handling and the over-cap tail.

## Goal

Build the guide-constrained sweep router that implements ADR 0012 (as
amended) end-to-end and satisfies the WS3.3 exit criteria in
[phase3-detailed-routing.md §WS3.3](phase3-detailed-routing.md). Pipeline:
**track-pitch chip grid → per-net guide-bbox sub-grid → batched small-grid
sweep + backtrace → commit to shared `w_cur` → conflict detect / rip-up →
coarsened-pass fallback for the ~6% over-cap tail.**

## Prerequisites

- [ADR 0012](../adr/0012-tile-decomposition.md) Amendments 1–4.
- [ADR 0007](../adr/0007-hpwl-ascending-net-ordering.md) — net ordering.
- [ADR 0008](../adr/0008-defer-route-nets-batched.md) — conflict-detect /
  ripup, unlocked by Slice 4 (now on guide sub-grids, not tiles).
- The four substrate pieces above; 96 passing tests on `main`.

## Scope-out (not in this plan)

- **Convergence-masking** and **option-B size bucketing** — the two deferred
  throughput levers ADR 0012 Amendment 4 names. Worthwhile only after the
  router exists and profiles show the slowest-net-bounds-the-batch cost or
  padding waste dominates. Post-ship optimization.
- **DEF-driven pin extraction.** The guide fixture is GCell-granular, not pin
  shapes; the formal pin-snap-vs-fine-region decision (ADR 0012 Am.3 open
  Q#1) is gated on it. Orthogonal tooling, lands independently.
- CUDA port ([ADR 0001](../adr/0001-pytorch-mps-host.md) keeps MPS as host).
- Topology / Steiner lever for the via-ratio gap (ADR 0012 walk-back §).

## Design source of truth

[ADR 0012](../adr/0012-tile-decomposition.md) as amended. This plan cites it;
it never restates design choices.

## Module note

The router replaces the tile machinery in `src/gpu_pnr/tile_router.py`. Most
of that module (`Tile`, `partition_chip`, `assign_net_to_tile`,
`classify_nets`) is tile-specific and dies with the fixed-tile model;
`net_bbox` survives as the no-guide fallback region. **Open question (Slice
1):** rename the module to `guide_router.py` (clearer) vs keep the
ADR-0012-§7 `tile_router.py` name. Leaning rename.

---

## Slicing strategy

Six slices, each a PR-sized commit (≤500 LOC net), tests green on `main`,
independently reviewable. CI slices route synthetic small grids; Hazard3
measurement runs live in `scripts/` and write `docs/results.md` entries.

The risk order: Slice 3 (does batched routing on the *shared* grid reproduce
the spike's 2.46–4.05×?) and Slice 4 (does conflict/ripup converge?) are the
load-bearing measurements, landed before the terminal integration.

---

### Slice 1 — `GuideRouter` skeleton + per-net sub-grid classification

**Deliverable:** `GuideRouter` class + the per-net region assignment — no
routing yet. API mirrors `route_multipin_nets_3d`: takes
`nets: list[list[(l,r,c)]]` + guides, returns `list[MultiPin3DResult]` in
input order.

**Sketch:**
- For each net: `guide_region(...)` → sub-grid bbox. No-guide nets fall back
  to `net_bbox(pins)` + margin.
- Classify **in-cap** (both axes ≤ 256, ADR 0012 §1 max sub-grid) vs
  **over-cap / no-guide** (→ coarsened tail, Slice 5).
- HPWL-ascending order ([ADR 0007](../adr/0007-hpwl-ascending-net-ordering.md)).
- `GuideRouter.route` raises `NotImplementedError`.

**Tests** (`tests/test_guide_router.py`):
- in-cap vs over-cap classification on synthetic guides.
- no-guide net falls back to pin-bbox region; region contains all pins.
- every net lands in exactly one of {in-cap, tail}.

**Exit:** new test file passes; full suite green.

---

### Slice 2 — Single-stream guide-constrained route (baseline)

**Deliverable:** route each in-cap net on its own sub-grid sliced from the
shared `w_cur`, **sequentially**, with backtrace + commit (routed cells →
`inf` in `w_cur`, so later nets detour). This is the track-pitch prototype
formalised into the router on a *shared* obstacle grid — cross-net conflict
emerges from routing order. Multi-pin nets use the existing incremental
tree-growth (`route_multipin_nets_3d` on the sub-grid).

**Tests:**
- single net: result matches `route_multipin_nets_3d` on the sub-grid.
- two nets sharing a corridor: second detours around the first's committed
  cells (cross-net via shared `w_cur`); 0 conflicts.
- HPWL-ascending order honoured.

**Exit:** tests pass; running on a Hazard3 sample reproduces the track-pitch
prototype's ms/net and 0 cross-net conflicts (`docs/results.md` Phase 3.3).

**Risk/walk-back:** none (this is the validated single-stream path).

---

### Slice 3 — Batched routing via `sweep_sssp_3d_batched`

**Deliverable:** replace the sequential per-net sweep with batched groups.
Collect K independent in-cap nets, pad+stack their sub-grids (Amendment 4's
padded-stack), one `sweep_sssp_3d_batched`, backtrace each slice against its
own sub-grid, commit. Nets in one batch route against the **same `w_cur`
snapshot** (they don't see each other's commits); resulting conflicts are
handled in Slice 4.

**Multi-pin batching (the key design fork — open question):** the batched
kernel is single-source, but multi-pin nets need multiple attachment sweeps
(tree growth). Options:
- **(c) start here:** batch the 2-pin nets (the bulk per Hazard3's M1/M2
  distribution); route ≥3-pin nets sequentially via Slice 2's path. Matches
  the data, lowest risk.
- (a)/(b) refinement: batch round-*r* attachment sweeps across all nets still
  growing. More throughput, more bookkeeping. Defer unless ≥3-pin nets
  dominate the profile.

**Tests:**
- batched group route == per-net sequential (same `w_cur` snapshot) for
  spatially disjoint nets — identical cell sets.
- 2-pin / ≥3-pin split routes both correctly.

**Exit:** tests pass; Hazard3 batched ms/net beats Slice 2 single-stream,
reproducing the spike's batched win in-router (`docs/results.md` Phase 3.3).

**Risk/walk-back:** if backtrace dominates the batched sweep saving (CPU-side
per-net work not amortised), that's the signal to push backtrace onto the
GPU or revisit the batch grouping — record as an ADR 0012 amendment.

---

### Slice 4 — Cross-net conflict detect + rip-up / reroute

**Deliverable:** after a batch commits, detect cells claimed by ≥2 nets; keep
the lowest-HPWL net (ADR 0007), requeue the losers; reroute them against the
updated `w_cur` in a subsequent batch. Bounded rounds (≤3, then mark
`routed=False`). This is the [ADR 0008](../adr/0008-defer-route-nets-batched.md)
unlock, on guide sub-grids.

**Tests:**
- two nets conflict → lower-HPWL wins, loser reroutes around.
- unroutable-after-ripup → loser `routed=False`, winner `True`, no crash.
- no conflicts → no extra reroute batch (assert via a sweep-call counter).

**Exit:** tests pass; Hazard3 final committed set has **0 cross-net
conflicts**.

**Risk/walk-back:** if ripup doesn't converge within 3 rounds for >1% of
nets, raise the cap to 5; else leave them failed (caller falls back to the
existing per-net mini-grid).

---

### Slice 5 — Coarsened-pass fallback for the over-cap / no-guide tail

**Deliverable:** the ~6% over-cap (Amendment 3) and no-guide nets route on a
4× coarsened grid first, pinned as obstacles before the in-cap batched
passes. (This is the one piece that survives largely intact from the previous
plan's §6.)

**Sketch:**
- `_coarsen_grid(w_chip, factor=4)` — conservative obstacle pooling (any
  sub-cell `inf` → coarse `inf`).
- route tail nets on the coarse grid via `route_multipin_nets_3d` (small
  fraction; sequential is fine).
- `_refine_coarse_route` — expand each coarse cell to its 4×4 fine footprint;
  pin as `inf` in `w_chip` before in-cap passes.

**Tests:**
- coarsen pools obstacles correctly; refine yields a connected fine footprint.
- over-cap net pinned; an in-cap net detours around its footprint.

**Exit:** tests pass; on Hazard3 the coarsened pass routes ≥80% of tail nets
(ADR 0012 §5: loses M1 fidelity but fine for M3+ long-haul).

**Risk/walk-back:** if coarse success <50%, walk back per ADR 0012 §3 to
bbox-splitting; land the decision as an ADR 0012 amendment.

---

### Slice 6 — Terminal integration: Hazard3 chip-scale + WS3.3 exit criteria

**Deliverable:** `scripts/run_guide_router_hazard3.py` end-to-end +
`tests/test_guide_router_4096.py` correctness gate.

**Sketch:**
- Script: full Hazard3 fixture through `GuideRouter.route`; capture
  wall-clock, conflicts (must be 0), tail fraction, routed fraction,
  wire/via vs TritonRoute → `docs/results.md` Phase 3.3.
- Gate: 4096² × L=5 synthetic, ~500 nets, `GuideRouter` vs
  `route_multipin_nets_3d` on the same grid — routed-fraction parity (≤1%
  abs) and HPWL parity (≤5% mean). The WS3.3 no-regression criterion.

**Tests:** `test_guide_router_4096_no_regression_vs_unbatched` (slow; budget
≤2 min).

**Exit:** both WS3.3 boxes in phase3-detailed-routing.md flip to `[x]`:
1. 4096² routed by guide-constrained sweep, no quality regression vs the
   un-batched `route_multipin_nets_3d` baseline.
2. Whole-chip Hazard3 competitive with TritonRoute (≤1.2× wire, ≤1.2× vias).

**Risk/walk-back:** a miss points to a Slice 3/4/5 walk-back trigger; the
chosen walk-back lands as an ADR 0012 amendment, no re-architecting here.

---

## Resolution / migration

When Slice 6 ships:
- Flip the WS3.3 boxes in [`phase3-detailed-routing.md`](phase3-detailed-routing.md).
- Any architectural shift from Slices 3/4/5 walk-backs lands as an ADR 0012
  amendment in the **same commit**.
- Promote the deferred levers (convergence-masking, option-B bucketing, DEF
  pin extraction) into a Phase 4 sketch.
- `git rm docs/handoffs/ws33-tile-decomposition-handoff.md` per its own
  Migration note.

## Open questions

1. **Multi-pin batching strategy** (Slice 3): start with (c) 2-pin-batch +
   sequential ≥3-pin, or invest in (a)/(b) per-round attachment batching
   up front? Leaning (c); revisit on profile data.
2. **Module rename** (Slice 1): `tile_router.py` → `guide_router.py`?
   Leaning rename; touches ADR 0012 §7's "module structure" line.

## References

- [ADR 0012](../adr/0012-tile-decomposition.md) Amendments 1–4 — design
  source of truth (guide-constrained pivot, track pitch, batched kernel).
- [`../spikes/batched-small-grid-sweep.md`](../spikes/batched-small-grid-sweep.md)
  — batched sweep validated (Slice 3's foundation).
- [`../spikes/slot-scale-parallelism.md`](../spikes/slot-scale-parallelism.md)
  — slot-scale parallelism framing.
- [ADR 0007](../adr/0007-hpwl-ascending-net-ordering.md),
  [ADR 0008](../adr/0008-defer-route-nets-batched.md) — ordering; ripup unlock.
- [`phase3-detailed-routing.md`](phase3-detailed-routing.md) §WS3.3 —
  high-level summary + exit criteria.
- `scripts/track_pitch_sweep_prototype.py`,
  `scripts/batched_sweep_prototype.py` — reusable per-slice units-of-debug.
