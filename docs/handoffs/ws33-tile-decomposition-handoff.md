# Handoff — WS3.3 guide-constrained sweep: track-pitch prototype measured

**Created:** 2026-05-28 (updated after the track-pitch sweep prototype;
supersedes the guide-region-mapper / track-pitch-finding handoff)
**Working tree:** clean (after this session's commit)
**Branch:** main

<!--
Reminder: a handoff is ephemeral. At resolution, every load-bearing piece
below migrates into a docs/adr/, docs/plans/, docs/spikes/, or design-doc
home, and this file is then `git rm`'d in the same commit as the migration.

See docs/handoff-discipline.md for the migration table.
-->

## Goal & next-up

**Goal of this session:** Follow-up 2 — prototype the **track-pitch
sweep** and measure real ms/net (the 0.24 ms/net figure was a *linear*
extrapolation that ignores fixed kernel-launch + convergence-sync
overhead), using a measure-first answer to the pin-access open question.

**Outcome (folded into `docs/results.md` Phase 3.3, "track-pitch sweep
prototype"):**
- Plumbed `pitch_dbu` through `build_chip_grid` + `rect_center_to_grid`
  (`scripts/_hazard3_io.py`); both defaulted to 200 DBU. 3 new tests.
- New `scripts/track_pitch_sweep_prototype.py` routes a sample of nets,
  each on its own `guide_region` sub-grid sliced from a 1120 DBU chip
  grid, via `route_multipin_nets_3d`.
- **Measured ms/net ≫ linear.** MPS median **16.8 ms/net** (vs 0.24
  linear, 70×); CPU median **2.57 ms/net**. CPU is **6.5× faster than
  MPS** at this grain — tiny ~4k-cell grids are launch/sync-bound, the
  GPU is starved. 200/200 in-cap nets routed. This **confirms the
  slot-scale spike**: per-net overhead is the binding constraint →
  batched small-grid kernel is the load-bearing next bet.
- **Pin access (measure-first):** intra-net pin merge is **0 at both 200
  and 1120 DBU** — coarsening to track pitch is cleared. The scary "100%
  cross-net collision" is a pitch-INVARIANT GCell-proxy artifact (guides
  are GCell-granular, not pin shapes); the per-net model avoids cross-net
  collision structurally. The formal pin-snap-vs-fine-region ADR
  amendment is gated on **DEF pin extraction**, not on the pitch change.

**Next session should pick up:** Follow-up 3 — the **batched small-grid
sweep kernel** (now the load-bearing bet, empirically). Prototype batching
K independent sub-grids in one kernel call vs K sequential single-grid
sweeps, at track pitch; measure ms/net. See
[`slot-scale-parallelism.md`](../spikes/slot-scale-parallelism.md). Likely
its own spike. Then follow-up 4 (rewrite the obsolete tile plan).

**Verification command:**

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/
# Expect: 91 passed

# Track-pitch sweep prototype (CPU is quick; default is MPS)
uv run python scripts/track_pitch_sweep_prototype.py --device cpu --sample 100
# Expect: 100/100 routed; intra-net merge 0 at both pitches;
#         CPU median ~2-3 ms/net (≫ the 0.24 linear figure)

grep -c "track-pitch sweep prototype" docs/results.md       # Expect: >=1
```

## Done this session

| Artifact | What | Notes |
|---|---|---|
| `scripts/_hazard3_io.py` | `pitch_dbu` param on `build_chip_grid` + `rect_center_to_grid` | default 200 (back-compat); pass 1120 for track pitch |
| `tests/test_hazard3_io.py` | 3 new pitch tests | suite 88 → 91 |
| `scripts/track_pitch_sweep_prototype.py` | track-pitch sweep prototype | per-net `guide_region` sub-grid route + pin-access geometry; `--pitch/--device/--sample` |
| `docs/results.md` Phase 3.3 | "track-pitch sweep prototype: measured ms/net" | the load-bearing result — measured ≫ linear, CPU>MPS, pin-access cleared |

## Open follow-ups (priority-ordered)

### 1. ✅ DONE — GRT guide-region mapper

`gpu_pnr.guides.guide_region`. Maps a net's guide rects → grid sub-grid
bbox. Validated on Hazard3 at both pitches.

### 2. ✅ DONE — Track-pitch sweep prototype

`scripts/track_pitch_sweep_prototype.py`. Measured ms/net ≫ the 0.24
linear figure (MPS 16.8, CPU 2.57 median); CPU beats MPS at this grain;
pin access cleared (intra-net merge 0 at both pitches). Folded into
`docs/results.md` Phase 3.3 "track-pitch sweep prototype". **Net result:
single-stream is overhead-bound → follow-up 3 is the real win.**

### 3. Batched small-grid sweep kernel (medium-large) — **next up**

The load-bearing GPU-parallelism bet, now empirically confirmed (the
prototype showed per-net launch/sync overhead dominates single-stream;
[`slot-scale-parallelism.md`](../spikes/slot-scale-parallelism.md)):
batch K independent small sub-grids in one kernel call vs K sequential
sweeps, at track pitch. Resolves whether the per-net launch overhead can
be amortized. May need its own spike. Distinct from the dead
K-batching-on-one-big-grid (Tier B).

### 4. Update WS3.3 plan (small)

`docs/plans/ws33-tile-router-implementation.md` is still the obsolete
8-slice fixed-tile plan. Rewrite around: track-pitch grid → per-net
guide-bbox sweep (option A) → batched small-grid kernel → coarsened
fallback for the ~6% over-cap tail.

### 5. Resolve CI bench baseline question (small, low priority)

Unchanged from prior handoff — Tier B already concluded environmental
regression; confirming Tier A's 4× on M2 at `e5dd5be` is optional.

## Critical context

**ADR 0012 Amendments 2 & 3 are the load-bearing docs.** Amendment 3 is
the one that changes the plan: the over-sampled grid, not the algorithm,
was the problem. Read it before touching the sweep.

**The `guide_region` mapper is pitch-agnostic**, and as of this session so
are `build_chip_grid` + `rect_center_to_grid` (`pitch_dbu` param, default
200). All three work at 1120.

**Pin access at track pitch is now measured, not feared.** Intra-net pin
merge is 0 at both 200 and 1120 DBU, so coarsening to track pitch does not
collapse any net's own pins — the pitch change is cleared. The tile
prototype's 21/27 failures were *cross-net* collisions on a *shared* grid,
which the per-net guide-constrained model structurally avoids (each net
routes alone). The genuine off-track-pin question needs **DEF pin
geometry** (the guide fixture is GCell-granular, not pin shapes); the
formal snap-vs-fine-region ADR amendment is gated on that, not on pitch.
See `docs/results.md` Phase 3.3 "Pin access at track pitch".

**`tile_router.py` (Slice 1) still has reusable geometry** —
`partition_chip`, `net_bbox`, `classify_nets`. Don't delete; the
guide-constrained router may repurpose the partition logic.

## References

- [ADR 0012](../adr/0012-tile-decomposition.md) Amendments 1–3 — the
  guide-constrained sweep design and the track-pitch pivot.
- [slot-scale-parallelism spike](../spikes/slot-scale-parallelism.md) —
  wafer.space 1×1 slot scaling; batched kernel is the next bet.
- [GPU vs DRT spike](../spikes/gpu-vs-drt-throughput.md) — search-space
  framing and DRT baseline.
- [`../results.md`](../results.md) Phase 3.3 — guide-region size
  distributions (200 vs 1120 DBU).
- [`../plans/ws33-tile-router-implementation.md`](../plans/ws33-tile-router-implementation.md)
  — obsolete 8-slice plan (follow-up 4 rewrites it).

## Migration note

When the remaining follow-ups land and this handoff resolves:

- Follow-up 2 (track-pitch prototype) → ✅ results already in
  `docs/results.md` Phase 3.3. Still open: the pin-access ADR amendment
  (close open question #1/#2), gated on DEF pin extraction.
- Follow-up 3 (batched kernel) → resolve the slot-scale spike + new ADR
  if the design is non-obvious.
- Follow-up 4 (plan rewrite) → updated
  `docs/plans/ws33-tile-router-implementation.md`.
- Then `git rm docs/handoffs/ws33-tile-decomposition-handoff.md` in the
  migration commit: `docs: resolve WS3.3 handoff — fold into plan +
  track-pitch prototype`.
