# Handoff — WS3.3 guide-constrained sweep: mapper landed, track-pitch finding

**Created:** 2026-05-28 (updated after the guide-region mapper +
track-pitch finding; supersedes the 2026-05-28 bench/DRT-comparison
handoff)
**Working tree:** clean (after this session's commit)
**Branch:** main

<!--
Reminder: a handoff is ephemeral. At resolution, every load-bearing piece
below migrates into a docs/adr/, docs/plans/, docs/spikes/, or design-doc
home, and this file is then `git rm`'d in the same commit as the migration.

See docs/handoff-discipline.md for the migration table.
-->

## Goal & next-up

**Goal of this session:** Implement ADR 0012 Amendment 1 follow-up 1 (the
GRT guide-region mapper) and validate Amendment 1's throughput model on
real Hazard3 guides.

**Outcome:** Mapper shipped (`gpu_pnr.guides.guide_region`). Measuring it
exposed two things, both folded into ADR 0012:
- **Amendment 2:** a guide *bounding box* is a poor search-space proxy —
  a net's guides are a thin snake; the bbox is the enclosing rectangle.
- **Amendment 3 (load-bearing):** our cost tensor is sampled at 200 DBU
  but gf180mcuD routing tracks are at **1120 DBU** — 5.6×/axis (31×)
  over-sampling. Re-measuring at track pitch *validates* Amendment 1
  (median 4,332 cells ≈ the ~3k estimate; over-cap 48%→6%). The A/B/C
  fork dissolves into "adopt track pitch first; plain bbox sweep
  (option A) viable for ~94% of nets; options B/C are a deferred tail
  optimization."

**Next session should pick up:** Prototype the **track-pitch sweep** and
measure real ms/net (the 0.24 ms/net figure is a *linear* extrapolation;
the actual per-net cost includes fixed kernel-launch + convergence-sync
overhead that dominates tiny grids). Concrete first step:

1. Build a chip-scale cost grid at the track pitch — `build_chip_grid`
   with `pitch=1120` (decide per-layer: M1–M4=1120, M5=1800, origin
   offset 560 — ADR 0012 Amendment 3 open question #2).
2. Slice one net's `guide_region` sub-grid from it and route via
   `route_multipin_nets_3d`; compare ms/net to the 200 DBU
   `tile_decomp_prototype.py` baseline (54 ms/net).
3. **Decide pin access** (Amendment 3 open question #1): off-track pins
   on a 1120 DBU grid may mis-snap and re-introduce the pin-collision
   failures the tile prototype saw (21/27). Pin snapping or a
   locally-finer region around pins.

**Verification command:**

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/
# Expect: 88 passed

# Guide-region mapper + measurement at both pitches
uv run python scripts/measure_guide_regions.py --pitch 1120
# Expect: median 4,332 cells, 6.3% over the 256² cap

grep -c "Amendment 3" docs/adr/0012-tile-decomposition.md   # Expect: 1
test -f docs/spikes/slot-scale-parallelism.md && echo ok    # Expect: ok
```

## Done this session

| Artifact | What | Notes |
|---|---|---|
| `src/gpu_pnr/guides.py` | `GuideRegion` + `guide_region()` mapper | half-open bbox, contiguous layer span, `pitch_dbu` param, chip clamp |
| `tests/test_guides.py` | 11 unit tests | suite 77 → 88 |
| `scripts/measure_guide_regions.py` | Hazard3 size distribution | `--pitch` flag (200 vs 1120) |
| ADR 0012 Amendment 2 | bbox-is-a-snake-enclosure finding | + mapper conventions locked |
| ADR 0012 Amendment 3 | track-pitch finding (the resolution) | adopt track pitch; fork dissolved |
| `docs/results.md` Phase 3.3 | 200-vs-1120 size distributions | the load-bearing tables |
| `docs/spikes/slot-scale-parallelism.md` | wafer.space 1×1 slot scaling | ~350k nets, overhead-bound, batched kernel is next bet |

## Open follow-ups (priority-ordered)

### 1. ✅ DONE — GRT guide-region mapper

`gpu_pnr.guides.guide_region`. Maps a net's guide rects → grid sub-grid
bbox. Validated on Hazard3 at both pitches.

### 2. Track-pitch sweep prototype (medium) — **next up**

See "Next session should pick up" above. Validates the 0.24 ms/net
estimate against a real sweep at track pitch, and forces the pin-access
decision. Produces a `docs/results.md` entry.

### 3. Batched small-grid sweep kernel (medium-large)

The load-bearing GPU-parallelism bet per
[`slot-scale-parallelism.md`](../spikes/slot-scale-parallelism.md):
batch K independent small sub-grids in one kernel call vs K sequential
sweeps, at track pitch. Resolves whether per-net launch overhead (the
slot-scale bottleneck) can be amortized. May need its own spike. Distinct
from the dead K-batching-on-one-big-grid (Tier B).

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

**The `guide_region` mapper is pitch-agnostic.** It takes `pitch_dbu` as
a parameter, so it already works at 1120; no change needed there for the
track-pitch prototype.

**Pin access is the real risk of the track-pitch move.** A coarser grid
can mis-snap off-track pins. The tile prototype's 21/27 failures were pin
quantization at 200 DBU — a 1120 DBU grid is 5.6× coarser, so this needs
a deliberate answer (snapping or local fine region) before trusting
routability numbers.

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

When the track-pitch sweep prototype lands and this handoff resolves:

- Follow-up 2 (track-pitch prototype) → results in `docs/results.md`;
  pin-access decision → ADR 0012 Amendment (close open question #1/#2).
- Follow-up 3 (batched kernel) → resolve the slot-scale spike + new ADR
  if the design is non-obvious.
- Follow-up 4 (plan rewrite) → updated
  `docs/plans/ws33-tile-router-implementation.md`.
- Then `git rm docs/handoffs/ws33-tile-decomposition-handoff.md` in the
  migration commit: `docs: resolve WS3.3 handoff — fold into plan +
  track-pitch prototype`.
