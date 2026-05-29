# Handoff — WS3.3 guide-constrained sweep: batched small-grid kernel resolved

**Created:** 2026-05-28; updated 2026-05-29 after the batched small-grid
sweep prototype (follow-up 3 resolved). Supersedes the track-pitch-prototype
handoff.
**Working tree:** clean (after this session's commit)
**Branch:** main

<!--
Reminder: a handoff is ephemeral. At resolution, every load-bearing piece
below migrates into a docs/adr/, docs/plans/, docs/spikes/, or design-doc
home, and this file is then `git rm`'d in the same commit as the migration.

See docs/handoff-discipline.md for the migration table.
-->

## Goal & next-up

**Goal of this session:** Follow-up 3 — prototype the **batched small-grid
sweep kernel** and measure whether packing K *independent* sub-grids into
one kernel call amortises the per-net launch+sync overhead the track-pitch
prototype found binding (the load-bearing open hypothesis of the slot-scale
spike).

**Outcome (new spike
[`batched-small-grid-sweep.md`](../spikes/batched-small-grid-sweep.md);
slot-scale spike's load-bearing hypothesis resolved YES):**
- New `sweep_sssp_3d_batched` (`src/gpu_pnr/sweep.py`): K *independent*
  per-net grids `(K,L,H,W)` in one fused call — the opposite of the dead
  Tier-B `_multi` (K sources on one *shared* grid). 4 new correctness tests.
- New `scripts/batched_sweep_prototype.py` times the per-net single-source
  sweep batched (padded stack) vs sequential, at track pitch.
- **Batching wins on GPU: MPS 2.46× (shuffled) to 4.05× (size-sorted)
  faster than sequential.** CPU is *slower* batched (0.15–0.41×) — no
  launch overhead to amortise, padding only adds work. The win is purely
  GPU-overhead amortisation, confirming the track-pitch diagnosis.
- **Padding waste is the option-A/B lever:** sorting cut waste 8.3×→3.2×
  and lifted the win 2.46×→4.05×. Option A (padded stack) already wins;
  option B (bucketing) is a ~1.6× deferred gain, not a prerequisite —
  matches ADR 0012 Amendment 3's framing.

**Next session should pick up:** Follow-up 4 — **rewrite the obsolete WS3.3
plan** (`docs/plans/ws33-tile-router-implementation.md`) around: track-pitch
grid → per-net guide-bbox sweep (option A) → **batched small-grid kernel
(now validated)** → coarsened fallback for the ~6% over-cap tail. Still
open beyond that: convergence-masking + option-B bucketing as the next
throughput levers (see the new spike's "next levers"), and the pin-access
ADR amendment gated on DEF pin extraction.

**Verification command:**

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/
# Expect: 95 passed

# Batched kernel correctness + the batched-vs-sequential measurement
uv run pytest tests/test_sweep_3d.py -k batched               # Expect: 4 passed
uv run python scripts/batched_sweep_prototype.py --device mps --sample 128 --batch 16
# Expect: batched ~2.5× faster than sequential (MPS); correctness Δdist 0

grep -c "Resolved YES" docs/spikes/batched-small-grid-sweep.md  # Expect: >=1
```

## Done this session

| Artifact | What | Notes |
|---|---|---|
| `src/gpu_pnr/sweep.py` | `sweep_sssp_3d_batched` | K independent per-net grids `(K,L,H,W)`, one source/net, batch-wide seg_barrier |
| `tests/test_sweep_3d.py` | 4 batched correctness tests | same-size, variable-size padded, anisotropic, bad-shape; suite 91 → 95 |
| `scripts/batched_sweep_prototype.py` | batched vs sequential sweep harness | `--device/--sample/--batch/--sort-by-size`; isolates kernel overhead (H2D + pad excluded) |
| `docs/spikes/batched-small-grid-sweep.md` | new spike, Resolved YES | MPS 2.46–4.05× faster batched; GPU-specific; padding-waste = option-B lever |
| `docs/spikes/slot-scale-parallelism.md` | status → hypothesis resolved | load-bearing batching claim confirmed; links new spike |

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

### 3. ✅ DONE — Batched small-grid sweep kernel

`sweep_sssp_3d_batched` + `scripts/batched_sweep_prototype.py`. Resolved
YES: K independent sub-grids in one kernel call is **2.46–4.05× faster than
sequential on MPS** (GPU-specific; CPU loses). Padding waste is the
option-B lever (sorting nearly doubles the win). Folded into new spike
[`batched-small-grid-sweep.md`](../spikes/batched-small-grid-sweep.md);
slot-scale spike's load-bearing hypothesis resolved. **Open levers:
convergence-masking + option-B size bucketing toward the bandwidth floor.**

### 4. Update WS3.3 plan (small) — **next up**

`docs/plans/ws33-tile-router-implementation.md` is still the obsolete
8-slice fixed-tile plan. Rewrite around: track-pitch grid → per-net
guide-bbox sweep (option A) → **batched small-grid kernel (now validated,
2.46–4.05× on MPS)** → coarsened fallback for the ~6% over-cap tail.

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
- Follow-up 3 (batched kernel) → ✅ resolved into
  `docs/spikes/batched-small-grid-sweep.md` + slot-scale spike status
  update. The design (padded stack of K independent grids, mirroring
  `_multi`) was obvious enough not to need its own ADR; the spike captures
  the finding and the option-A/B decision lives in ADR 0012 Amendment 3.
- Follow-up 4 (plan rewrite) → updated
  `docs/plans/ws33-tile-router-implementation.md`.
- Then `git rm docs/handoffs/ws33-tile-decomposition-handoff.md` in the
  migration commit: `docs: resolve WS3.3 handoff — fold into plan +
  track-pitch prototype`.
