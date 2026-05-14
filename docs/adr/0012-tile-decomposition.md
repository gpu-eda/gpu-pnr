# ADR 0012 — Tile decomposition for chip-scale 3D routing

**Status:** Proposed (2026-05-14).

## Context

WS3.3 replaces per-net mini-grids ([ADR 0009](0009-per-net-grids.md))
with a chip-scale routing substrate. The naive single-grid approach
hits three walls:

1. **Float32 precision wall** ([ADR 0005](0005-mask-based-segmented-scan.md)):
   the segmented-scan SEG_BARRIER product `max_seg_id * SEG_BARRIER`
   blows past float32 ULP budget around 5000² for dense obstacle
   patterns. Hazard3's full-chip die at gf180mcuD pitch is ~5000² per
   layer × 5 layers — right at the edge.
2. **Per-iter cost on a single grid:** the chip-scale prototype
   (commit `7acfd69`) measured 327 s per 2-pin net on a Hazard3-sized
   grid. That extrapolates to days for 24K nets, untenable.
3. **No batching headroom:** `sweep_sssp_3d_multi` collapses at
   512²+ (see [Tier A spike](../spikes/tier-a-sweep-sharing-throughput.md));
   chip-scale grids leave no room to use the K-source primitive that
   makes routing fast on Apple Silicon MPS.

The Tier A sweep-sharing spike (commit `e5dd5be`, resolved YES at
256²) empirically established that:

- **256² × L=5 is the throughput sweet spot.** K=100 sources gives
  **4.05× speedup** (125 → 31 ms per source) in 3D.
- 512² regresses catastrophically at K=100 (0.19×) due to MPS
  memory pressure on the multi-source distance tensor.
- The K-knee in 3D is sharper than 2D (4× vs 9× peak) because the
  per-layer sequential via-relax ([ADR 0006](0006-sequential-via-relax.md))
  doesn't share between sources.

Plus a structural unlock the chip-scale prototype surfaced: ~6,000
small Hazard3 nets are guide-locked to M1+M2 with no M3 in their GR
allocation, and our per-net mini-grid router can't escape because
off-guide cells are `inf`. A single global cost tensor makes GR
allocation advisory by construction — closes the small-net cohort
gap that `m1_penalty` / pin-only couldn't reach
([`../results.md`](../results.md) Phase 3.2 investigation, Finding 4).

## Decision

WS3.3 routes nets on a chip partitioned into **256² × L=5
overlapping tiles**, with the K-source kernel
(`sweep_sssp_3d_multi`) routing **up to K=100 nets per tile in one
sweep**. The detailed design follows.

### 1. Tile size: 256² × L=5 (locked)

- Per Tier A spike. Not negotiable on MPS without new measurement.
- 256 cells × 0.20 µm pitch ≈ 51 µm — about 4–6 standard-cell rows
  on Hazard3 at gf180mcuD. Most 2-pin nets are well within one tile.
- Memory per tile cost tensor: 256² × 5 × float32 = 1.3 MB per axis
  (need both `w_h` and `w_v`) = 2.6 MB. Multi-source distance tensor
  at K=100: 100 × 256² × 5 × float32 = 131 MB. Well inside MPS budget.

### 2. K-batch size: up to 100 sources per multi-source call (locked)

- Tier A peak is K=100. K=25-100 is the productive range; below K=10
  the GPU is launch-bound.
- Per-tile batching: collect up to 100 ready-to-route nets per tile,
  call `sweep_sssp_3d_multi`, backtrace each.

### 3. Halo width: 32 cells (initial — revisable from prototype data)

- Each tile owns a 256² region but routes within a `256 + 2 × halo`
  envelope (= 320² with 32-cell halo). Cells inside the halo are
  routable; the halo lets routes detour briefly into the adjacent
  tile's region without needing cross-tile coordination per-step.
- Memory overhead: 320² × 5 × float32 = 2 MB per cost-tensor axis,
  ~50% overhead vs the 256²-strict region. Multi-source distance at
  K=100: 100 × 320² × 5 × float32 = 205 MB. Still inside MPS budget.
- 32 cells = ~6.4 µm on gf180mcuD, roughly 60% headroom over the
  estimated longest in-tile detour bbox from Hazard3 (≤20 cells in
  the handoff's per-spike estimate). Prototype-measurable; revise
  ADR if data says otherwise.

### 4. Halo reconciliation: iterative re-sweep within halo (initial)

When two adjacent tiles both route nets that land in their shared
halo region, both tiles' routes may conflict on halo cells.
Approach: after both tiles commit their routes, **re-sweep only the
halo cells** with both tiles' committed routes visible as
obstacles. Re-attach any rip-up'd net via a second multi-source
call on the halo region.

- Cheaper than a global second pass on a coarsened grid for the
  common case (most halo cells aren't disputed).
- Local — only adjacent-tile pairs need to coordinate.
- Bounded — halo region is `2 × halo × 256` cells per adjacent
  pair = 16K cells for halo=32, tiny relative to tile area.
- See "Walk-back options" for the global-second-pass fallback.

### 5. Multi-tile-spanning nets: global coarsened pass (initial)

Nets whose bbox+halo crosses multiple tile boundaries (estimated
5-15% of nets per the handoff's analysis) route on a **coarsened
chip-scale grid** in a separate first pass.

- Coarsening factor: 4× downsampling. For Hazard3 (~5000² × 5),
  this is ~1250² × 5 — comfortably inside the float32 precision
  budget (per ADR 0005) and inside MPS memory.
- The coarsened pass produces a route at coarse resolution. Per-tile
  passes then commit those routes to the fine grid as obstacles
  before routing the rest.
- Coarsening loses M1 pin-access fidelity but fine for long-haul
  routing on M3-M5 where most multi-tile-spanning nets travel.

### 6. Per-tile net assignment: bbox-fits-in-tile (including halo)

- A net's `(bbox + halo)` fits in exactly one tile → assigned to that
  tile.
- Otherwise → multi-tile-spanning, goes through the coarsened pass.
- Tiebreak (a net's bbox+halo fits multiple tiles, e.g., the bbox is
  smaller than a tile and lies at the boundary of two): the tile
  whose **owned region** (the inner 256², not the halo) contains
  the net's bbox center.

### 7. Module structure

- New module `src/gpu_pnr/tile_router.py` containing a
  `TileRouter` class with the per-tile lifecycle.
- API-compatible with `route_multipin_nets_3d`: takes
  `nets: list[list[(l, r, c)]]`, returns
  `list[MultiPin3DResult]` in the same order.
- Reuses pin-reservation logic from `route_multipin_nets_3d`
  (it already does what tile-local routing needs).
- Implementation gated on a tile prototype first
  (`scripts/tile_decomp_prototype.py`) that measures halo cost
  empirically on one Hazard3 sub-region before committing to the
  full chip-scale tile manager.

## Consequences

**What this buys:**

- Chip-scale integration (24K Hazard3 nets in one routing pass with
  cross-net conflict detection) without hitting the float32
  precision wall on a single global grid.
- Per-tile K=100 batching turns each tile into a ~31 ms/net unit of
  work (Tier A measurement). Extrapolated to Hazard3-scale: ~22-44
  minutes total wall-clock on MPS, vs the per-net mini-grid
  spike's ~35 minutes and the chip-scale single-grid prototype's
  days. CUDA should push toward 2-5 min via its 7-15× memory
  bandwidth advantage (see [`../results.md`](../results.md) Phase
  3.2 sweep-sharing extrapolation).
- Subsumes the "guide as soft preference" hack ([plan
  Finding 4](../plans/phase3-detailed-routing.md)) — global cost
  tensor makes GR allocation advisory by construction. Closes the
  ~6,000 small-net cohort gap the per-net mini-grid spike couldn't
  reach.
- Unlocks `route_nets_batched` ([ADR 0008](0008-defer-route-nets-batched.md))
  per-tile — the deferred conflict-detection / ripup loop now lives
  on a small enough tile that the K-source kernel actually wins.

**What this costs:**

- The most complex routing primitive yet. Per-tile lifecycle, halo
  reconciliation, multi-tile-spanning fallback, coarsened-pass
  glue. Implementation estimate: 1-2 days for a working chip-scale
  pass, plus the tile prototype.
- Halo overhead: ~50% memory per tile vs strict 256². Total memory
  budget is still well inside MPS, but the constant matters for
  CUDA tile sizing later.
- The coarsened-pass for multi-tile-spanning nets introduces a
  routing primitive that doesn't match the per-tile kernel. Either
  the coarsened pass is a small fraction of work or its
  implementation merits its own ADR if it grows.
- Halo reconciliation is the highest-risk piece. If non-local
  detours dominate, the local re-sweep won't converge and we walk
  back to the global second pass.

**Operational impact:**

- New test surface area: tile boundary correctness, halo
  reconciliation correctness, coarsened-pass route reusability.
- The chip-scale-prototype script (commit `7acfd69`) becomes the
  worst-case baseline for TR comparison. WS3.3's chip-scale router
  needs to beat that prototype's 327 s/net by a wide margin to be
  worth shipping.

## Walk-back options

- **If halo reconciliation cost dominates** (the local re-sweep
  doesn't converge or eats a large fraction of total time) — switch
  reconciliation strategy from local re-sweep to a **global second
  pass on a coarsened grid**, similar to the multi-tile-spanning
  net pass. Reuse the coarsened-grid infrastructure.

- **If halo width 32 is too small** (routes hit the halo boundary
  often) — widen to 64. Memory cost: 384² per tile vs 320², ~44%
  more. Still inside MPS budget.

- **If the multi-tile-spanning net fraction is >25% on Hazard3** —
  the coarsened pass is doing too much work. Reconsider per-tile
  net assignment: split a too-big net across the tiles it touches
  with halo handshake. Adds complexity but avoids relying on a
  coarsened-grid pass for the common case.

- **If `sweep_sssp_3d_multi` regresses on tiles with high obstacle
  density** (the autotune SEG_BARRIER's lower bound creeps past
  the upper bound — see ADR 0005's float32-precision-budget walk-back)
  — drop tile size from 256² to 192² and re-measure. K-knee may
  shift; re-bench before committing.

- **If WS3.3 reveals a topology lever** (e.g., Steiner attachment
  on the chip-scale grid closes the 0.55× via-ratio gap from the
  multi-pin spike) — that's a separate workstream, doesn't
  invalidate this ADR. Decisions here are about *substrate*, not
  attachment heuristic.

## Links

- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md)
  — WS3.3 in the Phase 3 plan; this ADR captures the design.
- [`../spikes/tier-a-sweep-sharing-throughput.md`](../spikes/tier-a-sweep-sharing-throughput.md)
  — Tier A spike: empirical foundation for tile size and K-batch.
- [ADR 0005](0005-mask-based-segmented-scan.md) — float32 precision
  wall that motivates tile decomposition.
- [ADR 0006](0006-sequential-via-relax.md) — 3D via relax kernel
  (per-pair via_cost amendment 2026-05-14).
- [ADR 0008](0008-defer-route-nets-batched.md) — sweep-sharing
  deferred until tile decomposition; this ADR unlocks it.
- [ADR 0009](0009-per-net-grids.md) — per-net mini-grids that this
  WS3.3 supersedes.
- [`../results.md`](../results.md) — chip-scale prototype baseline
  (327 s/net) that WS3.3 must improve on.
