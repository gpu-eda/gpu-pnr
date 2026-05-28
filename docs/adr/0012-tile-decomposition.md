# ADR 0012 — Tile decomposition for chip-scale 3D routing

**Status:** Accepted (2026-05-14). Substrate validated by tile
prototype on the same day — see "Prototype findings" below.

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

## Prototype findings (2026-05-14)

`scripts/tile_decomp_prototype.py` (commit `26f0251`) routed the
densest 256² × 5 tile in Hazard3 (rows [7936:8192) cols
[2560:2816), 27 candidate nets, halo=32) on the tile-shared cost
grid via sequential `route_multipin_nets_3d`. Headline results:

- **Substrate correctness: 0 cross-net cell conflicts** on the
  routes that ran. Tile-shared `w_cur` propagates obstacles across
  nets exactly as designed.
- **Routability: 6 / 27 (22%).** All 21 failures are
  pin-collision — two distinct nets quantize to the same
  `(layer, row, col)` cell because this prototype uses guide-rect
  centers as pin coords on Hazard3's 200nm cell pitch.
  `route_fail=0` confirms every net that could *legally* route did
  route. This validates the substrate; pin-collision avoidance is
  upstream (DEF-driven pin extraction), not a router concern.
- **Halo occupancy: 0%.** All committed cells fell inside the inner
  256² owned region; none reached the halo ring. Insufficient
  data to conclude halo=32 is over-provisioned — this workload
  was M1+M2-confined (670/694 cells on M2), so halo's natural
  use case (M3+ long-haul detours) wasn't exercised here.
- **Throughput: 54 ms/net.** Comparable to the per-net mini-grid
  spike's 41–50 ms/net at 2-pin, with the added benefit of
  cross-net obstacle awareness. The K=100 batched regime
  (`sweep_sssp_3d_multi`) would push this toward 31 ms/source per
  Tier A — to be measured in the full tile router.

Net consequence for the design: substrate locked, halo width
deferred (refine to 16 or 64 once a workload that actually uses
M3+ via-stacks lands).

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

## Amendment 1 (2026-05-28): guide-constrained sweep replaces fixed-tile K-batching

Three findings since acceptance invalidate the original throughput model
and motivate an architectural pivot:

### Finding 1: K-batching is dead

The [Tier B spike](../spikes/tier-b-envelope-throughput.md) (resolved
2026-05-19) measured `sweep_sssp_3d_multi` at envelope sizes > 256² and
re-measured 256². Results:

- K-batching only wins at envelope=256², and only 1.46× (was 4.05× in
  Tier A — the erosion is environmental/MPS-firmware, not code).
- At all larger envelopes (320²+), sequential routing is faster than
  K-batched multi.
- Bisect-by-worktree at `e5dd5be` (exact Tier A commit) confirms: Tier
  A's 4.05× does not reproduce on current MPS. CI golden bench on M2
  Mac Mini also shows 0 K-batching benefit.

**The §2 "K-batch size: up to 100" and the §Consequences "31 ms/net
unit of work" are obsolete.** Sequential per-source is the design
parameter: 18 ms/net on M4 Pro MPS, 31 ms/net on M2 MPS (CI golden),
78–432 ms/net on CPU (size-dependent).

### Finding 2: 42% multi-tile-spanning at halo=32

[Slice 2 measurement](../results.md) on Hazard3 at halo=32 found 42.1%
of routable nets span multiple tiles — well past the §Walk-back 25%
gate. The §5 coarsened-pass would handle nearly half the nets, not the
estimated 5–15%.

### Finding 3: 65–328× search space gap vs TritonRoute

The [GPU vs DRT throughput spike](../spikes/gpu-vs-drt-throughput.md)
(2026-05-28) compared our per-net sweep cost against TritonRoute's
detailed router on the same dualcore design:

- DRT: ~1 ms/net (14 threads), ~11 ms/net single-threaded — routing
  within guide-constrained A\* on ~1,000–5,000 grid cells per net.
- GPU sweep: 18 ms/net MPS on full 256² × 5 grid (327,680 cells).
- The gap is dominated by **search space**, not compute speed. We
  sweep 65–328× more cells per net than DRT.

GPU MPS is 7–13× faster than CPU on the same grid (confirmed on both
M2 CI and M4 Pro local). The acceleration is real — but the fixed-tile
approach wastes it on cells no net needs to visit.

### Revised decision

Replace the fixed 256²-tile + K-batching model with
**guide-constrained adaptive sweep**:

1. **Ingest global-routing guides** from OpenROAD GRT (or our own
   future GR). Each net gets a set of GCell bounding boxes defining
   its routable region.

2. **Per-net sub-grid sweep.** For each net, compute the union bbox
   of its guides + a margin (e.g., 1–2 GCells). Sweep only that
   sub-grid, not a full 256² tile. A typical net with 2–3 guides on
   15×15 GCells sweeps ~3,000 cells instead of 327,680.

3. **Batched small-grid sweep.** Many independent per-net sub-grids
   can be batched into a single GPU kernel call. Unlike K-batching on
   a single large grid (which is dead per Tier B), batching many
   *small independent grids* should parallelise well — each is
   independent, and small grids fit in GPU cache. This is the new
   GPU parallelism model.

4. **Keep the chip-scale cost grid.** The shared `w_cur` obstacle
   encoding (validated in the tile prototype) remains — it's how
   cross-net conflicts propagate. The sub-grid sweep indexes into
   the chip-scale tensor; it doesn't copy it.

5. **Drop K-batching (§2).** The `sweep_sssp_3d_multi` machinery is
   no longer the performance-critical path. Sequential per-source
   sweep on adaptive sub-grids is both faster (smaller grid) and
   simpler (no multi-source distance tensor).

6. **Drop fixed tile assignment (§6) and halo reconciliation (§4).**
   Guide-constrained sweep doesn't need owned/halo regions — each
   net's search space is defined by its guides, not by a tile grid.
   Adjacent nets don't conflict via tile boundaries; they conflict
   via the shared `w_cur` tensor, handled by routing order and
   rip-up.

7. **Retain the coarsened-pass concept (§5) as a fallback** for nets
   with no guides or with guides spanning too large a region to
   sweep efficiently (>512² equivalent). These route on a coarsened
   grid first, then refine.

### Expected throughput

A net with a 50×30×2 guide region (~3,000 cells): at our measured MPS
throughput scaling, ~0.16 ms/net — ahead of DRT's 11 ms/net
single-threaded. For Hazard3's ~20k nets at this per-net cost:
~3.2 seconds total GPU sweep time, vs DRT's ~35s initial pass (14
threads). Even with overhead for backtrace, conflict detection, and
rip-up iterations, this is in the right ballpark to be competitive.

On CI (M2 Mac Mini) the per-cell sweep cost is ~3× higher, so expect
~0.5 ms/net for a 3,000-cell sub-grid — still well ahead of DRT
single-threaded.

### What survives from the original decision

- §1 tile size 256² as maximum sub-grid cap (not the default)
- §7 module structure (`tile_router.py`, API compatibility)
- The chip-scale cost grid substrate (prototype-validated)
- Walk-back options framework (adapted to guide-constrained context)

### What is superseded

- §2 K-batch size K=100 — dropped
- §3 halo width 32 — replaced by guide margin
- §4 halo reconciliation — not needed
- §5 multi-tile-spanning nets as primary concern — subsumed by
  per-net guide regions
- §6 per-tile net assignment — replaced by per-net guide assignment
- Throughput model: "31 ms/net K=100 batched" → "0.16–0.5 ms/net
  guide-constrained sequential"

### New risks

- **Guide ingestion.** Reading OpenROAD's guide format and mapping to
  our grid coordinates is new work. If guides are unavailable, fall
  back to bbox-based sub-grids (net pins bbox + margin).
- **Batched small-grid sweep kernel.** The current `sweep_sssp_3d`
  operates on a single contiguous tensor. Batching many variable-size
  sub-grids requires either padding to a common size or a new kernel
  that indexes into the chip-scale tensor with per-net offsets.
  Design TBD.
- **Quality without guides.** If we route without GRT guides (e.g.,
  using pin-bbox + margin), the search space is larger and quality
  may degrade for long-haul nets. Guides are load-bearing for both
  throughput and quality.

## Amendment 2 (2026-05-28): guide-region mapper lands; bbox throughput model refuted

Amendment 1 named a guide ingestion path and an expected throughput
(~3,000 cells, ~0.16 ms/net per net). Building the mapper and measuring
the real distribution settles two things: the **mapper conventions are
locked** (decisions below), and the **bbox throughput model is wrong**
(finding below).

### Decisions locked: the `guide_region` mapper

`src/gpu_pnr/guides.py` implements the Amendment 1 §1 ingestion path.
`guide_region(rects, chip_origin, layer_order, pitch_dbu, *, margin,
chip_shape)` maps a net's GRT guide rectangles to a `GuideRegion`
sub-grid bounding box. Conventions, now fixed:

1. **Half-open bounds.** A region is exactly `w_chip[l0:l1, r0:r1,
   c0:c1]` — same slicing convention as `tile_router` and
   `build_chip_grid`, so a region lines up cell-for-cell with the
   chip-scale cost tensor. The half-open upper bound is "last cell any
   guide rect touches, plus one".
2. **Contiguous layer span.** The layer range is `[min_guide_layer,
   max_guide_layer]` inclusive — a guide on M1+M3 yields layers
   `[0:3]`, **including the empty M2**. Via transitions relax through
   adjacent layers ([ADR 0006](0006-sequential-via-relax.md)); a
   sub-grid that dropped the empty middle layer would sever the via
   stack. Margin never widens the layer range.
3. **Margin in grid cells, default 4.** Slack added on every side of
   the row/col bbox (≈0.8 µm at 200 DBU pitch) for pin reach and short
   detours just outside the guides — mirrors DRT's "small expansion
   margin". Deliberately **not** in GCells: one GCell is 84 grid cells,
   which would balloon a compact region.
4. **`chip_shape` clamps** the returned region to `[0,L)×[0,H)×[0,W)`
   so it is always a valid slice; `None` when no rect lands on a layer
   in `layer_order`.

The mapper is pure integer geometry — no torch, no PDK coupling beyond
the caller-supplied `layer_order`. The raw `*.guide` text parser stays
in `scripts/_hazard3_io.parse_guides` (fixture I/O).

### Finding: guide-bbox sub-grids are 31× larger than estimated

`scripts/measure_guide_regions.py` over the full Hazard3 fixture
([`../results.md`](../results.md) Phase 3.3):

- **Median sub-grid ~93k cells (176×176×3), not 3,000** — 31× the
  Amendment 1 estimate. Mean 784k.
- Guide-bbox sweep shaves only **~3.5× vs the full 256²×5 tile**, not
  the ~100× Amendment 1 assumed, and stays **19–93× above DRT's
  1,000–5,000 cells/net**.
- **47.6% of nets exceed the 256² per-axis cap** — nearly half would
  fall to the coarsened-pass fallback, not a direct sub-grid sweep.
- Linear (optimistic) throughput: median ~5 ms/net, total ~884 s on
  M4 Pro — vs the Amendment's 0.16 ms/net and 3.2 s.

**Root cause: a guide set is a thin snake; its bounding box is the
enclosing rectangle.** The dense-tensor sweep pays for every cell in the
bbox, including the empty area the snake routes around. DRT searches the
snake (guide cells only), not the rectangle. So Amendment 1 §2's
"sweep only that sub-grid" reduces work far less than assumed, because
"that sub-grid" is the bbox, not the guides.

### Open architectural question (NOT decided here)

What search space should the sweep actually visit? The bbox is a poor
proxy; the realistic options, in increasing cost:

- **(A) Accept the bbox.** Sweep the ~3.5×-reduced bbox and bank the win
  on GPU batching of many independent sub-grids (Amendment 1 §3).
  Simplest; but ~48% over-cap nets need the fallback, and batching is
  dominated by large variable-size grids.
- **(B) Per-segment (snake-following) sweep.** Decompose a multi-pin /
  L-bending net into guide-segment-sized boxes (à la DRT's two-pin
  Steiner segments), each ~1–2 GCells, and sweep each small box. Keeps
  the dense kernel; shrinks the search space to near the snake. Adds a
  decomposition + stitch step.
- **(C) Sparse guide-cell graph.** Abandon the dense rectangular tensor
  for a packed graph over guide cells only (à la DRT's `FlexGridGraph`).
  Closes the search-space gap fully but is a major kernel rewrite and
  loses the dense-sweep `cumsum`/`cummin` formulation
  ([ADR 0002](0002-scan-based-sweeps.md)).

This fork gates the sweep prototype (Amendment 1 follow-up 2) and needs a
decision before that work starts. It likely warrants its own spike to
measure option (B)'s decomposition overhead against (A)'s batching win.

### What survives

- Amendment 1's pivot away from fixed-tile K-batching stands — that was
  driven by Tier B (K-batching dead) and the DRT search-space gap, both
  independent of this finding.
- The `guide_region` mapper is the ingestion primitive for all three
  options above; none of them discard it.
- The chip-scale shared `w_cur` cost grid (Amendment 1 §4) is unchanged.

## Amendment 3 (2026-05-28): the cost grid is over-sampled 5.6× — route on the track pitch

Amendment 2 found guide-bbox sub-grids 31× larger than Amendment 1's
estimate and framed an A/B/C fork over what the sweep should visit.
Checking the grid pitch against the PDK shows the 31× is almost entirely
an **over-sampling artifact**, and the fork largely dissolves.

### Finding: our grid pitch is 5.6× finer than the routing tracks

From the Hazard3 DEF (gf180mcuD):

| Quantity | Value |
|---|---|
| DEF units | 2000 DBU/µm |
| Routing track pitch, M1–M4 (`TRACKS ... STEP`) | **1120 DBU = 0.56 µm** |
| Routing track pitch, M5 | 1800 DBU = 0.9 µm |
| GCell (`GCELLGRID ... STEP`) | 16,800 DBU = **15 tracks** |
| Our cost-tensor pitch (`_hazard3_io.PITCH_DBU`) | **200 DBU = 0.1 µm** |

The cost tensor samples at 200 DBU but wires only sit on 1120 DBU tracks
— **5.6× finer per axis ≈ 31× more cells than there are track
intersections**, for the same silicon. The 200 DBU value was an early
arbitrary quantization; no ADR or comment justifies it. (Re-measuring at
`--pitch 1120` validates this directly — see below.)

### Re-measurement at track pitch validates Amendment 1

`scripts/measure_guide_regions.py --pitch 1120` over the full Hazard3
fixture ([`../results.md`](../results.md) Phase 3.3):

| Metric (median net) | 200 DBU (Amendment 2) | 1120 DBU (track) | Amendment 1 estimate |
|---|---:|---:|---:|
| Sub-grid cells | 92,928 | **4,332** | ~3,000 |
| ms/net (M4 Pro, linear) | 5.10 | **0.24** | 0.16 |
| Total, 20.5k nets | 884 s | **31 s** | 3.2 s |
| Nets over 256² cap | 47.6% | **6.3%** | "5–15%" |

At track resolution the median net is 1.4× the Amendment 1 estimate (not
31×), and the over-cap fraction lands inside the "5–15%" Amendment 1
anticipated. **Amendment 1's throughput model was right; it was implicitly
reasoning in tracks (15 per GCell), and the tensor was over-sampled.**

### Decision direction: adopt the track-pitch grid

Route on a grid sampled at the routing-track pitch, not 200 DBU. This is
**higher-leverage and lower-risk than the A/B/C fork** and reshapes it:

- The pitch fix is a `build_chip_grid` / `Pdk.pitch_dbu` change — **no
  routing-algorithm change** — and it shrinks every net ~31×.
- At track pitch, **option A (plain guide-bbox sweep) is viable for ~94%
  of nets** (median 0.24 ms/net, ~31 s total single-stream on Hazard3).
- **Options B (corridor decomposition) and C (sparse guide-cell graph)
  become a deferred tail-optimization** for the ~6% over-cap / bendy
  nets — not a prerequisite. Defer until a track-pitch sweep prototype
  measures whether the tail actually hurts.

### Open implementation questions (not yet decided)

The pitch change is not free; these gate a clean track-pitch router:

1. **Pin access.** Access points need not lie on tracks. A pure track
   grid can miss them. DRT's answer is *off-track* grid lines near pins;
   ours will need pin snapping or a locally-finer region around pins.
   Risk: a coarser grid that mis-snaps pins re-introduces the
   pin-collision failures the tile prototype already saw (§Prototype
   findings: 21/27 failures were pin quantization on the 200 DBU grid).
2. **Per-layer / non-uniform pitch.** M1–M4 are 1120 DBU; M5 is 1800;
   the track origin offset is 560. A uniform tensor wants one pitch.
   Either pick the finest (1120, over-sampling M5 ~1.6×) or carry a
   non-uniform grid (more work, affects the `cumsum`/`cummin` axis
   assumptions of [ADR 0002](0002-scan-based-sweeps.md)).
3. **Via alignment.** On a track grid vias land on track intersections
   naturally; on 200 DBU they are mis-quantized today. Re-pitching
   should *help* via correctness, but the via-relax kernel
   ([ADR 0006](0006-sequential-via-relax.md)) needs re-checking at the
   new pitch.

### Slot-scale implication

At track pitch the approach plausibly scales to a wafer.space 1×1 slot
(~12.9 mm², ~350k routable nets, ~9.6 B grid cells). The parallel work
vastly exceeds what saturates an M4 Pro GPU; the limiter is per-net
kernel-launch overhead, which the batched small-grid sweep (Amendment 1
§3) targets. Full analysis in
[`../spikes/slot-scale-parallelism.md`](../spikes/slot-scale-parallelism.md).

### What survives / is superseded

- **Survives:** the `guide_region` mapper (Amendment 2) — it takes
  `pitch_dbu` as a parameter, so it already works at any pitch. The
  guide-constrained pivot (Amendment 1). The chip-scale `w_cur` grid.
- **Superseded:** Amendment 2's framing of A/B/C as a near-term fork —
  it is now "pitch fix first, then A; B/C deferred". The implicit
  assumption throughout ADR 0012 that the cost tensor is sampled at
  200 DBU.

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
- [`../spikes/tier-b-envelope-throughput.md`](../spikes/tier-b-envelope-throughput.md)
  — Tier B spike: K-batching dead past 256², sequential is the
  design parameter (Amendment 1).
- [`../spikes/gpu-vs-drt-throughput.md`](../spikes/gpu-vs-drt-throughput.md)
  — GPU vs DRT comparison: 65-328× search space gap motivates
  guide-constrained sweep (Amendment 1).
