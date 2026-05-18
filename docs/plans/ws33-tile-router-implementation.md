# Plan — WS3.3 `TileRouter` implementation

**Status:** Proposed (2026-05-15).

<!--
Status lifecycle:
  Proposed → Active → Closed (YYYY-MM-DD)
Companion to docs/plans/phase3-detailed-routing.md §WS3.3 (which keeps
the high-level design summary). This plan covers the build slicing only.
-->

## Goal

Build `src/gpu_pnr/tile_router.py` — the chip-scale tile manager that
implements [ADR 0012](../adr/0012-tile-decomposition.md) end-to-end and
satisfies the WS3.3 exit criteria in
[phase3-detailed-routing.md §WS3.3](phase3-detailed-routing.md#ws33--tile-decomposition).
The substrate is already validated (commit `26f0251`, prototype findings
folded into ADR 0012). This plan is about composing that substrate into
a full router across many tiles plus the coarsened multi-tile-spanning
pass plus halo reconciliation.

## Prerequisites

- [ADR 0012](../adr/0012-tile-decomposition.md) Accepted.
- [ADR 0008](../adr/0008-defer-route-nets-batched.md) — per-tile
  conflict-detect/ripup work unlocked by this plan.
- [ADR 0007](../adr/0007-hpwl-ascending-net-ordering.md) — ordering used
  at tile and intra-tile granularity.
- `sweep_sssp_3d_multi` (sweep.py) and `route_multipin_nets_3d`
  (router.py) shipped.
- Hazard3 fixture at `~/Code/Apitronix/hazard-test` (per project memory)
  and `scripts/_hazard3_io.py` parser available.
- 69 passing tests on `main` (verification command in the handoff).

## Scope-out (not in this plan)

- Open follow-up §2 kernel optimization (`torch.compile`, eliminating
  `.item()` syncs). Premature until the full router exists; profile data
  already captured in commit `7acfd69`.
- Open follow-up §3 DEF-driven pin extraction. Orthogonal tooling; can
  land independently. The 22% prototype routability is a pin-extraction
  artifact, not a router concern.
- CUDA port. MPS is the host per [ADR 0001](../adr/0001-pytorch-mps-host.md).
- Topology lever / Steiner attachment for the 0.55× via-ratio gap. Listed
  in ADR 0012 walk-back as a separate workstream.
- Per-pair `via_cost` retuning. WS3.2 shipped the API; chip-scale data
  will inform follow-up.

## Design source of truth

[ADR 0012](../adr/0012-tile-decomposition.md). This plan never restates
design choices; it cites the ADR. Walk-back triggers below map to the
ADR's "Walk-back options" §.

## Slicing strategy

Eight slices, ordered so the highest-risk measurements (multi-tile-
spanning fraction, halo reconciliation convergence) land in cheap early
slices instead of inside the terminal integration. Each slice is a
single PR-sized commit (≤500 LOC net new), tests pass on `main`, and is
independently reviewable.

The Hazard3 fixture is large; CI-level slices route synthetic small
grids only. Hazard3 measurement runs land in `scripts/` and produce
`docs/results.md` entries via slice-level scripts, not pytest.

---

### Slice 1 — Tile geometry + partition module

**Deliverable:** `src/gpu_pnr/tile_router.py` skeleton with pure data
classes and the assignment algorithm — no routing yet.

**Implementation sketch:**

- `@dataclass Tile`: owned-region `(r0, c0, r1, c1)`, halo, routable
  envelope.
- `def partition_chip(chip_h, chip_w, tile_size=256, halo=32) -> list[Tile]`
  generating the non-overlapping owned 256² grid covering the chip
  (right/bottom edge tiles may be partial); halo is implicit per ADR 0012 §3.
- `def net_bbox(pins) -> (rmin, cmin, rmax, cmax)` over 3D pin cells (l
  ignored).
- `def assign_net_to_tile(bbox, tiles, halo) -> Tile | None` returning the
  unique tile whose owned+halo region contains bbox, with the tiebreak
  rule from ADR 0012 §6 (owned region containing bbox center). Returns
  `None` for multi-tile-spanning nets.
- `def classify_nets(nets, tiles, halo) -> tuple[dict[Tile, list[int]], list[int]]`
  returning per-tile net indices and the multi-tile-spanning index list.
- Public surface defined but `TileRouter.route` raises `NotImplementedError`.

**Tests added:** `tests/test_tile_router.py` (new file, follow
`tests/test_router_3d.py` style):

- `test_partition_covers_chip` — every owned cell is owned by exactly
  one tile; partition is a tiling.
- `test_assign_net_bbox_fits_owned` — net in one tile's owned region is
  assigned to that tile.
- `test_assign_net_bbox_in_halo_only` — net whose bbox center is in tile
  A's owned region but bbox+halo straddles tile B's owned region is
  assigned to A (tiebreak).
- `test_assign_net_multi_tile_spanning` — net whose bbox+halo crosses
  multiple owned regions returns `None`.
- `test_classify_partitions_all_nets` — every input net is in exactly
  one of {per-tile lists, multi-tile-spanning list}.

**Exit criterion:** new test file passes (5 tests); `uv run pytest
tests/` shows 74 passed.

**Risk/walk-back:** none.

---

### Slice 2 — Hazard3 partition measurement: multi-tile-spanning fraction

**Deliverable:** `scripts/measure_tile_partition.py` — load Hazard3
nets, run `classify_nets`, report the fraction of nets that are
multi-tile-spanning at `halo ∈ {16, 32, 64}`.

**Implementation sketch:**

- Reuse `parse_guides`, `parse_def_diearea`, `_net_chip_pins` from
  `scripts/_hazard3_io.py` and `scripts/tile_decomp_prototype.py`.
- For each halo value: classify; print {per-tile count distribution,
  multi-tile-spanning count, fraction}.
- No routing; partition + counting only. Should run in seconds.

**Tests added:** none (script-only measurement).

**Exit criterion:** multi-tile-spanning fraction recorded in
`docs/results.md` §Phase 3.3, for halo=32 across the full Hazard3 fixture.

**Measurements logged:** multi-tile-spanning fraction at halo
∈ {16, 32, 64}; per-tile net count histogram (min/median/p90/max).

**Risk/walk-back:** if multi-tile-spanning fraction at halo=32 is >25%,
trigger ADR 0012 walk-back: split a too-big net across adjacent tiles
with halo handshake instead of coarsened-pass. This decision lands as
an ADR 0012 amendment before Slice 6 starts. If 5-15%, proceed as
designed.

---

### Slice 3 — Per-tile slicing and net rebasing

**Deliverable:** `_slice_tile(w_chip, tile, halo) -> w_tile` and
`_rebase_pins(net, tile) -> local_pins` helpers; `TileRouter` can
extract one tile and route its assigned nets sequentially via
`route_multipin_nets_3d` (no K-batching yet, no halo reconciliation).

**Implementation sketch:**

- Lift the chip-to-tile slicing logic from
  `scripts/tile_decomp_prototype.py` (lines ~220-238) into
  `tile_router.py`. Out-of-chip cells become `inf` per the prototype.
- `_rebase_pins`: subtract `(tile_r0 - halo, tile_c0 - halo)`.
- HPWL-ascending intra-tile ordering ([ADR 0007](../adr/0007-hpwl-ascending-net-ordering.md));
  reuse `ordering.order_nets` semantics or compute inline for 3D nets.
- `TileRouter.route` (partial): iterate tiles in HPWL-ascending order
  (tile HPWL = sum of contained nets' HPWLs), call
  `route_multipin_nets_3d` on each tile's slice, commit results back
  into chip-global coordinates.
- Multi-tile-spanning nets: return `MultiPin3DResult(pins, None)` for
  now (no coarsened pass yet — wired up in Slice 6).

**Tests added:**

- `test_slice_tile_padding` — slicing a tile partially off-chip yields
  `inf` outside chip bounds, exact chip-cost values inside.
- `test_rebase_pins_roundtrip` — chip → local → chip is identity.
- `test_tilerouter_single_tile_synthetic` — 4×256² grid, all nets fit
  one tile, results identical to `route_multipin_nets_3d` on the chip
  grid directly.
- `test_tilerouter_two_tiles_no_spanning` — 2-tile synthetic chip
  (512×256), nets split across two tiles, no cross-tile routes, zero
  conflicts (`_check_conflicts` from prototype, lifted into the test
  helper).

**Exit criterion:** test file extended to ~9 tests passing; running
`scripts/tile_decomp_prototype.py 256 32 1.0` and Slice 3's `TileRouter`
on the same single-tile sub-region produces byte-identical
`MultiPin3DResult.cells`.

**Risk/walk-back:** none (deterministic slicing).

---

### Slice 4 — K=100 batching via `sweep_sssp_3d_multi`

**Deliverable:** Replace the sequential `route_multipin_nets_3d` call
inside each tile with a batched K=100 sweep + per-net backtrace.

**Implementation sketch:**

- For each tile, collect 2-pin nets (the bulk per the Hazard3 prototype's
  M1+M2-confined distribution); batch up to K=100 into one
  `sweep_sssp_3d_multi` call. Backtrace each independently against the
  tile-shared `w_cur`.
- N-pin nets (≥3 pins): fall back to sequential
  `route_multipin_nets_3d` within the tile for now. Incremental
  tree-growth needs per-net sweep dependencies that don't share well at
  K=100 (the prototype already proved this); revisit only if profile
  shows it dominates.
- Pin reservation: same logic as `route_multipin_nets_3d` (reserve all
  net pins as obstacles, temporarily restore the routing net's pins).
  Lift to a helper rather than duplicating.
- Per-net commit: mark routed cells `inf` in `w_cur` (and `w_v_cur`)
  before the next net's backtrace.

**Tests added:**

- `test_tilerouter_kbatch_matches_sequential` — synthetic tile with 20
  2-pin nets; K=20 batched route produces identical cell sets to
  sequential `route_multipin_nets_3d`.
- `test_tilerouter_mixed_pin_counts` — tile mixing 2-pin and 3-pin
  nets; 3-pin nets fall through to sequential and still route
  correctly.

**Exit criterion:** tests pass; running Slice 4 on the densest Hazard3
tile (same one as `scripts/tile_decomp_prototype.py`) hits
**≤50 ms/source** averaged across the batch (prototype baseline was
54 ms/net sequential; Tier A predicts 31 ms/source — anywhere in
between is acceptable as long as it improves on sequential).

**Measurements logged:** ms/source for K=100 batched vs sequential on
the densest tile (`docs/results.md` Phase 3.3).

**Risk/walk-back:** if K=100 regresses vs sequential on dense tiles
(autotune SEG_BARRIER lower bound creeps past upper bound — ADR 0005
walk-back), drop K to 50 and re-measure; if still regressing, drop tile
size to 192² (ADR 0012 walk-back §4) and re-bench before proceeding.

---

### Slice 5 — Per-tile conflict detect + ripup/reroute

**Deliverable:** Per-tile conflict-detect-and-ripup loop on top of the
K=100 batched sweep. This is the [ADR 0008](../adr/0008-defer-route-nets-batched.md)
unlock.

**Implementation sketch:**

- After backtracing all K nets, scan committed cells; nets sharing a
  cell collide.
- Ripup policy: keep the lowest-HPWL net (ADR 0007 logic); requeue the
  others to a per-tile retry batch.
- Re-route the retry batch via a second `sweep_sssp_3d_multi` call with
  the winners' cells already `inf`. Bound retries (e.g. ≤3 rounds);
  any net still unrouted after the cap is marked `MultiPin3DResult(pins, None)`.
- All of this lives **inside one tile** — no cross-tile dependency.

**Tests added:**

- `test_tilerouter_conflict_ripup_simple` — two synthetic nets sharing
  a single cell; lower-HPWL wins, the other reroutes around.
- `test_tilerouter_conflict_unroutable_after_ripup` — construct a tile
  where ripup is forced and reroute is impossible; result is
  `routed=False` for the loser, `True` for the winner.
- `test_tilerouter_no_conflict_means_no_extra_sweep` — when the K=100
  batch produces no conflicts, the retry batch is empty and no second
  sweep runs (assert via a sweep-call counter).

**Exit criterion:** tests pass; running Slice 5 on the densest Hazard3
tile still yields **0 cross-net conflicts** in the final committed set
(`_check_conflicts(results) == 0`); this matches the prototype's
substrate invariant.

**Measurements logged:** conflict count after the K=100 first pass
(pre-ripup) and after final retry, on the densest Hazard3 tile.

**Risk/walk-back:** if ripup retries don't converge within 3 rounds for
>1% of tiles on Hazard3, increase the cap to 5 and re-measure;
otherwise leave the unroutable nets marked failed (the caller can fall
back to per-net mini-grid for those, which is existing infrastructure).

---

### Slice 6 — Coarsened multi-tile-spanning pass

**Deliverable:** Global 4× coarsened-grid pre-pass for multi-tile-
spanning nets; their routes pinned as obstacles before per-tile passes
run.

**Implementation sketch:**

- `_coarsen_grid(w_chip, factor=4) -> w_coarse` — pool obstacles
  conservatively (any sub-cell `inf` → coarse cell `inf`; otherwise
  min/mean of finite values).
- `_coarsen_pins(pins, factor) -> coarse_pins` — integer division.
- Route multi-tile-spanning nets on the coarsened grid via
  `route_multipin_nets_3d` (sequential; this fraction is small — ≤25%
  by Slice 2's exit-gate, expected 5-15% — so K-batching here is a
  micro-opt, not load-bearing).
- `_refine_coarse_route(coarse_path, factor, w_chip) -> fine_cells` —
  expand each coarse cell to its 4×4 sub-region of fine cells; mark
  those as the multi-tile-spanning net's footprint.
- In `TileRouter.route`: run coarse pass first, mark refined footprints
  as `inf` in `w_chip` before per-tile slicing.
- Multi-tile-spanning nets get `MultiPin3DResult` populated from the
  refined cells; ordering in the output list is preserved.

**Tests added:**

- `test_coarsen_grid_pools_obstacles` — 8×8 fine grid with a single
  `inf` cell coarsens to 2×2 where exactly one coarse cell is `inf`.
- `test_coarsen_route_refine_roundtrip` — coarse path with valid
  adjacency, refined to fine cells, has a connected (post-coarsening)
  fine footprint.
- `test_tilerouter_spanning_net_pinned_in_tile` — synthetic 2-tile chip
  with one multi-tile-spanning net and one per-tile net; the per-tile
  net detours around the spanning net's refined footprint.

**Exit criterion:** tests pass; on Hazard3, the coarsened pass routes
≥80% of multi-tile-spanning nets (this is the conservative lower bound
— ADR 0012 §5 anticipates the coarsened pass loses M1 pin-access
fidelity but works for M3+ long-haul).

**Measurements logged:** multi-tile-spanning route success rate;
fraction of chip area occupied by refined spanning-net footprints (this
caps what per-tile passes can do).

**Risk/walk-back:** if coarsened-pass success rate <50%, walk back per
ADR 0012 walk-back §3 to bbox-splitting across tiles. Decision lands
as an ADR 0012 amendment.

---

### Slice 7 — Halo reconciliation

**Deliverable:** After per-tile passes complete, re-sweep adjacent tile
pairs' shared halo regions; ripup/reroute halo conflicts. ADR 0012 §4
strategy (a): iterative local re-sweep.

**Implementation sketch:**

- Detect halo conflicts: for each pair of adjacent tiles, find cells in
  the shared halo region that are committed by both tiles' nets.
- Re-sweep scope: just the halo region (2 × halo × tile_size cells per
  adjacent pair) with both tiles' committed routes visible as
  obstacles, ripped-up cells freed.
- Convergence: bounded iteration (≤3 rounds per tile pair); track
  whether any halo cell changed ownership in the round.
- HPWL tiebreak for ripup (ADR 0007).
- Walk-back hook: if the iteration doesn't converge for any tile pair,
  log it and (optionally) fall back to the global second pass on the
  coarsened grid for the unresolved nets. ADR 0012 walk-back §1.

**Tests added:**

- `test_halo_conflict_detected_across_pair` — synthetic 2-tile chip
  with hand-crafted nets routing into the shared halo; conflict cell
  identified.
- `test_halo_resweep_resolves_conflict` — same setup, resweep produces
  a final commit with 0 cross-net conflicts.
- `test_halo_resweep_failure_marks_net` — synthetic case where
  reconciliation is impossible (e.g., halo too narrow for any detour);
  loser net ends up `routed=False`, no crash.

**Exit criterion:** tests pass; on Hazard3, halo reconciliation
converges within 3 rounds for ≥99% of tile pairs; final committed cell
set has 0 cross-net conflicts.

**Measurements logged:** halo cell occupancy across all tiles
(per-layer histogram — drives ADR 0012 §3's halo-width refinement
question); halo conflict count pre-reconciliation and post-
reconciliation; halo-reconciliation wall-clock as a fraction of total.

**Risk/walk-back:** if halo reconciliation cost is >20% of total
wall-clock or fails to converge for >1% of tile pairs, walk back to ADR
0012 walk-back §1 (global second pass on coarsened grid; reuse Slice 6
infra). If halo cells are >80% occupied on M3+ layers, walk back to ADR
0012 walk-back §2 (widen halo to 64); amend ADR 0012.

---

### Slice 8 — Terminal integration: Hazard3 chip-scale + WS3.3 exit criteria

**Deliverable:** `scripts/run_tile_router_hazard3.py` end-to-end script
+ a `tests/test_tile_router_4096.py` correctness gate.

**Implementation sketch:**

- Script: load full Hazard3 fixture, run `TileRouter.route`, capture
  the four headline measurements (wall-clock, conflicts, multi-tile-
  spanning fraction, halo occupancy) into `docs/results.md` Phase 3.3.
- Test: a 4096² × L=5 synthetic grid with ~500 nets routed by
  `TileRouter`; compared against `route_multipin_nets_3d` on the same
  grid for routed-fraction parity (within 1% absolute) and HPWL parity
  (within 5% mean). This is the WS3.3 4096² no-regression exit
  criterion.
- TritonRoute parity on Hazard3 (1.2× wire, 1.2× via) is the second
  WS3.3 exit criterion. Use the existing comparison infrastructure
  referenced in `docs/results.md` Phase 3.2.

**Tests added:**

- `test_tile_router_4096_no_regression_vs_untiled` — the 4096² gate.
  Slow test (mark `@pytest.mark.slow` if the suite has a marker
  convention; otherwise just budget ≤2 min wall-clock).

**Exit criterion:** both WS3.3 boxes in
`phase3-detailed-routing.md §WS3.3` flip from `[ ]` to `[x]`:

1. 4096² grid routed by tile-decomposition with no quality regression
   vs un-tiled.
2. Whole-chip Hazard3 integration competitive with TritonRoute (within
   1.2× wire, 1.2× vias).

**Measurements logged:** full Hazard3 chip-scale wall-clock, total
conflicts (must be 0), per-layer wire/via counts vs TritonRoute,
multi-tile-spanning fraction, halo occupancy.

**Risk/walk-back:** if either WS3.3 exit criterion misses, the root
cause analysis points to one of the earlier slices' walk-back triggers
(K-batch, coarsened-pass, or halo reconciliation). ADR 0012 amendment
captures the chosen walk-back; no re-architecting in this slice.

---

## Resolution / migration

When Slice 8 ships:

- Mark WS3.3 shipped in
  [`phase3-detailed-routing.md`](phase3-detailed-routing.md).
- Any architectural shift triggered by Slices 2/5/6/7 walk-backs lands
  as an ADR 0012 amendment in the **same commit** as Slice 8.
- Fold any in-flight follow-ups (kernel optimization, DEF pin
  extraction) back into a successor plan (Phase 4 sketch).
- `git rm docs/handoffs/ws33-tile-decomposition-handoff.md` per the
  handoff's own "Migration note" §.

## References

- [ADR 0012](../adr/0012-tile-decomposition.md) — design source of
  truth. Walk-back triggers in this plan map to its "Walk-back options".
- [ADR 0007](../adr/0007-hpwl-ascending-net-ordering.md) — ordering
  semantics used at tile and intra-tile granularity.
- [ADR 0008](../adr/0008-defer-route-nets-batched.md) — the per-tile
  conflict-detect/ripup work unlocked by Slice 5.
- [`phase3-detailed-routing.md`](phase3-detailed-routing.md) §WS3.3 —
  high-level design summary and WS3.3 exit criteria.
- [`../handoffs/ws33-tile-decomposition-handoff.md`](../handoffs/ws33-tile-decomposition-handoff.md)
  — current-session handoff; resolved by Slice 8's migration commit.
- [`../spikes/tier-a-sweep-sharing-throughput.md`](../spikes/tier-a-sweep-sharing-throughput.md)
  — Tier A empirical foundation (256² × L=5 × K=100 → ~31 ms/source).
- `scripts/tile_decomp_prototype.py` — substrate validator; reusable
  as a per-slice unit-of-debug.

## Open questions for the user

None. ADR 0012 is decisive on all design questions; this plan only
slices the build. The walk-back triggers in Slices 2, 4, 5, 6, 7 are
data-driven and land as ADR 0012 amendments at the slice they fire in,
not as upfront decisions.
