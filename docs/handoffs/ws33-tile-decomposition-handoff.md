# Handoff — WS3.3 tile decomposition: substrate validated, full chip-scale router still to build

**Created:** 2026-05-15 (rewritten after substrate validation session)
**Working tree:** clean
**Branch:** main

<!--
Reminder: a handoff is ephemeral. At resolution, every load-bearing piece
below migrates into a docs/adr/, docs/plans/, docs/spikes/, or design-doc
home, and this file is then `git rm`'d in the same commit as the migration.

See docs/handoff-discipline.md for the migration table.
-->

## Goal & next-up

**Goal of this session:** Lock the WS3.3 tile-decomposition design in
an ADR and validate the tile-shared cost-grid substrate on a real
Hazard3 tile, ahead of building the full chip-scale tile router.
Both done; substrate is correct. WS3.2 also closed out (per-pair
`via_cost` shipped as API plumbing — empirically silent on the
smallest-500 multi-pin spike, but structurally necessary for WS3.3).

**Next session should pick up:** [Phase 3 plan WS3.3 — Tile
decomposition](../plans/phase3-detailed-routing.md#ws33--tile-decomposition).
Specifically: build `src/gpu_pnr/tile_router.py` per
[ADR 0012](../adr/0012-tile-decomposition.md) — the full chip-scale
tile manager that iterates tiles, batches K=100 nets per tile via
`sweep_sssp_3d_multi`, reconciles halos, and routes
multi-tile-spanning nets on a coarsened grid first.

**Verification command:**

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/
# Expect: 69 passed

# Optional: re-run the tile prototype to confirm substrate correctness
uv run python scripts/tile_decomp_prototype.py 256 32 1.0
# Expect: 0 cross-net cell conflicts; 22% routability is the
# pin-extraction artifact, not a regression.
```

## Done this session

| Commit | Subject | Notes |
|---|---|---|
| `72de221` | Per-pair via_cost in 3D sweep, router, and reference (WS3.2 deliverable 5) | API change: scalar → `float \| Sequence[float] \| Tensor`. 8 new tests. |
| `fabb0d4` | Spike: per-pair via_costs CLI knob + multi-pin smallest-500 measurement | Negative result: per-pair has no effect on M1+M2-only workload. |
| `620e366` | docs: per-pair via_cost shipped — plan + results + ADR 0006 amendment | WS3.2 fully shipped (all exit criteria checked). |
| `1f46567` | docs: ADR 0012 — tile decomposition design for WS3.3 | Design captured; tile=256², K=100, halo=32 initial, iterative halo re-sweep, coarsened pass for multi-tile-spanning nets. |
| `26f0251` | WS3.3 tile prototype: route one 256² × 5 Hazard3 tile (ADR 0012) | Substrate validated: 0 conflicts, 22% routability all from pin-collision (extraction artifact, not a bug). |
| `ce88593` | docs: ADR 0012 Accepted; fold tile-prototype findings into Decision | ADR moves Proposed → Accepted with prototype evidence. |

## Open follow-ups (priority-ordered)

### 1. Full tile-decomposition router (large; ~1-2 days)

`src/gpu_pnr/tile_router.py` per [ADR 0012](../adr/0012-tile-decomposition.md).
Shape:

- `TileRouter` class API-compatible with `route_multipin_nets_3d`
  (`route(nets) -> list[MultiPin3DResult]`).
- Partition the chip into 256² owned tiles with halo=32 → 320²
  routable per tile. Iterate tiles in some order (HPWL-ascending
  per [ADR 0007](../adr/0007-hpwl-ascending-net-ordering.md) by tile, then by
  net within tile).
- Per-tile: batch up to K=100 nets through `sweep_sssp_3d_multi`,
  backtrace each, detect conflicts, ripup/reroute conflicts. The
  conflict-detect-and-ripup logic is the deferred
  [`route_nets_batched`](../adr/0008-defer-route-nets-batched.md);
  building it on top of a tile-bounded substrate is what makes it
  worth doing.
- Multi-tile-spanning nets (bbox + halo crosses tile boundary): route
  globally on a 4× coarsened grid first; pin those routes as
  obstacles in the per-tile passes.
- Halo reconciliation: after both adjacent tiles commit, re-sweep
  halo cells with both tiles' committed routes visible; ripup-reroute
  any halo conflicts. ADR 0012 picked option (a) iterative re-sweep;
  walk back to option (b) global-second-pass if local re-sweep
  doesn't converge.

Initial measurements to capture in `docs/results.md` Phase 3.3
section:
- Wall-clock per tile, end-to-end on Hazard3 (expect ~22-44 min total
  on MPS per the bull-case extrapolation).
- Cross-net conflicts (must stay 0).
- Multi-tile-spanning net fraction (estimated 5-15%; if >25%, revisit
  per-tile net assignment).
- Halo cell occupancy across tiles (refines halo width — 0% on the
  prototype's M1+M2 workload means insufficient signal yet).

### 2. Kernel optimization within tiles (medium; optional, post-WS3.3)

From `scripts/profile_chip_sweep.py` findings (commit `7acfd69`):

- `aten::where`: 49% (memory-bound)
- `aten::_local_scalar_dense`: 28% (.item() syncs in
  `_autotune_seg_barrier`)
- `aten::flip`: 13%

Headroom: 1.5-3× per-iter speedup via `torch.compile` operator
fusion + eliminating `.item()` syncs. **Defer until item 1 lands;**
chasing this before the full router exists is premature.

### 3. Pin extraction from DEF (small; orthogonal)

The tile prototype's 22% routability is an artifact of using
guide-rect centers as pin coords. Real ASIC routers pull pin
coordinates from the DEF's PIN / COMPONENT sections. Improving the
extraction would let the prototype (and ultimately the full router)
measure real routability without the cell-quantization collisions.
Orthogonal to the WS3.3 substrate; can land any time.

## Critical context

**ADR 0012 locked the substrate; the prototype confirmed it
works.** The next session does not need to re-prove tile-shared
correctness or re-pick tile size. The substrate (256² owned + 32
halo, sequential routing on shared `w_cur`) produces 0 cross-net
conflicts on real Hazard3 nets. The full router builds on top of
this without re-litigating fundamentals.

**The halo width may need to drop — but we don't know yet.** Halo
occupancy was 0% on the prototype's densest Hazard3 tile because
that tile is M1+M2-confined (670 of 694 cells on M2). The natural
use of halo is M3+ via-stack detours, which this workload doesn't
exercise. The full router will route nets across many tiles
including M3+ traffic; halo measurement there is the load-bearing
data for refinement. Don't drop halo speculatively.

**Multi-tile-spanning nets are the highest-risk piece.** The ADR
picks "coarsened-grid global first pass" for these. If on Hazard3
that fraction is >25% (vs the estimated 5-15%), or if the coarsened
pass is too lossy, walk back per ADR 0012's walk-back options.
Measure early.

**`sweep_sssp_3d_multi` K=100 throughput at 256² × 5 is empirically
~31 ms/source** (Tier A spike). The full router's per-tile K-batched
pass should hit that regime; if it doesn't, profile before
optimizing. ADR 0012's wall-clock estimate (22-44 min for Hazard3 on
MPS) depends on hitting it.

**Per-pair `via_cost` is API-only on this workload.** WS3.2
deliverable 5 shipped the structurally-correct per-pair plumbing,
but on Hazard3's smallest-500 multi-pin nets it has no measurable
effect — those nets are M1+M2-only. The 0.55× via-ratio gap vs
TritonRoute is **tree-topology**, not cost-model. The full tile
router doesn't change that; topology improvement is a separate
workstream if it ever lands. Per-pair will become a measurable
lever on nets that cross M3+ via stacks, expected in chip-scale
runs.

## References

- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md)
  — WS3.3 section. Captures the design summary and the queued
  prototype/next-slice.
- [ADR 0012](../adr/0012-tile-decomposition.md) — locked
  substrate design; Accepted with prototype evidence.
- [`../spikes/tier-a-sweep-sharing-throughput.md`](../spikes/tier-a-sweep-sharing-throughput.md)
  — Tier A empirical foundation: 256² × L=5 K=100 → 4.05× speedup
  at ~31 ms/source.
- [ADR 0006](../adr/0006-sequential-via-relax.md) — 3D via relax
  with per-pair via_cost amendment (commits `72de221`+`fabb0d4`).
- [ADR 0008](../adr/0008-defer-route-nets-batched.md) — the
  deferred conflict-detect/ripup work. The full router (item 1)
  finally unlocks this.
- `scripts/tile_decomp_prototype.py` — the single-tile prototype
  that validated the substrate. Reusable as a unit-of-debug for
  the full router (route a single tile in isolation, check
  invariants).

## Migration note

When item 1 (full tile router) lands and this handoff resolves:

- Open follow-up 1 → marks WS3.3 shipped in
  [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md);
  any new architectural decisions made during the build (e.g., halo
  width refinement, multi-tile-spanning policy change) → ADR 0012
  amendment.
- Open follow-up 2 → either a new ADR if `torch.compile` lands or a
  future-work note. Profile data already captured in commit
  `7acfd69`; no doc migration needed.
- Open follow-up 3 → tooling-only; commit + note in
  `_hazard3_io.py` if the DEF-parsing helper lands.
- Then `git rm docs/handoffs/ws33-tile-decomposition-handoff.md`
  in the migration commit. Commit message: `docs: resolve WS3.3
  handoff — fold into plan + ADR 0012 amendment`.
