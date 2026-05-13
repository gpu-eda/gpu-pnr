# Handoff — WS3.3 tile decomposition: build the chip-scale router using sweep-sharing per tile

**Created:** 2026-05-13
**Working tree:** clean
**Branch:** main

<!--
Reminder: a handoff is ephemeral. At resolution, every load-bearing piece
below migrates into a docs/adr/, docs/plans/, docs/spikes/, or design-doc
home, and this file is then `git rm`'d in the same commit as the migration.

See docs/handoff-discipline.md for the migration table.
-->

## Goal & next-up

**Goal of this session:** validated the WS3.3 architectural premise
(sweep-sharing amortises at 256² × K=100 = 4× in 3D, Tier A spike) and
empirically measured the worst-case "no-decomposition" baseline (chip-scale
single grid = 327 s per 2-pin net). Both pieces clear the runway for
WS3.3 itself, which is *not* yet built. The kernel-level optimization
profile is also captured for use during WS3.3 implementation.

**Next session should pick up:** [Phase 3 plan WS3.3 — Tile
decomposition](../plans/phase3-detailed-routing.md#ws33--tile-decomposition).
Open architectural questions are listed in the plan; start by writing the
WS3.3 design ADR or sketching `scripts/tile_decomp_prototype.py` (single
tile, real Hazard3 sub-region, full multi-source K-batched routing) per
"Critical context" below.

**Verification command:**

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/
# Expect: 61 passed

# Optional: regenerate the Tier A 3D bench numbers
uv run python scripts/bench_sweep_sharing.py --mode 3d --size 256 \
  --layers 5 --via-cost 5.0 --ks 1 10 25 50 100
# Expect: K=100 ~4x speedup, ~31 ms/source on M-series MPS
```

## Done this session

| Commit | Subject | Notes |
|---|---|---|
| `907f632` | Encode M1-as-pin-only as a PDK rule (WS3.2 deliverable 3) | Pdk dataclass + apply_pin_access_rules; N=500 spike numbers unchanged. |
| `eb42447` | docs: ADR 0011 + plan WS3.2 deliverable 3 marked shipped | ADR 0011 captures structural-vs-cost split. |
| `a7aa2d5` | Multi-pin nets via incremental tree growth (WS3.2 deliverable 4) | route_multipin_nets_3d + 8 new tests; 0.94×/0.60× vs TR at N=500 multi-pin. |
| `e687e41` | Renderer chip mode: --multipin flag for 3+-pin nets | Renderer supports multi-pin via _sample_nets generalisation. |
| `41bc65a` | docs: WS3.2 deliverable 4 marked shipped + results section | Multi-pin results captured. |
| `edd53a9` | Multi-pin safeguards: per-net timeout + progress logging + pin-count cap | net_timeout_s on route_multipin_nets_3d; MAX_PINS=20. |
| `e5dd5be` | Add sweep_sssp_3d_multi kernel + 3D sweep-sharing bench (Tier A) | The kernel WS3.3 will use; 4.05× at 256² × K=100. |
| `9e87e89` | docs: Tier A sweep-sharing spike resolved YES at 256³ | Tier A spike doc + results.md + plan update. |
| `7acfd69` | Chip-scale prototype + kernel profile: per-iter bottleneck breakdown | Chip-scale single-grid baseline + PyTorch profile findings. |

## Open follow-ups (priority-ordered)

### 1. WS3.3 design decision: write an ADR or jump straight to a tile prototype? (small-to-medium)

The architectural pieces are mostly empirically settled. **Locked design
parameters** (from Tier A):

- Tile size: 256² × L=5 layers. Memory ~6.5 MB per tile cost tensor; multi-source distance tensor for K=100 = 660 MB (well within the 30 GB MPS cap).
- K-batch size: 100 sources per `sweep_sssp_3d_multi` call gives peak 4.05× speedup.
- Sweep-sharing primitive: `sweep_sssp_3d_multi` (commit `e5dd5be`) — tested, validated.

**Open architectural decisions** (worth an ADR; ~1-2 hours to draft):

- **Halo width.** Must exceed the longest in-tile detour. Data-dependent; estimate from Hazard3's longest 2-pin route bbox (probably ~5-20 cells).
- **Halo reconciliation strategy.** Two candidates: (a) re-sweep within halos after each tile sees adjacent tiles' committed routes, or (b) global second pass on a coarsened grid. (a) is simpler to implement, (b) handles non-local detours better.
- **Multi-tile-spanning nets.** Nets whose bbox crosses tile boundaries: route them globally on a chip-scale grid first (slow but rare; ~5-15% of nets per the Tier A spike's analysis), or route them per-tile with halo handshake?
- **Per-tile net assignment.** Each tile owns some set of nets; assignment policy: bbox-center-in-tile, or split nets across tiles they touch.

### 2. Tile prototype: route one Hazard3 tile end-to-end (medium)

`scripts/tile_decomp_prototype.py`: pick a 256² × 5 sub-region of the
Hazard3 chip, identify all 2-pin and small multi-pin nets whose bbox
fits inside that single tile (skip nets crossing tile boundaries for
now), route them with `sweep_sssp_3d_multi` in K=100 batches. Measure:

- wall-clock per K=100 batch (expect ~340 ms from Tier A)
- successful-route fraction
- cross-net cell conflicts (should be 0 since we share the grid)
- comparison vs `route_multipin_nets_3d` on the same nets sequentially (baseline)

This validates the design before building the full chip-scale tile
manager. Probably 2-3 hours of code.

### 3. Full tile-decomposition router (large)

A new module — likely `src/gpu_pnr/tile_router.py` — that:

- Partitions the chip into 256² tiles with configurable overlap (halo).
- Assigns each net to a tile based on bbox.
- Routes each tile's nets via `sweep_sssp_3d_multi` (K=100 batches).
- Reconciles halos (re-sweep within overlap regions; see decision 1 above).
- Handles multi-tile-spanning nets via a fallback path.
- Returns per-net `MultiPin3DResult` data structurally compatible with
  the existing `route_multipin_nets_3d` API.

Roughly 1-2 days of focused work, gated on (1) and (2).

### 4. Kernel optimization within tiles (medium; optional, post-WS3.3)

From `scripts/profile_chip_sweep.py` findings (commit `7acfd69`):

- `aten::where`: 49% (memory-bound)
- `aten::_local_scalar_dense`: 28% (.item() syncs)
- `aten::flip`: 13%

Headroom: 1.5-3× per-iter speedup via `torch.compile` operator fusion +
eliminating `.item()` syncs in `_autotune_seg_barrier`. **Defer until WS3.3
is working;** chasing this before tile decomposition is premature.

## Critical context

**Tile decomposition is fundamentally different from per-net mini-grids.**
The current spike's per-net mini-grids (via `build_grid(rects)` per net) are
*a* form of decomposition — small grids, one per net, with no shared state.
Tile decomposition is fixed-size spatial tiles where *many nets share one
tile grid*. The key difference: per-net mini-grids can't prevent inter-net
cell conflicts (different nets routing through the same chip cell from
their own mini-grids); tile decomposition can, because all nets in a tile
see the same `w_cur`.

**`route_multipin_nets_3d` is the right interface, the wrong granularity.**
It already takes a list of nets, applies pin reservation across all of
them, and commits each net's cells as inf for the next. The "shared
working grid" semantics are exactly what tile decomposition needs. The
missing piece is *partitioning* — picking which nets go in which tile —
and *batching* — using `sweep_sssp_3d_multi` for K nets in parallel
instead of one-at-a-time. So the tile router can probably reuse most of
`route_multipin_nets_3d`'s pin-reservation logic.

**The chip-scale prototype (commit `7acfd69`) was the worst-case baseline,
not tile decomposition.** It builds one giant grid for the whole chip and
routes sequentially. The 327 s/net wall-clock is the *upper bound* —
proves tile decomposition is *needed*, not just *nice to have*. Don't
mistake the prototype for a step toward WS3.3; it was a "what if we
didn't decompose at all" experiment.

**Tier A numbers (commit `e5dd5be`) lock the tile-size choice.** Don't
re-litigate 256² vs 512² without new data — the 3D 512² K=100 collapse to
0.19× was measured and is in `docs/spikes/tier-a-sweep-sharing-throughput.md`.

**Wafer.space-scale extrapolation depends on this work landing.** The
"3-10× faster than TR at wafer.space scale" claim in `docs/results.md`
Phase 3.2 sweep-sharing section assumes WS3.3 lands with the predicted
per-tile throughput. If halo reconciliation costs much more than expected,
that ratio shrinks. Validate halo cost early (in tile prototype step 2).

## References

- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md)
  — WS3.3 section, "Status: not started; architecturally validated."
- [`../spikes/tier-a-sweep-sharing-throughput.md`](../spikes/tier-a-sweep-sharing-throughput.md)
  — Tier A spike, resolved YES at 256². The K-knee and tile-size
  decision rationale lives here.
- [`../results.md`](../results.md) — Phase 3.2 sweep-sharing section
  has the 2D and 3D bench tables and the wafer.space-scale
  extrapolation.
- [ADR 0008](../adr/0008-defer-route-nets-batched.md) — original
  sweep-sharing observation; deferred `route_nets_batched` until WS3.3.
  When WS3.3 lands, this ADR should be amended (or superseded) to
  reflect the actual delivered K-source architecture.
- [ADR 0010](../adr/0010-per-axis-cost-tensors.md), [ADR 0011](../adr/0011-pdk-rules-as-structural-constraints.md)
  — per-axis costs and PDK rules that the tile router must preserve.

## Migration note

When WS3.3 lands and this handoff resolves:

- Open follow-up 1 (halo width, reconciliation, multi-tile-net policy)
  → new ADR (probably 0012) capturing the WS3.3 design decisions.
- Open follow-up 2 (tile prototype) → mostly throwaway. Numbers from
  it go into `docs/results.md` Phase 3.3 section.
- Open follow-up 3 (full tile router) → marks WS3.3 shipped in
  `docs/plans/phase3-detailed-routing.md`.
- Open follow-up 4 (kernel optimization) → either a new ADR if
  `torch.compile` lands, or a future-work note. Profile data already
  captured in commit `7acfd69`; no doc migration needed.
- Then `git rm docs/handoffs/ws33-tile-decomposition-handoff.md` in
  the migration commit. Commit message: `docs: resolve WS3.3 handoff
  — fold into ADR 0012 + plan + results`.
