# Plan — Phase 3: detailed routing on real fixtures

**Status:** Active.

## Goal

Take the sweep-based router from working on synthetic grids to working on real
ASIC routing fixtures (Hazard3 on gf180mcuD), closing the most load-bearing
gaps versus TritonRoute one at a time. The implementing ADRs are
[0005](../adr/0005-mask-based-segmented-scan.md) (mask-based obstacles),
[0006](../adr/0006-sequential-via-relax.md) (3D vias), and
[0009](../adr/0009-per-net-grids.md) (per-net independent grids for the spike).

## Prerequisites

- ADR 0001–0008 accepted.
- Hazard3 LibreLane fixture present at
  `~/Code/Apitronix/hazard-test/hazard3/librelane/runs/RUN_2026-05-08_22-32-24/`
  (memo: see `~/.claude/projects/-Users-roberttaylor-Code-gpu-pnr/memory/hazard3_fixture.md`).
- 43/43 tests green.

## Where things stand (2026-05-11)

- **3.1 mask-based obstacles** — shipped. Replaced INF_PROXY with a true
  segmented scan (ADR 0005). 4096² grids now correct; new precision wall
  between 4096² and 8192².
- **3.4 multi-layer + via cost** — shipped. `sweep_sssp_3d` and
  `route_nets_3d` route through an `(L, H, W)` cost tensor with via
  transitions (ADR 0006).
- **3.2 real-fixture spike** — landed. Single-net, multi-net, TritonRoute
  comparison, M1-pin-access experiment all done. See
  [`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md).
  Remaining work: preferred-direction modelling, multi-pin nets, per-via-pair
  via_cost (see WS3.2 below).
- **3.3 tile decomposition** — not started. Gated on preferred direction
  landing first (so the post-tile TritonRoute comparison is interpretable).

## Workstreams

### WS3.1 — Mask-based obstacle handling

**Status:** Shipped 2026-05-10 (`bee24df`, `10e128c`).

Implemented per [ADR 0005](../adr/0005-mask-based-segmented-scan.md). The
module-constant `SEG_BARRIER` was replaced with a per-call autotune in WS3.2;
those changes are tracked there but the kernel-side decision sits in
ADR 0005's Decision §4.

### WS3.4 — Multi-layer + via cost

**Status:** Shipped 2026-05-10 (`5019127`, `b77e5d4`).

Implemented per [ADR 0006](../adr/0006-sequential-via-relax.md).

### WS3.2 — Real-fixture integration (Hazard3 on gf180mcuD)

**Status:** Preferred-direction shipped (2026-05-11, commit `eee87bf`,
[ADR 0010](../adr/0010-per-axis-cost-tensors.md)); multi-pin nets and
per-via-pair `via_cost` remain.

Spike outcomes captured in
[`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md).
Headline measurements after preferred-direction (commit `eee87bf`):

| Sample | wire ratio vs TR | via ratio vs TR |
|--------|------------------|-----------------|
| 50     | 1.08×            | 0.76×           |
| 200    | 1.26×            | 0.78×           |
| 500    | 1.36×            | 0.80×           |

Numbers tie with the spike's `m1_cost=10` row exactly; preferred-direction
is a kernel-correct generalisation of the M1-cost knob but does not close
the residual gap on its own. The residual ~20% via gap and the 1.08× →
1.36× wire-ratio drift are now attributable to per-via-pair cost
asymmetry (deliverable 3) and multi-pin Steiner topology (deliverable 2);
see the spike doc's "Preferred-direction experiment" amendment.

**WS3.2 deliverables (priority order):**

1. **Per-layer preferred direction.** Shipped 2026-05-11, commit `eee87bf`,
   [ADR 0010](../adr/0010-per-axis-cost-tensors.md). The kernel now takes
   `(w, w_v)` with `axis_costs(w, h_mult, v_mult)` as the option-A builder
   on top of the general option-B surface.
2. **Multi-pin nets.** ~11K of 24K Hazard3 nets have 3+ pins; spike was 2-pin
   only. Sequential point-to-point with re-rooting, or a Steiner-tree-flavored
   heuristic. **Promoted to next:** the preferred-direction experiment ruled
   itself out as the cause of the residual gap, so this is now load-bearing.
3. **Per-via-pair `via_cost`.** Replace the scalar with a length-`(L-1)` array.
   Tiny API change; finishes the realism story. Defer until (2) lands so the
   TR-comparison deltas are interpretable.

**Exit criteria for WS3.2:**

- [x] Preferred-direction landed; via ratio vs TR closes from ~0.78× toward
      ~1.0×, wire ratio stabilises across sample sizes (no longer drifts
      1.08× → 1.36×).
      *Landed but did not close the gap — numbers tie with `m1_cost=10`.
      Hypothesis falsified, gap re-attributed to deliverables 2 and 3.*
- [ ] Multi-pin nets supported by `route_nets_3d`; at least 80% of Hazard3's
      multi-pin nets route end-to-end.
- [ ] Per-via-pair `via_cost` plumbed through; TR comparison re-run with
      per-pair gf180mcuD values.

### WS3.3 — Tile decomposition

**Status:** Not started. Gated on WS3.2.1 (preferred direction) landing first.

Splits a too-big grid (e.g., chip-scale) into overlapping tiles, routes within
each, reconciles at halos. Unlocks two things:

1. Whole-chip integration (cells beyond the float32 precision wall — see
   [ADR 0005](../adr/0005-mask-based-segmented-scan.md) walk-back).
2. The multi-source kernel's 3.10× regime per tile, enabling
   [`route_nets_batched`](../adr/0008-defer-route-nets-batched.md) on top.

Design choices to make when this starts:

- Tile size (256² is the sweet spot for the multi-source kernel; the natural
  default).
- Halo width (must exceed the longest in-tile detour; data-dependent).
- Halo reconciliation strategy: re-sweep within halos with both tiles'
  committed routes visible, or run a global second pass on a coarsened grid.

**Exit criteria for WS3.3:**

- [ ] A 4096² grid is routed by tile-decomposition with no quality regression
      vs un-tiled at the same scale (4096² being the current correctness
      ceiling, see ADR 0005).
- [ ] Whole-chip integration on Hazard3 produces results competitive with
      TritonRoute (within 1.2× wire, within 1.2× vias).

## Phase 3 exit criteria

When all of these are true, this plan closes and a Phase 4 plan opens:

- [ ] WS3.2 fully shipped (preferred direction, multi-pin, per-via-pair).
- [ ] WS3.3 fully shipped (tile decomposition + whole-chip integration).
- [ ] Updated TritonRoute comparison numbers documented in
      [`../results.md`](../results.md).
- [ ] Phase 4 sketches (DRC kernel co-iteration, CUDA port) promoted to a
      successor plan.

## References

- [ADR 0005](../adr/0005-mask-based-segmented-scan.md) — mask-based obstacles.
- [ADR 0006](../adr/0006-sequential-via-relax.md) — 3D via relax.
- [ADR 0008](../adr/0008-defer-route-nets-batched.md) — sweep-sharing
  deferred until tile decomposition.
- [ADR 0009](../adr/0009-per-net-grids.md) — per-net grids for the spike.
- [`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md)
  — spike results that motivate this plan.
- [`../results.md`](../results.md) — benchmark numbers.
