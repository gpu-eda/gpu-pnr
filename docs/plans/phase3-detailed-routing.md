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

**Status:** Preferred-direction + `m1_penalty` landed (commits `eee87bf`,
`8ecc95c`, [ADR 0010](../adr/0010-per-axis-cost-tensors.md)). Visualization
infrastructure (`scripts/render_routes.py`) landed too (commit `2dc5b1c`).
Multi-pin nets, per-via-pair `via_cost`, and M1-as-PDK-rule encoding
remain.

Spike outcomes captured in
[`../spikes/phase32-hazard3-real-fixture.md`](../spikes/phase32-hazard3-real-fixture.md).
Headline numbers from the smallest-500 spike sample after the
preferred-direction landing:

| Sample | wire ratio vs TR | via ratio vs TR |
|--------|------------------|-----------------|
| 50     | 1.08×            | 0.76×           |
| 200    | 1.26×            | 0.78×           |
| 500    | 1.36×            | 0.80×           |

These numbers tied with the spike's `m1_cost=10` row exactly, which we
initially read as "PD is a kernel-correct generalisation of `m1_cost`."
The **full-chip 12,770-net visualization falsified that** (see
[`../results.md`](../results.md) Phase 3.2 investigation section): PD
alone leaves M1-H cheap on the preferred axis, so the router routinely
runs ~100K cells of illegal M1-H wire at full chip scale. Adding
`m1_penalty=10` (both-axes penalty on M1) drops M1 wire 91% and pushes
the displaced traffic onto M3 (ours/TR M3 ratio: 0.41× → 0.73×). M4 and
M5 binned counts are *literally unchanged* by m1_penalty — those nets'
decisions are driven by long-haul distance, not the M1↔M3 cost balance,
so closing the M4/M5 gap needs a different lever.

**WS3.2 deliverables (priority order, revised):**

1. **Per-layer preferred direction.** Shipped 2026-05-11, commit `eee87bf`,
   [ADR 0010](../adr/0010-per-axis-cost-tensors.md). Kernel now takes
   `(w, w_v)`; `axis_costs(w, h_mult, v_mult)` builds the factored option-A
   form on top of the general option-B surface.

2. **M1-penalty / pin-only knobs.** Shipped 2026-05-11, commit `8ecc95c`.
   `m1_penalty` (both-axes cost) and `m1_pin_only` (inf mask except pins +
   landing pad) are available as experiment-time knobs on
   `scripts/spike_route_many_nets.py` and `scripts/render_routes.py`. The
   "pin-only as PDK rule" refactor (next deliverable) supersedes them.

3. **Encode M1-as-pin-only as a PDK rule.** Shipped 2026-05-11, commit
   `907f632`. New `Pdk` dataclass + `GF180MCUD` instance in
   `scripts/_hazard3_io.py` holds technology constraints (layer order,
   preferred direction, pin-access-only layer indices, pitch).
   `apply_pin_access_rules(w, pdk, pin_cells)` is the structural
   encoding of the no-wire-on-M1 DRC rule. Both scripts apply rules by
   default; `--no-pdk-rules` is a legacy bypass for ablation studies.
   Numbers match the prior `m1_pin_only=True` knob; behaviour is now
   the default instead of opt-in.

4. **Multi-pin nets.** Shipped 2026-05-11, commit `a7aa2d5`. Kernel
   gains `extra_sources` for multi-source SSSP / multi-target
   backtrace; `route_multipin_nets_3d` implements incremental tree
   growth (seed at pins[0], each iteration runs SSSP from the current
   tree, attaches the closest unrouted pin via backtrace, re-iterates
   until all pins connected). New `MultiPin3DResult` dataclass holds
   the per-edge paths and dedup'd cells set. Renderer chip mode gains
   `--multipin` flag (commit `e687e41`). Spike script's 6th positional
   arg switches to multi-pin sampling.

   Headline TR comparison on the smallest 500 multi-pin (3+-pin) nets:

   | Metric | ours | TR | ratio |
   |---|---|---|---|
   | routed | 500 / 500 (100%) | — | — |
   | wirelength | 130,725 cells | 139,583 | 0.94× |
   | vias | 1,671 | 2,773 | **0.60×** |
   | M3 use | 85 / 500 nets (17%) | — | — |

   We use *less* wire than TR (0.94×) and *substantially fewer* vias
   (0.60×). Likely because our sequential-attachment trees are more
   compact than TR's Steiner topology on these small mini-grids, and
   because the per-via-pair-cost gap (deliverable 5) is now load-bearing
   for the via ratio. 100% routing success on the smallest 500 nets
   exceeds the 80% exit criterion.

5. **Per-via-pair `via_cost`.** *Next.* Replace the scalar with a
   length-`(L-1)` array. Targets both the M4/M5 gap from earlier work
   and the 0.60× via ratio from the multi-pin spike. With per-pair
   costs reflecting gf180mcuD's Via1/Via2/Via3/Via4 resistance and DRC
   asymmetry, the router should pick a topology closer to TR's.

**Soft-guide note (not promoted to a deliverable).** The chip-scale
visualization revealed that ~6,000 small nets are guide-locked to M1+M2
(no M3 in their GR allocation) and our router can't escape because
off-guide cells are `inf`. The natural fix is **chip-scale routing
(WS3.3)** which drops per-net mini-grids entirely; making guides a soft
preference instead is an interim hack we'd throw away. Skip it.

**Exit criteria for WS3.2:**

- [x] Preferred-direction landed; ADR 0010 captures the decision.
- [x] M1-as-pin-access encoding (knob form, commit `8ecc95c`).
- [x] M1-as-pin-only encoded as a PDK rule (commit `907f632`). Defaults
      on; bypass via `--no-pdk-rules` for ablation.
- [x] Multi-pin nets supported by `route_multipin_nets_3d` (commit
      `a7aa2d5`). 500/500 (100%) of smallest multi-pin nets routed
      end-to-end -- exceeds the 80% target.
- [ ] Per-via-pair `via_cost` plumbed through; TR comparison re-run with
      per-pair gf180mcuD values. Particular focus on closing the M4/M5
      gap that m1_penalty didn't move plus the 0.60× via ratio that
      the multi-pin spike surfaced.

### WS3.3 — Tile decomposition

**Status:** Not started. Gated on WS3.2 deliverables 3 and 4 landing
(M1-as-PDK-rule encoding + multi-pin nets).

Splits a too-big grid (e.g., chip-scale) into overlapping tiles, routes within
each, reconciles at halos. Unlocks **three** things:

1. Whole-chip integration (cells beyond the float32 precision wall — see
   [ADR 0005](../adr/0005-mask-based-segmented-scan.md) walk-back).
2. The multi-source kernel's 3.10× regime per tile, enabling
   [`route_nets_batched`](../adr/0008-defer-route-nets-batched.md) on top.
3. **Naturally subsumes "guide as soft preference"** — once we route on a
   single global cost tensor instead of per-net mini-grids, GR allocation
   becomes advisory because every cell already exists in the grid. Closes
   the small-net-cohort gap that m1_penalty/pin-only couldn't reach
   (see [`../results.md`](../results.md) Phase 3.2 investigation,
   Finding 4).

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
