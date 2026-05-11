# Spike — Phase 3.2: does the sweep kernel route real LibreLane geometry end-to-end?

**Status:** Resolved (2026-05-11) — **YES**. Preferred-direction has since
landed ([ADR 0010](../adr/0010-per-axis-cost-tensors.md), commit `eee87bf`)
and ruled itself out as the cause of the residual TR gap; multi-pin nets
and per-via-pair via cost remain deferred to the
[Phase 3 plan](../plans/phase3-detailed-routing.md).

## Question

Does `sweep_sssp_3d` / `route_nets_3d` route real LibreLane post-GR geometry
end-to-end on a real ASIC fixture (Hazard3 on gf180mcuD), and what does its
output look like vs TritonRoute's?

## Outcome

The kernel routes real geometry without modification — 500/500 sampled small
2-pin nets route end-to-end at 41–50 ms each. Two design adjustments surfaced
during the spike and were folded into [ADR 0005](../adr/0005-mask-based-segmented-scan.md)
(`SEG_BARRIER` per-call autotune, Decision §4) and the M1-cost knob experiment.
Aggregate via ratio vs TritonRoute closed from 0.15× (no pin-access model) to
0.78× (`m1_cost=10`); the remaining gap is consistent with unmodelled per-layer
preferred direction, which is the next slice in the
[Phase 3 plan](../plans/phase3-detailed-routing.md).

The full narrative below is preserved as the spike record.

## Original notes

This document captures the results of a deliberately-tiny Phase 3.2 spike: take
one net from a real LibreLane GR run for Hazard3 level_3 on gf180mcuD, build a
cost grid from its guide rectangles, route it through `sweep_sssp_3d` /
`route_nets_3d`, and write down what was surprising.

The fixture used is the LibreLane run at
`~/Code/Apitronix/hazard-test/hazard3/librelane/runs/RUN_2026-05-08_22-32-24/`
(50,099 instances, 5 metal layers, gf180mcuD, 0.20um wire pitch). All inputs
came from `39-openroad-globalrouting/after_grt.guide` -- 24,123 nets in
human-readable format, parsed by ad-hoc Python in 30 lines (`scripts/spike_route_one_net.py`).

No LibreLane execution was needed; no LEF/DEF parsing was needed for the spike
itself. Pin coordinates came from the centers of the first/last `Metal1`
rectangles in the guide -- correct for 2-pin nets where the two Metal1 patches
*are* the two pins.

## What worked

**Trivial 2-pin same-layer case (`_00013_`, 5 rectangles).**
Source and sink one gcell apart on Metal1. Sweep converges in 16 iterations,
distance is exactly the column delta (84 cells), zero vias, path is a straight
Metal1 line. Multi-net `route_nets_3d` agrees. Sanity-check pass.

**Multi-layer 2-pin case (`_00000_`, 10 rectangles, M1->M2->M3->M2->M1 guide).**
Source and sink 1344 cells apart in Manhattan, on opposite corners of the
guide. Sweep converges in 16 iterations, distance is 1364 = 1344 + 4 vias x 5
via_cost. Path uses Metal1, Metal2, Metal3 with 4 via transitions; takes M2
vertically from source to mid-grid, M3 horizontally across, M2 vertically down,
M1 to sink. Sensible layer hierarchy use even without preferred-direction
modelling.

End-to-end, the kernel runs on real LibreLane geometry without modification.
That's the headline.

## What didn't work, and why

**`SEG_BARRIER=2e4` (the Phase 3.1 module-constant default) corrupts distances
on real per-net guides.**

Synthetic Phase 3.1 tests had ~5% obstacle density across the whole grid (~200
obstacle cells in any one row of 4096). Real per-net guides invert this:
93% of the grid is "outside the routable region" (i.e., obstacle), so a typical
row has *thousands* of consecutive obstacle cells. For `_00000_` row 42 of
Metal1 has 924 consecutive obstacle cells; `seg_id * SEG_BARRIER` reaches
`924 * 2e4 = 1.85e7`, where float32 ULP is ~2-3. Distances of order 1000 get
corrupted by integer-scale errors and the kernel reports `d[sink] = 984`, which
is *less than the Manhattan minimum (1344)*. Backtrace then fails because no
valid predecessor chain exists.

Empirically:

| `SEG_BARRIER` | iters | `d[sink]` | Note |
|---|---|---|---|
| 2e4 | 240 | 984.0 | bogus -- below Manhattan |
| 1e4 | 16 | 1364.0 | correct |
| 5e3 | 16 | 1364.0 | correct |
| 2e3 | 16 | inf | polluted-mask false-positives (threshold too low) |

The valid range is `2 * max_legit_distance < SEG_BARRIER < 1.6e7 / max_seg_id`.
For this spike: `2 * 1364 = 2728 < SEG_BARRIER < 1.6e7 / 924 = 17,316`. The
Phase 3.1 default `2e4` is just outside this range.

The spike script accepts `SEG_BARRIER` as a CLI argument; `5e3` works for these
nets. **The proper fix is per-call auto-tuning**: derive `SEG_BARRIER` from
`(grid_size, max(seg_id_per_row))` at sweep entry. That's a small kernel API
change (constructor argument or computed-from-w default) and is the next
natural piece of work.

## What was deliberately punted in this spike (and remains punted)

1. **Multi-pin nets.** Hazard3 has many >2-pin nets; we only handled the
   12,770 two-pin ones in the guide. Multi-pin needs Steiner-tree-flavored
   handling on top of route_nets_3d.
2. **Preferred routing direction.** gf180mcuD has Metal1=H, Metal2=V,
   Metal3=H, ... Our cost model is uniform per-cell. Real routes here would
   penalize non-preferred-direction segments. We got away with it because
   the guide regions already constrain layer use to align with preferred
   direction (the GR step picked layer-direction allocation for us); but
   for tighter optimality this needs to be modelled.
3. **Per-via-pair `via_cost`.** Single scalar `via_cost` is a simplification.
4. **DRC compliance.** Not checked.
5. **Comparison to TritonRoute.** We produced a path; we haven't yet loaded
   `final/def/synth_top_level_3.def` to see what TritonRoute did for the same
   nets. That comparison is the next obvious deliverable.
6. **LEF/DEF parsing.** The guide-only approach worked for 2-pin nets where
   pin coords are inferable from Metal1 patch centers. Multi-pin or
   pin-on-non-Metal1 cases will need real LEF parsing. The `lefdef` PyPI
   package was tried first and **does not work on macOS** (ships only Windows
   `.dll` and Linux `.so`, no `.dylib`). Either build it from source, switch
   to ad-hoc DEF parsing, or use OpenROAD's `odb` Python (which requires an
   OpenROAD install).

## SEG_BARRIER autotune (landed)

Replaced the module-constant `SEG_BARRIER=2e4` with a per-call autotune that
derives the value from grid shape and obstacle density:

```
lower = 2 * max_legit_distance_estimate     # H+W times max_w_finite, plus L*via_cost
upper = FLOAT32_PRECISION_BUDGET / max_seg_id   # 1e7 / max obstacle count per axis
seg_barrier = sqrt(lower * upper)           # geometric mean of valid range
```

If `lower >= upper` the autotune falls back to `upper * 0.99` and the workload
is documented as exceeding the float32 precision budget (this is what 8192^2
unit-weight grids hit -- the new wall, same place Phase 3.1 documented).

**Cost of the autotune:** ~3 GPU syncs per sweep call (~1.5ms on MPS):
- one masked `max(w_finite)` reduction
- two `cumsum(obstacle_mask) + max + .item()` per spatial axis
At Phase 3.1's 1024^2 sweet spot of ~94ms/sweep, the autotune adds ~1.6%
overhead. For per-net routing the autotune fires per net (each has its own
obstacle pattern), but the per-net work itself is dominated by the
convergence loop, so the autotune cost is in the noise.

**Synthetic perf preserved:** post-autotune `bench_scaling.py` numbers are
within run-to-run noise of the post-Phase-3.1 hoisted-precompute version
(1024^2: 2.34 -> 2.41 ms/iter; 4096^2: 31.34 -> 32.83 ms/iter). 4096^2 still
correct; 8192^2 still hits the precision wall (best-effort fallback).

**Real-fixture perf unblocked:** `_00000_` now routes correctly with no
manual override -- d[sink]=1364 in 16 iterations.

## Multi-net spike (landed)

`scripts/spike_route_many_nets.py` runs `route_nets_3d` on N independent
2-pin nets (each with its own per-net grid built from its guide rectangles).
Sample of the 50 smallest 2-pin nets:

```
=== Aggregate over 50 nets ===
  routed: 50 / 50 (100.0%)
  total wirelength: 7664 cells
  total via transitions: 20
  avg per-net time: 51.1 ms

Layer occupancy (number of routed nets that used the layer):
  Metal1: 50
  Metal2: 10
  Metal3: 0
  Metal4: 0
  Metal5: 0
```

Scaling to 200 and 500 nets: still 100% routed, per-net time stable at 41-45ms.
Most short nets stay on M1 because M1 cost (1 per cell) beats via_cost (5 per
via, 4 vias minimum to use M3 = 20 cost) for short routes. This is the
expected cost-model behavior; preferred-direction modelling would push more
of them off M1, which is left for the next iteration.

**Per-net latency is launch-overhead-dominated** at these tiny per-net grids.
50K Hazard3 nets at 41ms = ~35 minutes total -- comparable to TritonRoute on
a desktop, but kernel-launch overhead would drop substantially with batching
or `torch.compile`. The autotune's ~1.5ms is a small fraction of the per-net
budget here.

## TritonRoute comparison (landed)

The post-DR DEF (`final/def/synth_top_level_3.def`) contains TritonRoute's
actual per-net wires + vias in the standard DEF NETS-section format. A
~80-line ad-hoc parser (`parse_def_nets` in `scripts/_hazard3_io.py`)
extracts per-net wirelength (sum of segment Manhattan distances) and via
count (segments ending in `Via*` token names) in 0.3s for the full 24K-net
design. `RECT` annotations and multi-line connection lists are handled.

Hand-traced two nets to validate: `_00013_` (117 cells, 4 vias) and
`_00000_` (1495 cells, 4 vias) match the parser exactly.

Aggregate vs our `route_nets_3d` over the smallest N 2-pin nets:

| Sample | Our wire | TR wire | wire ratio | Our vias | TR vias | via ratio |
|---|---|---|---|---|---|---|
| 50 nets | 7664 | 7180 | 1.07x | 20 | 132 | 0.15x |
| 200 nets | 21884 | 17695 | 1.24x | 44 | 512 | 0.09x |
| 500 nets | 48622 | 36567 | 1.33x | 70 | 1246 | 0.06x |

### What this tells us

1. **TritonRoute uses ~10x more vias because it pays the pin-access tax.**
   In real gf180mcuD designs, Metal1 is reserved for intra-cell routing;
   any inter-cell wire has to hop M1->M2 (or higher) at each pin before
   traversing, then back down to M1 at the destination. That's at least
   2 mandatory vias per net just for pin access. TritonRoute's average of
   ~2.5 vias/net at the smallest 500 matches: 2 access vias + occasional
   layer changes for routing optimization. Our router doesn't know about
   this constraint -- it sees M1 as freely routable wire and stays there
   whenever the guide allows. 0.4 vias/net on average is "no vias except
   when forced by the guide topology."
2. **Our wire ratio looks competitive (1.07x at 50 nets) but is
   misleading.** Both ends are on M1; our straight-line on-M1 path simply
   isn't a legal route in real ASIC routing. If we modelled pin access
   correctly (force every net to use M2+ for the wire body), our wire
   length would jump because the M2/M3 routes that TritonRoute takes
   often need detours around obstacles in those layers, plus the via
   stack itself contributes a few cells of "wire" at each end.
3. **Wire ratio grows with sample size.** 1.07x at 50 nets to 1.33x at
   500 nets. Larger nets have more degrees of freedom; our router
   exploits the M1-stays-cheap loophole more aggressively at scale.

### What this means for the cost model

The next-priority item in this list shifts: **preferred-direction +
M1-pin-only cost model** is now the main thing keeping the comparison
honest, not a polish on top of "we're already close." We're not really
within 7% of TritonRoute on wirelength -- we're 10x cheating on vias
and the wirelength happens to be close because our tiny grids collapse
nicely on M1.

## M1-as-pin-access experiment (landed)

Hypothesis: applying a large multiplier to Metal1 wire cost forces the
router to via-stack from each pin up to M2+ for the wire body and back
down at the sink, approximating gf180mcuD's pin-access-only convention
for M1 -- and the via count should converge towards TritonRoute's.

`scripts/spike_route_many_nets.py` got an `m1_cost` CLI argument that
multiplies the cost of every Metal1 wire cell by that factor (applied
post-`build_grid`, before the route call). Sweep at N=50 nets:

| `m1_cost` | wire ratio | via ratio |
|---|---|---|
| 1.0  | 1.07x | 0.15x |
| 10   | 1.08x | 0.76x |
| 100  | 1.08x | 0.76x |
| 1000 | 1.08x | 0.76x |

Saturates at `m1_cost=10`. Beyond that, the router consistently
via-stacks instead of routing on M1, and adding more penalty doesn't
change the topology. The wire ratio barely moves (vias add 2 path cells
per net for the via-stack endpoints, ~+1% aggregate).

Scaled sample with `m1_cost=10`:

| Sample | wire ratio | via ratio |
|---|---|---|
| 50  | 1.08x | 0.76x |
| 200 | 1.26x | 0.78x |
| 500 | 1.36x | 0.80x |

The via ratio is now stable across sample sizes (~0.78x) instead of
drifting from 0.15x to 0.06x. The 1.07x->0.78x of-via gap closure is
the pin-access tax the experiment was designed to surface.

**The remaining ~20% via deficit** (0.78x rather than 1.0x) is the next
layer of unmodelled constraints. Most likely candidates:

1. **Preferred-direction transitions.** gf180mcuD's M2 prefers vertical,
   M3 horizontal, M4 vertical, etc. TritonRoute pays one or two extra
   vias per direction change to keep wire on the correct layer.
2. **Per-via-pair cost asymmetry.** Real PDKs have different via
   resistance and DRC for each layer pair (M1-M2 cheaper than M3-M4).
   Our scalar `via_cost` averages this away.
3. **Multi-pin Steiner topology.** Our `route_nets_3d` is point-to-point;
   even 2-pin nets in this comparison are subsets of larger nets in
   the actual fixture, where TritonRoute's tree may add intermediate
   layer hops.

The wire ratio (1.08x -> 1.36x with sample size) is probably the same
preferred-direction story: bigger nets give us more rope to take a
suboptimal M2-only L-shape where TritonRoute would have gone direct on
M3 horizontal.

## Next steps

In rough priority order, post-comparison:

1. **Per-layer preferred direction.** Now that the via gap is mostly
   closed by M1 penalty, the next ~20% of via deficit and the
   sample-size-dependent wire-ratio drift are both consistent with
   "we don't know M2 prefers vertical, M3 horizontal." Cheapest
   model: per-axis cost multiplier per layer (e.g., `axis_cost[layer]
   = (h_cost, v_cost)`). Slightly bigger surgery: separate H-edge and
   V-edge costs per cell. Decide based on what the kernel can absorb
   without breaking the cumsum-cummin trick.
2. **Multi-pin nets.** ~11,000 of the 24K nets have 3+ Metal1
   rectangles. Router-level change (sequential point-to-point
   construction with re-rooting, or Steiner-tree-flavored heuristic).
3. **Per-via-pair `via_cost`.** Replace the scalar with a length-(L-1)
   array. Tiny API change, finishes the realism story.
4. **Whole-chip integration.** Replace per-net mini-grids with a
   chip-scale grid that tracks committed routes globally. Gates on
   (1) and probably tile decomposition (Phase 3.3) to fit at scale.

## Preferred-direction experiment (landed 2026-05-11, commit `eee87bf`)

`sweep_sssp_3d` / `route_nets_3d` were generalised to accept an optional
per-axis cost tensor `w_v`; `axis_costs(w, h_mult, v_mult)` builds the
factored (w_h, w_v) pair from gf180mcuD's per-layer alternation. The
spike script's `m1_cost` knob was replaced by an `off_mult` arg.
Implementation decision: [ADR 0010](../adr/0010-per-axis-cost-tensors.md).

Per-net topology now matches the cost model — a pure-V Hazard3 net that
previously ran illegal V wire on M1 now via-stacks to M2 for the wire
body and back to M1 at the sink (2 vias, correct topology).

### Aggregate TR comparison (off_mult=10)

| Sample | wire ratio | via ratio | M1 use   | M2 use   | M3+ use |
|--------|-----------|-----------|----------|----------|---------|
| 50     | 1.08×     | 0.76×     | 50/50    | 50/50    | 0       |
| 200    | 1.26×     | 0.78×     | 200/200  | 200/200  | 0       |
| 500    | 1.36×     | 0.80×     | 500/500  | 500/500  | 0       |

Numbers are **identical to the `m1_cost=10` table** above. The numbers
saturate immediately — flat across `off_mult ∈ [2.0, 100.0]` at every
sample size. For these small in-guide 2-pin nets, "M1 wire cost is
expensive" and "per-layer preferred direction" produce the same
topology: one via-stack at each pin and a single intermediate layer
for the wire body.

### Amendment to the "what this means" prediction above

The original "what this tells us" section listed three candidates for
the residual 20% via gap: preferred-direction transitions, per-via-pair
cost asymmetry, and multi-pin Steiner topology. The preferred-direction
candidate is now ruled out by data — its strongest implementation
(option B in ADR 0010) produces TR numbers identical to the `m1_cost`
knob. The two remaining candidates own the gap:

1. **Per-via-pair cost asymmetry** ([plan deliverable 3](../plans/phase3-detailed-routing.md#ws32--real-fixture-integration-hazard3-on-gf180mcud)).
   gf180mcuD's M1-M2 via differs from M3-M4 via in resistance and DRC.
   Our scalar `via_cost` averages this away, which probably biases the
   router toward over-using the M1-M2 pair.
2. **Multi-pin Steiner topology** ([plan deliverable 2](../plans/phase3-detailed-routing.md#ws32--real-fixture-integration-hazard3-on-gf180mcud)).
   Our 2-pin sample is a subset of larger nets in the actual fixture;
   TritonRoute's Steiner trees add via hops that point-to-point
   routing doesn't see, even on the "2-pin" subset.

The wire-ratio drift (1.08× → 1.36× with sample size) likewise survives
preferred-direction unchanged. Probably the same per-via-pair story —
larger nets cross more layer pairs, where our averaged scalar
diverges further from the real per-pair cost structure.

## Files added

- `scripts/_hazard3_io.py` -- shared parsers (guides, post-DR DEF NETS) and
  grid construction. Used by both spike scripts.
- `scripts/spike_route_one_net.py` -- single-net debugging driver. Accepts a
  net name and an optional SEG_BARRIER override.
- `scripts/spike_route_many_nets.py` -- multi-net aggregate-stats driver.
  Includes TritonRoute comparison (wire-length and via count ratios).
- `docs/spikes/phase32-hazard3-real-fixture.md` -- this document.
- `~/.claude/projects/-Users-roberttaylor-Code-gpu-pnr/memory/hazard3_fixture.md`
  -- reference memory for the fixture location.
