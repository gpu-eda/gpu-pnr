# Phase 1 / 2 results

All numbers from `scripts/bench_scaling.py` and `scripts/demo_multinet.py` on an
Apple Silicon M-series with PyTorch 2.11 MPS. Seed 42, 5% obstacle density.

## Single-net SSSP scaling (Phase 1, INF_PROXY-based)

| Size | Cells | Sweep (MPS) | Iters | ms/iter | Mcells/s | Dijkstra (CPU) | Speedup | Status |
|---|---|---|---|---|---|---|---|---|
| 256² | 65K | 41 ms | 24 | 1.72 | 1.6 | 45 ms | 1.09× | ✓ |
| 512² | 262K | 39 ms | 24 | 1.64 | 6.7 | 186 ms | 4.73× | ✓ |
| **1024²** | **1.05M** | **83 ms** | **40** | **2.06** | **12.7** | **784 ms** | **9.51×** | ✓ |
| 2048² | 4.19M | 381 ms | 64 | 5.95 | 11.0 | 3344 ms | 8.78× | ✓ |
| 4096² | 16.8M | 2596 ms | 104 | 24.96 | 6.5 | (skip) | — | ✗ inf |
| 8192² | 67.1M | 14821 ms | 120 | 123.51 | 4.5 | (skip) | — | ✗ inf |

## Single-net SSSP scaling (Phase 3.1, mask-based via SEG_BARRIER)

After replacing `INF_PROXY` with a true segmented scan (see `docs/architecture.md`
and the `sweep.py` module docstring), then hoisting the loop-invariant
`cumsum`/`cummax`/`seg_cw`/`seg_id_barrier` precompute out of the convergence
loop:

| Size | Cells | Sweep (MPS) | Iters | ms/iter | Mcells/s | Dijkstra (CPU) | Speedup | Status |
|---|---|---|---|---|---|---|---|---|
| 256² | 65K | 49 ms | 24 | 2.05 | 1.3 | 44 ms | 0.90× | ✓ |
| 512² | 262K | 43 ms | 24 | 1.78 | 6.1 | 184 ms | 4.30× | ✓ |
| **1024²** | **1.05M** | **94 ms** | **40** | **2.34** | **11.2** | **786 ms** | **8.39×** | ✓ |
| 2048² | 4.19M | 508 ms | 64 | 7.94 | 8.3 | 3352 ms | 6.60× | ✓ |
| **4096²** | **16.8M** | **3259 ms** | **104** | **31.34** | **5.1** | **(skip)** | — | **✓ NEW** |
| 8192² | 67.1M | 25402 ms | 192 | 132.30 | 2.6 | (skip) | — | ✗ inf |

### Three things this tells us about Phase 3.1

**1. The 4096² wall is gone; the new wall is between 4096² and 8192².**
With `SEG_BARRIER=2e4` and the polluted-mask threshold at `MAX_LEGIT_DISTANCE
= SEG_BARRIER/2 = 1e4`, the masked sweep correctly handles grids whose max
legit distance is under ~10,000. For unit weights that's `2*(N-1) < 10000`,
i.e., grids up to ~5000 per side. 4096² (max distance 8190) fits comfortably;
8192² (max distance 16384) overflows the threshold and gets falsely masked.
Bumping the wall further is mechanical: increase `SEG_BARRIER` and re-tune
the threshold, trading float32 ULP at intermediate values for max-distance
headroom.

**2. Per-iter cost is essentially Phase 1 parity after the precompute hoist.**
Compare `1024² ms/iter`: Phase 1 was 2.06; first-cut Phase 3.1 was 4.06
(~2× slowdown, expected — extra cumsum + cummax + two whers). The hoist
moves `cumsum(w_clean)`, `cumsum(obstacle_mask)`, `cummax(cw_at_obs)`, and
`seg_cw = cw - cw_recent_obs` out of the convergence loop (they depend on
`w` only, not `d`), so the per-iter inner work collapses to one cummin
plus a few arithmetic ops. Result: 1024² is now 2.34 ms/iter, 8.39×
speedup vs CPU — within ~12% of Phase 1's 9.51× while gaining correctness
past 2048². At 4096² the hoist halves per-iter cost (67.93 → 31.34 ms/iter).

**3. Iteration count grows roughly linearly with N (24/24/40/64/104/192).**
Same diameter-bounded behavior as Phase 1; the masked sweep has the same
convergence properties as the proxy-based one.

## Per-iter overhead progression

| Variant | 1024² ms/iter | Notes |
|---|---|---|
| Python-loop sweeps | ~6 | One MPS kernel per row/col = 1024 launches per sweep |
| Scan-based (cumsum+cummin) per-iter sync | 5.97 | `torch.equal` per iter forces CPU↔GPU pipeline flush |
| Scan-based + check_every=8 sync | 3.78 | Async pipelining across 8 iters between syncs |

37% per-iter improvement from removing the sync — and the gain widens with
grid size since each pipeline flush costs more on bigger tensors.

`torch.compile` adds another 10–20% on top (`inductor` slightly better than
`aot_eager` on MPS in 2.11) but is not yet wired into the production kernel.

## Multi-net sequential routing

`scripts/demo_multinet.py`, 50 random nets, 5% obstacles, seed 42:

| Grid | Routed | Per-routed-net | Total time |
|---|---|---|---|
| 256² | 23/50 | 24 ms | 0.55 s |
| 1024² | 23/50 | 145 ms | 3.34 s |

**Per-net cost** at 1024² is ~1.7× standalone single-net sweep (83 ms) — the
overhead is the per-net `sweep_sssp` invocation plus the path-marking loop.
Earlier this was 815 ms/net; the fix was running `backtrace` on a `.cpu()`
view to avoid per-cell `.item()` sync (cheap on Apple Silicon's unified
memory).

### Phase 2.1 endpoint reservation: a useful negative result

I expected pin reservation (mark all sources/sinks as obstacles up-front,
temporarily un-reserve a net's own pins while it routes) to recover the
27/50 failures by stopping early nets from running through later nets'
pins. **It did not.** Measurements at 256² with 50 nets, seed 42:

| nets | naive | reserved |
|---|---|---|
| 5 | 4/5 | 5/5 |
| 10 | 9/10 | 9/10 |
| 20 | 14/20 | 11/20 |
| 30 | 19/30 | 15/30 |
| 50 | 23/50 | 23/50 |
| 80 | 26/80 | 26/80 |

Reservation is a *correctness invariant* (no two distinct nets share a
wire — the naive version violated this by chance) but is **not** a
success-rate optimization on random workloads. Mechanism: reserving all
pins forces early nets to take longer paths around other-pin obstacles,
and those longer paths create more barriers that block later nets.

In-isolation control: every individual net is routable on the empty
grid (20/20). The failures are pure sequential-routing interference,
not anything about the kernel.

**Implication**: net ordering (Phase 2.2) and sweep-sharing /
ripup-and-reroute (Phase 2.3) are load-bearing for actual success rate,
not polish on top of reservation.

### Phase 2.2 net ordering: HPWL-ascending is the clean win

Three strategies on the same workload (256², 5% obstacles, seed 42,
reserve_pins=True):

| nets | identity | hpwl_asc | hpwl_desc |
|---|---|---|---|
| 10 | 9/10 | **10/10** | 9/10 |
| 20 | 11/20 | **18/20** | 12/20 |
| 30 | 15/30 | **21/30** | 10/30 |
| 50 | 23/50 | **32/50** | 15/50 |
| 80 | 26/80 | **41/80** | 25/80 |

HPWL-ascending consistently wins. At 80 nets, success rate goes from
26 → 41 (+58%). Per-routed-net wirelength also improves: at 50 nets
identity gives 198 wl/routed-net vs 168 for hpwl_asc.

HPWL-descending is a clean negative result — long nets routed first
dominate the grid and choke off the short ones. Don't use it.

The ordering change is ~30 lines (`gpu_pnr.ordering`) and adds zero
algorithmic complexity. This is the kind of low-cost lever that
sequential routing benefits from.

### Phase 2.3a sweep-sharing kernel: helps only in the small-grid regime

`sweep_sssp_multi(w, K-sources)` extends the scan-based sweep to a batch
dim, computing K shortest-path distance maps in one fused pass instead
of K sequential calls. Correctness test: agrees with per-source
`sweep_sssp` within float32 sum-order tolerance.

Throughput vs sequential, K = number of concurrent sources:

| Grid | K=1 | K=10 | K=50 |
|---|---|---|---|
| 256² | 1.11× | 1.41× | **3.10×** |
| 512² | 1.41× | 0.78× | 1.30× |
| 1024² | 0.37× | 0.91× | 0.97× |

Negative result at our target scale: at 1024² the per-source kernel is
already memory-bandwidth-bound (~20 ms/source for both sequential and
multi). Sharing the sweep doesn't reduce arithmetic and doesn't reduce
data movement — it just moves the same memory passes into wider tensors
that take proportionally longer.

Sweep-sharing only wins when the per-source kernel is launch-bound,
which happens at grids ≤ 256². The path to value at chip scale is
**tile decomposition** — split the 1024² grid into 4×4=16 tiles of
256² each, sweep-share within tile, reconcile at borders. That's a
Phase 3 architectural change, not a Phase 2 polish.

The kernel is committed because it is the right primitive for both
tile-decomposition and (later) batched per-net routing on smaller
sub-blocks extracted from a real fixture. But the planned Phase 2.3b
(`route_nets_batched` with conflict detection) is **deferred** —
without speedup at the target grid size, batched routing would just
add complexity without gain.

**The 23/50 success rate** is naive sequential routing on randomly-pre-chosen
endpoints. Smaller workloads succeed proportionally:

| Nets | Routed |
|---|---|
| 10 (seed 42) | 9/10 |
| 20 (seed 42) | 14/20 |
| 50 (seed 42) | 23/50 |
| 10 (seed 7) | 10/10 |

Phase 2 endpoint reservation should recover most of these.

## Surprises and learnings

- **Float32 + cumsum precision is the real ceiling** at this scale, not raw
  GPU throughput. The "obvious" `INF_PROXY = 1e10` was wrong by ~6 orders of
  magnitude; correct ceiling is `~4e6 / N` for unit-accurate distances.
- **Per-iter `torch.equal` sync is invisible until you measure it** — it
  doesn't show up as compute time but as pipeline-flush stalls.
- **`torch.compile` for MPS is cautiously useful** in PyTorch 2.11 — both
  `aot_eager` and `inductor` backends ran without crashing on the sweep
  kernel; gains modest (10–20%) because it can't fuse across PrimTorch ops
  that are already MPS-tuned.
- **Apple Silicon unified memory is genuinely a feature**, not just a
  marketing point. `.cpu()` is metadata-only — the backtrace fix wouldn't
  work nearly as cleanly on a discrete-GPU host.

# Phase 3.4 — multi-layer + via cost

`sweep_sssp_3d` and `route_nets_3d` extend the kernel and router to
operate on a `(L, H, W)` cost tensor with via transitions between
adjacent layers. Edge model: horizontal arrival pays `w[l, r, c]`; via
arrival pays only `via_cost`. Vias respect obstacles — neither the
kernel nor the reference Dijkstra allow a via to land on or chain
through a blocked cell.

## What changed in the inner loop

Per outer iteration, after the four intra-layer sweeps + `_mask_polluted`:

```
for l in 1..L-1:                      # upward
    d[l] = min(d[l], d[l-1] + via_cost)
    d[l] = where(obstacle[l], inf, d[l])
for l in L-2..0:                      # downward
    d[l] = min(d[l], d[l+1] + via_cost)
    d[l] = where(obstacle[l], inf, d[l])
```

A naive cumsum-cummin scan along `axis=0` (with `via_cost * arange(L)`
as the offset) was the first attempt — it's a single parallel pass per
direction. **It was incorrect under obstacles**: the scan adds
`via_cost * |Δl|` for any layer-pair regardless of intermediate
obstructions, allowing vias to "chain through" blocked layers. The
sequential per-layer form is correct, costs `2(L-1)` GPU min/where ops
per iter, and is dwarfed by the four intra-layer scans for typical
ASIC stacks (L=4-12).

## Negative finding worth flagging

The first thing I tried (cumsum-cummin layer scan) is the obvious
"keep everything as a parallel scan" move and it would have been an
attractive headline result. It happens to give the right numbers in
test_two_layers_zero_via_collapses_to_2d_min and test_high_via, then
fails the moment a multi-net router commits an obstacle on the
destination layer. That's a recurring shape: the parallel-scan
formulation tempts you with elegance and silently mis-models the very
thing the kernel exists to handle.

## Tests

16 new tests across `tests/test_sweep_3d.py` (9) and
`tests/test_router_3d.py` (7); full suite is 35/35 green.
The single-layer 3D matches existing 2D `sweep_sssp` exactly,
the cross-layer detour case is exercised both at the kernel level
(distances agree with 3D Dijkstra) and the router level (a route
forced to use a via to bypass a wall).

## Performance — not yet measured

No bench script for 3D yet. Per-iter cost should scale linearly with L
(the four intra-layer sweeps run vectorised over the layer dim, costing
the same as a single 2D sweep at `(L*H, W)` because cumsum/cummin only
parallelise along the scan axis). Iteration count grows as the diameter
of the 3D graph, not just the 2D one — vias contribute up to L steps
of latency on top of the H+W horizontal diameter. Adding a 3D bench is
left for the next session, gated on whether 3.2 (real fixture) needs
absolute numbers or only relative speedup vs CPU.

# Phase 3.2 — preferred-direction (per-axis costs)

`sweep_sssp_3d` and `route_nets_3d` now accept an optional `w_v`
per-axis cost tensor (see [ADR 0010](adr/0010-per-axis-cost-tensors.md)).
`scripts/spike_route_many_nets.py` replaces the old `m1_cost` knob
with `off_mult`, which builds gf180mcuD's per-layer preferred-direction
multipliers (M1=H, M2=V, M3=H, M4=V, M5=H) via `gpu_pnr.sweep.axis_costs`
and passes them as `(w_h, w_v)` to the router.

## TritonRoute comparison (off_mult sweep, N = 50 / 200 / 500)

| Sample | wire ratio vs TR | via ratio vs TR | M1 use | M2 use | M3+ use |
|--------|------------------|-----------------|--------|--------|---------|
| 50     | 1.08×            | 0.76×           | 50/50  | 50/50  | 0       |
| 200    | 1.26×            | 0.78×           | 200/200| 200/200| 0       |
| 500    | 1.36×            | 0.80×           | 500/500| 500/500| 0       |

Numbers are flat across `off_mult ∈ [2.0, 100.0]` — once the off-axis
penalty exceeds the via-stack overhead (≈ 2× `via_cost`), the router
commits to a fixed topology: via-stack at each pin from M1 to the
nearest preferred-axis layer, traverse there, via-stack back. M3+ is
never used at these in-guide 2-pin nets because using M3 instead of M1
costs 4 extra vias and saves nothing (M3-H and M1-H carry identical
unit cost; the spike's `via_cost=5` makes the swap a net loss for any
realistic segment length).

## Hypothesis comparison

The preferred-direction handoff predicted that this slice would close
the via-ratio toward 1.0× and flatten the 1.08× → 1.36× wire-ratio
drift. Measured numbers are **identical to the `m1_cost=10` row of
the spike's M1-as-pin-access experiment**. Per-net topology inspection
confirms the kernel is doing the *right thing* — a pure-V Hazard3 net
that previously ran illegal V wire on M1 now via-stacks M1 → M2(V) →
M1 — but the aggregate TR numbers tie because for these small in-guide
2-pin nets, "M1-cost penalty" and "per-layer preferred direction"
produce identical routing topology.

The hypothesis is falsified: preferred direction does not, by itself,
close the residual 20% via gap or the wire-ratio drift. The remaining
gap is now attributable to the other two unmodelled constraints flagged
in the spike doc: **per-via-pair cost asymmetry** (gf180mcuD's M1-M2
via differs from M3-M4 via in resistance and DRC) and **multi-pin
Steiner topology** (our route_nets_3d is point-to-point; ~11K of 24K
Hazard3 nets have 3+ pins, and TritonRoute's tree routing adds via
hops that our 2-pin sample doesn't see). See
[`plans/phase3-detailed-routing.md`](plans/phase3-detailed-routing.md)
deliverables 2 and 3.

## What this slice did buy

- The kernel API is now factored at the right layer: `(w, w_v)` is
  the general form. Option B (free per-cell anisotropy) and option A
  (factored from per-layer multipliers via `axis_costs`) use the same
  kernel.
- Per-net topology is now correct under preferred-direction: routes
  on M1-pref layers stay horizontal, V wire migrates to M2 via the
  via stack. This is a precondition for any further-realistic
  comparison.
- The remaining unmodelled constraints (per-via-pair, multi-pin) are
  now the only items left in WS3.2.

# Phase 3.2 — visualization and M1-penalty / pin-only investigation

After the preferred-direction landing, `scripts/render_routes.py`
provides per-layer chip-scale overlays of our routes vs TritonRoute's
geometry parsed from the post-DR DEF. Two modes: chip-scale binned
overlay (the same smallest-N sample as `spike_route_many_nets.py`,
one PNG per layer) and per-net per-layer detail view. The chip-scale
images surfaced two findings that the aggregate spike numbers had
hidden.

## Finding 1: PD alone leaves long M1 wires at full chip scale

The smallest-500 spike numbers tied with `m1_cost=10` (1.36×/0.80×).
We took that as evidence that preferred direction and M1-cost
penalty were equivalent at the topology level. The full 12,770-net
chip view falsified that:

| Layer | ours (binned) | TR (binned) | ours/TR | Notes |
|-------|---------------|-------------|---------|---|
| M1    | **100K wire** | 29K (anchors only) | 3.45× | PD model routes long M1-H wires |
| M2    | 220K          | 292K        | 0.75×   | V-pref backbone |
| M3    | 113K          | 273K        | **0.41×** | TR escapes to M3 far more |
| M4    | 31K           | 60K         | 0.52×   | long-haul V |
| M5    | 30K           | 32K         | 0.94×   | long-haul H, near parity |

The PD model makes M1-V expensive (10× off-axis) but leaves **M1-H
cheap** (preferred axis on a horizontal-pref layer). For long
horizontal nets, the router routinely runs M1-H wire — which the
gf180mcuD DRC forbids. At the small-N scale that was invisible
because per-net horizontal runs were short; at full chip those
M1-H runs accumulate to ~100K cells of illegal routing.

This corrects ADR 0010's "PD = m1_cost generalization" framing. The
two are equivalent at the kernel level (axis-aware costs) but
diverge at the modeling level: `m1_cost` penalised both axes; PD
only penalises off-preferred.

## Finding 2: `m1_penalty=10` recovers the missing constraint

Adding back the both-axes penalty (`m1_penalty`) on M1 specifically,
on top of the PD model, drops M1 wire dramatically and pushes the
displaced traffic onto M3:

| Layer | PD only | + m1_penalty=10 | Δ | TR baseline | new ours/TR |
|-------|---------|-----------------|---|-------------|-------------|
| M1 (wire) | 100K | **9.1K** | **−91%** | 29K | 0.31× |
| M2    | 220K | 167K | −24% | 292K | 0.57× |
| M3    | 113K | **198K** | **+75%** | 273K | **0.73×** ↑ from 0.41× |
| M4    | 31K  | 31K  | unchanged | 60K | 0.52× |
| M5    | 30K  | 30K  | unchanged | 32K | 0.94× |

M4 and M5 numbers are **literally identical** between the two runs
(same binned pixel counts). Those nets' decisions are driven by
long-haul distance, not the M1↔M3 cost balance; m1_penalty doesn't
move them.

## Finding 3: pin-only mode is equivalent on M3-in-guide nets

A strict mode (`--m1-pin-only`) marks every M1 cell as `inf` except
the source/sink pin coords and a 1-cell-radius landing pad. Forces
via-stacking off M1 unconditionally — the closest approximation of
gf180mcuD's no-M1-wire DRC rule.

On 100 nets that have M3 in their guide:

| Knob | M1 wire share | M3 cells |
|---|---|---|
| PD only | 41.4% | 9,823 |
| `m1_penalty=10` | 1.2% | 16,732 |
| `m1_pin_only`   | 1.8% | 16,632 |

Pin-only is ~1.8% (3-cell landing pads × 2 pins = 6 cells per net),
m1_penalty is ~1.2% (just the pin cell × 2). Effectively identical
TR-style topology.

On the smallest 100 nets (no M3 in their guide), all three modes
produce the *same* numbers: M1=1.6%, M2=98.4%. The router was
already using M1 only at pin endpoints because PD-on-M2-V was
cheaper than long M1-H for those short routes. The M1-wire problem
was specifically a **middle-cohort** problem (nets with M3 in guide
but our PD model picking M1 over M3).

## Finding 4: there's a structural cohort our model can't reach

Of the 12,770 two-pin nets, ~6,000 have **no M3 in their guide** —
GR allocated them to M1+M2 only, deciding the route is short enough
not to need M3. TritonRoute breaks out of the guide for these (M1
wire is DRC-illegal, so TR escapes to M2+ regardless of guide). Our
spike builds per-net mini-grids with off-guide cells as `inf`, so
the router *can't* escape — it's a hard constraint, not a soft
preference.

This is the residual gap on M3 (still ~27% vs TR) after the
m1_penalty fix. Two ways to close it:

1. **Soft guide**: make off-guide cells a finite penalty instead of
   `inf`. Quick fix, but it's an interim hack.
2. **Chip-scale grid (WS3.3, tile decomposition)**: drop per-net
   mini-grids entirely. Once we route on a single global cost
   tensor, the GR allocation becomes naturally advisory — any net
   can escape into cells GR didn't assign to it. Subsumes the
   soft-guide approach.

The latter is on the Phase 3 plan as the next architectural slice
after the WS3.2 deliverables. Choosing it over soft-guide avoids
investing in an interim mechanism we'd throw away.

## Implementation conclusion: M1-as-pin-only should be a PDK rule

The M1-penalty knob exists in two forms now: `m1_penalty` (soft cost
multiplier) and `m1_pin_only` (hard `inf` mask). Both are
experiment-time tunables. The right encoding is to lift the rule
into the cost-grid construction itself: `build_grid` should take a
PDK descriptor naming which layers are pin-access-only, and emit
the `inf` mask automatically. Costs would then be cost weights;
pin-only-ness would be a structural constraint, mirroring how real
DR tools split technology rules from heuristic weights. Deferred
until the cost-grid construction is refactored more broadly
(probably during WS3.3). *Shipped 2026-05-11, commit `907f632`,
[ADR 0011](adr/0011-pdk-rules-as-structural-constraints.md).*

# Phase 3.2 — multi-pin nets

`route_multipin_nets_3d` implements incremental tree growth on top of
the per-axis 3D SSSP kernel (commit `a7aa2d5`,
[plan deliverable 4](plans/phase3-detailed-routing.md#ws32--real-fixture-integration-hazard3-on-gf180mcud)).
For an N-pin net the router seeds a tree at `pins[0]`, then iteratively
runs SSSP from the current tree (every tree cell at distance 0 via the
new `extra_sources` kwarg on `sweep_sssp_3d`), picks the unrouted pin
with smallest distance, and backtraces to attach it. The kernel-level
multi-source semantic is the canonical one: SSSP from `{s1, s2, ...}`
equals the element-wise min of SSSP from each source individually,
verified against Dijkstra in
[`tests/test_sweep_3d.py`](../tests/test_sweep_3d.py).

## TritonRoute comparison (smallest 500 multi-pin nets)

| Metric | ours | TR | ratio |
|---|---|---|---|
| routed | 500 / 500 (100%) | — | — |
| wirelength | 130,725 cells | 139,583 | **0.94×** |
| vias | 1,671 | 2,773 | **0.60×** |
| avg per-net time | 159 ms | — | — |

Two surprising findings:

1. **We use *less* wire than TR (0.94×).** Sequential nearest-pin
   attachment produces tight trees on small per-net mini-grids. TR's
   Steiner topology is more sprawling, partly because it can detour
   off-guide and partly because it optimises for a different cost
   model (DRC + RC, not just Manhattan length).
2. **Our via count is 60% of TR's.** That's the per-via-pair-cost
   story showing up unambiguously: TR uses Via2/Via3 transitions to
   escape to upper layers that our scalar `via_cost=5` makes uniformly
   expensive. With per-pair costs (next deliverable) reflecting
   gf180mcuD's real M1↔M2 vs M2↔M3 asymmetry, we'd expect our via
   ratio to climb toward 1.0×.

## Layer occupancy

Of the smallest 500 multi-pin nets, **85 (17%) used Metal3** — the
first slice in any of our spike samples to show meaningful M3 use by
our router. Multi-pin nets are larger and their guides include M3,
unlocking the escape that the smaller 2-pin nets couldn't reach.
Confirms that the per-net guide-locking constraint is the actual
limiter, not the cost model.

## Implementation notes

- The kernel changes are additive: `sweep_sssp_3d(..., extra_sources=())`
  and `backtrace_3d(..., extra_sources=())` default to the empty
  sequence, preserving existing call sites and tests.
- `MultiPin3DResult` exposes `paths: list[list[cell]]` (one per
  attachment edge), `cells: set[cell]` (dedup'd footprint), and
  `length: int` (cells − pins).
- Per-net pin reservation works the same way as the 2-pin case: all
  nets' pins are inf at the start; each net's pins are restored when
  it routes; tree cells become inf after the net completes.
- Multi-pin nets cost ~3× the per-net time of 2-pin nets because
  each tree-attachment edge runs its own SSSP. For an N-pin net the
  total work is N-1 sweeps.

# Phase 3.2 — sweep-sharing throughput (Tier A)

Before committing to WS3.3 (tile decomposition / chip-scale routing),
we needed to verify the architectural premise: that multi-source SSSP
(`sweep_sssp_3d_multi`, commit `e5dd5be`) actually amortises compute
across K sources at chip-scale-relevant tile sizes. ADR 0008's earlier
2D measurement said it did, at 256²; the question was whether the 3D
kernel preserves that scaling. See
[`spikes/tier-a-sweep-sharing-throughput.md`](spikes/tier-a-sweep-sharing-throughput.md).

## 2D sweep-sharing refresh (peak speedup vs K sequential calls)

| Tile size | K=1 | K=10 | K=25 | K=50 | K=100 | K=200 |
|---|---|---|---|---|---|---|
| 256² | 1.5× | 2.1× | 5.6× | 4.0× | **8.9×** | **9.1×** |
| 512² | 1.2× | 1.0× | 2.9× | 2.8× | 2.2× | 2.3× |
| 1024² | 1.5× | 1.2× | 1.0× | 0.9× | 0.8× | 0.8× |

ADR 0008's pattern is confirmed and the 256² peak is now even higher
than originally measured (9× vs the prior 3.1×). MPS perf has improved.

## 3D sweep-sharing (the regime tile decomposition will actually use)

3D 256² × 5 layers, via_cost=5.0:

| K | seq ms | multi ms | speedup | ms/source seq | ms/source multi |
|---|---|---|---|---|---|
| 1 | 225 | 315 | 0.72× | 225 | 315 |
| 10 | 641 | 313 | 2.05× | 64 | 31 |
| 25 | 2,079 | 531 | 3.92× | 83 | 21 |
| 50 | 4,227 | 1,264 | 3.35× | 85 | 25 |
| **100** | **12,542** | **3,101** | **4.05×** | 125 | **31** |

3D 512² × 5 layers: peaks at 1.5× at K=10-25, then **collapses to
0.19× at K=100** as MPS hits memory-pressure on the
(K, L, H, W) = (100, 5, 512, 512) × float32 = 524 MB distance tensor.

3D 1024² × 5 layers: timed out at 7+ minutes on the K=1 case (a single
SSSP run at this size is already memory-bandwidth-bound; sweep-sharing
won't help).

## Implications for wafer.space-scale routing

- 256² × L=5 tiles are the throughput sweet spot.
  ~5,000 tiles cover wafer.space (15K × 21K cells).
- At 31 ms/source via K=100 batching, per-tile throughput is **100
  nets in ~3 s**.
- Conservative whole-chip estimate on MPS:
  4,980 tiles × 0.5 s (K=20 avg) = **~40 min** for routing the
  signal nets at wafer.space scale.
- TR baseline estimate at wafer.space scale: ~1-3 hours.
- CUDA scale-up (5-15× memory bandwidth): **~3-8 min** total on
  modern data-center GPUs. The bull case.

The numbers do not yet account for halo reconciliation, rip-up, or
DRC iteration — those will erode the gap. But the *primary mechanism*
(GPU SSSP amortising across many sources within a tile) is now
empirically verified to deliver the predicted scaling.

## Tile-size choice

256² × L=5 is the architectural target for Phase 3.3. The 512² result
in 3D specifically warns against picking larger tiles: the K=100
multi-source kernel regresses by ~5× at 512² and is dramatically
slower than the 256² version even per-source. The win comes from
*many small tiles in parallel*, not from *fewer larger tiles*.

## Caveats

- The wafer.space numbers above are extrapolated, not measured. A
  validation routing run on a real wafer.space LibreLane fixture is
  the next obvious experiment.
- Tile decomposition halo handling is not yet designed; reconciliation
  costs at tile boundaries could eat a meaningful fraction of the
  gain.
- TR's rip-up-and-reroute iteration is unaccounted for; if we need
  similar iteration for quality, total wall-clock grows linearly with
  iteration count.
- The 0.19× collapse at 3D 512² K=100 is MPS-specific — CUDA's larger
  memory bandwidth and explicit memory management would likely make
  that regime productive too, widening the architectural flexibility.
