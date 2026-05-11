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
