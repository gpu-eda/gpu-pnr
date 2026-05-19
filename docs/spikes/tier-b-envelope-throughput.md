# Spike — `sweep_sssp_3d_multi` throughput at envelope sizes > 256²

**Status:** Resolved (2026-05-19).

## Question

Slice 2 of the WS3.3 tile router plan measured a 42.1% multi-tile-spanning
fraction at halo=32 on Hazard3 — well past ADR 0012's 25% walk-back gate.
Three amendment paths emerged ([results.md §Phase 3.3](../results.md)):

- **B.** Uniform halo widening (96 / 128 / 192 / 256).
- **C.** Adaptive per-net envelope sized to bbox + margin.
- **A.** Net-splitting at tile boundaries with halo handshake.

Options B and C depend on the same kernel question: **does
`sweep_sssp_3d_multi` survive (and how does it scale) at envelope > 256²?**
[Tier A](tier-a-sweep-sharing-throughput.md) measured 256² (peak K=100 → 4.05×) and 512²
(K=100 → 0.19× collapse) but never benched the candidate halo settings
(96 / 128) or finer K-sweeps at 512²+.

## Approach

Re-run `scripts/bench_sweep_sharing.py --mode 3d --layers 5 --via-cost 5.0`
at `size ∈ {256, 320, 448, 512, 768}` corresponding to envelopes for
halo ∈ {0, 32, 96, 128, 256} on top of a 256² owned region, with K-sweep
`{1, 10, 25, 50, 100}` (K=50/100 capped at envelope=768 to avoid the
~1.5 GB distance tensor). Uniform-cost grid as in Tier A — kernel
throughput characterization, not realistic-workload measurement.

## Findings

### Per-envelope K-sweep (ms/source via the multi kernel)

| envelope | K=1 | K=10 | K=25 | K=50 | K=100 |
|---:|---:|---:|---:|---:|---:|
| 256² | 109.20 | 21.33 | **13.27** | 20.57 | 15.73 |
| 320² | 174.38 | 45.66 | **19.94** | 24.07 | 26.94 |
| 448² | 80.11 | 41.76 | 43.74 | 46.73 | 52.62 |
| 512² | 75.29 | 49.66 | 49.98 | 52.71 | 65.57 |
| 768² | 138.69 | 114.52 | 116.77 | — | — |

Bold = K-peak (minimum ms/source) for that envelope.

### Sequential baseline (same kernel call repeated K times, ms/source)

| envelope | seq ms/source (flat across K) |
|---:|---:|
| 256² | ~19.3 |
| 320² | ~18.7 |
| 448² | ~29.5 |
| 512² | ~38.1 |
| 768² | ~90.5 |

### Multi vs sequential speedup, by envelope, at best K

| envelope | best multi K | multi ms/src | seq ms/src | speedup |
|---:|---:|---:|---:|---:|
| 256² | K=25 | 13.27 | 19.37 | **1.46×** |
| 320² | K=25 | 19.94 | 18.71 | 0.94× |
| 448² | K=10 | 41.76 | 29.50 | **0.71× (loses)** |
| 512² | K=10 | 49.66 | 38.32 | **0.77× (loses)** |
| 768² | K=10 | 114.52 | 90.47 | **0.79× (loses)** |

## Headline

**Two big findings, both load-bearing for the ADR 0012 amendment:**

1. **K-batching only wins at envelope=256².** At every other envelope
   tested, sequential routing is faster per source than the K-batched
   multi kernel. The "256² is the throughput sweet spot" finding from
   Tier A holds — but the moment the envelope grows past 256² (even to
   320², the current design's halo=32 envelope), K-batching becomes a
   net loss.

2. **The 256² K=100 4.05× speedup from Tier A is gone.** Today the
   peak speedup at 256² is 1.46× at K=25 (multi 13 ms/source vs
   sequential 19 ms/source). The `72de221` per-pair `via_cost` commit
   that landed after Tier A is the only sweep.py change in the
   interval — likely added per-iter overhead to the multi kernel that
   doesn't amortize at high K. Sequential routing got faster too
   (19 ms vs Tier A's 125 ms), narrowing the multi advantage.

### Subordinate findings

- **No memory cliff observed.** At envelope=512² × K=100 the multi
  kernel ran (526 MB distance tensor) but ran slower than sequential —
  not the "catastrophic regression" Tier A reported. Either MPS perf
  improved at large tensors, or Tier A's collapse was a different
  failure mode (e.g., thrashing) that current perf no longer hits.
  Envelope=768² × K=100 deliberately not measured (~1.2 GB distance
  tensor; expected to be in the same losing regime).
- **Sequential ms/source roughly scales with envelope area past 256²:**
  256→19, 320→19, 448→30, 512→38, 768→90. The 256↔320 plateau
  suggests sub-linear scaling at small envelopes.
- **K-peak shifts down as envelope grows:** K=25 at 256², K=10 (or 1)
  at 448²+. The K-knee gets sharper and earlier; the K-batch payoff
  is concentrated in a narrower regime than Tier A's "K=25-100
  productive range".

## Implications for the ADR 0012 amendment

The throughput-model premise of the original ADR 0012 — "per-tile
K=100 batching turns each tile into a ~31 ms/net unit of work" — is
now obsolete. **K-batching no longer drives Phase 3.3's wall-clock
case.** Sequential routing per net is the right design parameter.

The amendment can rule out parts of each option from this data:

- **Path B (uniform halo widening).** Wall-clock-viable at any halo
  tested. Sequential per-net cost: ~30 ms at halo=96, ~38 ms at
  halo=128, ~90 ms at halo=256. K-batching is dead past 256² so
  this is sequential-only by necessity. Concrete prediction:
  Hazard3 at halo=128 ≈ 4399 spanning + 16125 per-tile-routable;
  per-tile sequential at envelope=512²: 16125 × 38 ms ≈ 10 min, plus
  ~21.4% coarsened-pass workload and reconciliation. Order-of-magnitude
  matches ADR 0012's 22-44 min estimate.
- **Path C (adaptive envelope).** Lets the 256² fast path (19 ms/source)
  carry the 57.9% of nets that fit one tile's standard envelope, with
  wider envelopes (sequential) for the medium spans. Best wall-clock
  by routing-cost arithmetic, but the per-net kernel-config switching
  needs design attention — not free.
- **Path A (net-splitting).** Was previously sold on protecting a
  ~4× K-batch fast path. That fast path is gone — multi at 256² is
  only 1.46× sequential. The architectural value of net-splitting is
  reduced; the residual benefit is avoiding the coarsened-pass quality
  hit, not throughput.

**Recommended amendment direction (architect to confirm):** Path C
or a hybrid B+C. C uses the 256² sequential regime for short nets and
larger sequential envelopes only as needed. B at halo=128 is simpler
and viable. The K-batching machinery (planned for Slice 4) should
probably be dropped from the amendment — sequential is competitive
and far simpler.

## What the spike does NOT cover

- **Non-uniform cost grids.** Bench uses `w = ones`. Real workloads
  have obstacle patterns that affect iteration count. Tier A noted
  obstacle patterns affect iters more than per-iter cost, so the
  relative ordering should hold.
- **Real backtrace overhead.** Bench measures the sweep kernel only;
  per-net backtrace + commit-to-w_cur cost is excluded. Backtrace is
  typically ~10% of sweep cost per the prototype.
- **CUDA.** Apple Silicon MPS only. CUDA's K-batch behavior may be
  different — likely better given memory bandwidth, but unmeasured.

## Logs

- `/tmp/claude/tier-b-env256.log`
- `/tmp/claude/tier-b-env320.log`
- `/tmp/claude/tier-b-env448.log`
- `/tmp/claude/tier-b-env512.log`
- `/tmp/claude/tier-b-env768.log`

## Links

- [Tier A](tier-a-sweep-sharing-throughput.md) — the original
  sweep-sharing characterization (2026-05-12). This spike updates it.
- [`../results.md`](../results.md) §Phase 3.3 — the spanning-fraction
  measurement that motivated this spike.
- [ADR 0012](../adr/0012-tile-decomposition.md) — design that this
  spike's findings will inform amending.
- [`../plans/ws33-tile-router-implementation.md`](../plans/ws33-tile-router-implementation.md)
  — the plan whose Slice 4 (K=100 batching) is now in question.
