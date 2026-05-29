# Spike — Slot-scale GPU parallelism for guide-constrained sweep

**Status:** Open (2026-05-28); load-bearing hypothesis **resolved YES
2026-05-29**. Analytical projection from the Hazard3 track-pitch
measurement — **not** an empirical run. The load-bearing claim (GPU
batching can amortise per-net launch overhead) was the open bet; it is now
confirmed empirically — batched small-grid sweep is 2.46–4.05× faster than
sequential on MPS. See
[batched-small-grid-sweep](batched-small-grid-sweep.md). The slot-scale
*area* projection below remains analytical.

## Question

At wafer.space 1×1 slot scale (3.05 × 4.24 mm = 12.93 mm²), how much
parallel work does guide-constrained sweep expose, and what actually
bounds GPU utilisation — work, memory, occupancy, or per-call overhead?

## Context

Two findings reset the throughput model for WS3.3:

1. **Guide-constrained sweep** ([ADR 0012](../adr/0012-tile-decomposition.md)
   Amendment 1): route each net on a sub-grid sized to its guides, not a
   full tile.
2. **Track-pitch grid** (ADR 0012 Amendment 3): the cost tensor should be
   sampled at the real routing-track pitch (1120 DBU for gf180mcuD
   M1–M4), not the 200 DBU grid the prototype used. At track pitch the
   median Hazard3 net is 38×38×3 ≈ 4,332 cells, matching Amendment 1's
   estimate (see [`../results.md`](../results.md) Phase 3.3).

This spike asks the next question: does that per-net economy scale to a
full slot, and is there enough parallelism to keep a GPU busy?

## Hardware anchor

Measurements are anchored to the development machine:

- **Apple M4 Pro, 20-core GPU** — 2,560 FP32 ALUs, ~9 TFLOPS peak,
  **~273 GB/s** memory bandwidth, **24 GB** unified memory (Metal 4).

CUDA (production target, [ADR 0001](../adr/0001-pytorch-mps-host.md)) has
~10× the bandwidth and ~50–100× the ALU count; the conclusions only get
stronger there.

## Method — area-scale from Hazard3

The Hazard3 fixture is a ~100%-packed core (24k cells ≈ 0.72 mm² of cell
area in a 0.76 mm² die), so area-scaling to a high-utilisation slot is
defensible. All per-net figures are the **track-pitch (1120 DBU)**
measurement.

```
slot     3.05 × 4.24 mm   = 12.93 mm²
Hazard3  864.4 × 882.3 µm =  0.763 mm²   (DEF DIEAREA)
area ratio                = 16.95×
```

| | Hazard3 (measured) | Slot (× 16.95) |
|---|---:|---:|
| Routable nets (2–20 pin) | 20,524 | **~348,000** |
| Total nets | 24,123 | ~409,000 |
| Median cells/net | 4,332 | (same) |
| Mean cells/net | 27,590 | (same) |
| Total sweep footprint | 0.57 B cells | **~9.6 B cells** |

> At the old 200 DBU grid the slot footprint would be ~31× larger
> (~300 B cells). The track-pitch fix is what makes the slot tractable
> at all.

## Three nested levels of parallelism

### L1 — within one net's sweep (data-parallel cells)

A median sub-grid is 38×38×3 = 4,332 cells. Each axis sweep
([ADR 0002](../adr/0002-scan-based-sweeps.md)) is a `cumsum`+`cummin`
over the grid; an H-sweep runs `layers × rows = 38 × 3 = 114` independent
scan lanes of length 38. So ~4,000-way data parallelism per net.

**Limit surfaced:** ~4k cells is *below* the GPU's comfort zone. 2,560
ALUs want ~10⁵ cells in flight to hide memory latency. A single small
net **starves** the GPU — which is the entire motivation for L2.

### L2 — across nets (batched independent sub-grids) — the lever

Batch K independent nets into one kernel call (Amendment 1 §3). Three
candidate bounds:

- **Memory:** ~70 KB working set per net (bucketed, unpadded: `w_h`,
  `w_v`, `d`, scratch ≈ 4 tensors × 4,332 cells × 4 B). 16 GB of GPU
  budget holds **~230,000 nets at once** — the whole slot's median nets
  fit in unified memory simultaneously.
- **Occupancy:** ~100–1,000 median nets (0.4M–4M cells) saturate the
  2,560 ALUs and hide latency.
- **Dependency:** nets sharing cells conflict via the shared `w_cur`
  obstacle tensor; spatially disjoint nets batch freely. Across 348k
  nets over 12.9 mm², ~thousands are independent at any instant
  (DRT-style region independence).

The binding bound is occupancy/dependency (~10²–10³ nets), reached **far**
before memory runs out.

### L3 — total work for the slot

```
~348k nets × ~27.6k mean cells ≈ 9.6 × 10⁹ cell-updates per sweep pass
  × ~tens of sweep iterations × ~5 rip-up rounds ≈ 10¹²–10¹³ cell-ops
```

## The binding constraint

The available parallel work (~9.6 B cells, ~348k independent nets) is
**4–5 orders of magnitude more than needed to saturate a 2,560-ALU GPU**
(~10⁵ cells). The slot is never *work-starved*. What limits it:

| Limit | Slot value | Verdict |
|---|---|---|
| Memory capacity | whole slot working set fits in 24 GB | not binding |
| ALU occupancy | saturated by ~hundreds of batched nets | easy |
| **Per-net launch / sync overhead** | ~348k tiny kernels if serial | **THE bottleneck** |
| Memory bandwidth (273 GB/s) | the true floor once amortised | the goal |

The hard part is **packing many tiny sweeps into few kernel launches** so
348k nets don't each pay a fixed launch + convergence-sync cost
([ADR 0003](../adr/0003-async-convergence-check.md) already amortises the
sync *within* a sweep; this is the *across-sweeps* analogue). That is
precisely the unproven Amendment 1 §3 hypothesis. Note the contrast with
[Tier B](tier-b-envelope-throughput.md): K-batching on *one big* grid is
dead; batching *many small independent* grids is the open bet, and
slot-scale net counts are what make it worth proving.

## Throughput projection (caveated)

```
single-stream linear (feed the GPU one net at a time):
   31.1 s (Hazard3, M4 Pro, track pitch) × 16.95 ≈ 527 s ≈ 8.8 min / pass
   × ~5 rip-up rounds                              ≈ 44 min  full route
```

Caveats on the single-stream number:
- **Optimistic on cells, pessimistic on parallelism.** The linear model
  (ms ∝ cells, anchored at 18 ms / 327,680-cell net) *under-counts* fixed
  per-net overhead: a 0.24 ms budget cannot absorb a ~0.5 ms kernel
  launch + sync. Real single-stream is likely **worse** than 8.8 min.
- **L2 batching is unmodelled here** — it is the win that claws the
  overhead back, plausibly toward the bandwidth floor (order tens of
  seconds/pass on M4 Pro), but **unmeasured**.

```
CUDA (production): ~10× bandwidth (~3 TB/s), ~50–100× ALUs
   → bandwidth floor ~seconds/pass; batching becomes essential just to
     feed that many ALUs.
```

## Conclusion

For a wafer.space 1×1 slot: **~350k routable nets / ~9.6 B grid cells** of
work at track pitch. The GPU has *far* more parallel work than it can
use — ~348k independent sub-grid sweeps where only ~hundreds saturate the
M4 Pro. The open question is **not** "is there enough to parallelise"
(emphatically yes) but **"can we batch the tiny sweeps to amortise launch
overhead and reach the bandwidth ceiling."**

**Next empirical step — DONE (2026-05-29):** the batched small-grid sweep
kernel was prototyped and measured — K independent sub-grids vs K sequential
single-grid sweeps at track pitch. **Result: 2.46–4.05× faster on MPS**
(GPU-specific; CPU loses). The load-bearing hypothesis is resolved YES; see
[batched-small-grid-sweep](batched-small-grid-sweep.md).

## What this spike does NOT cover

- **Slot-scale batched throughput.** The batched kernel is now built and
  measured at *Hazard3* scale ([batched-small-grid-sweep](batched-small-grid-sweep.md));
  the *slot*-area projection above (×16.95) remains analytical.
- **Pin access on a track grid.** Off-track pins need snapping or a
  locally-finer region (ADR 0012 Amendment 3 open question).
- **Rip-up / conflict convergence at slot scale.** The ~5-round figure is
  borrowed from DRT on dualcore ([GPU-vs-DRT spike](gpu-vs-drt-throughput.md)),
  not measured for our router.
- **DRC.** As elsewhere, we don't check DRC during routing; quality
  comparison is tracked in `results.md`.
- **Real slot utilisation.** Net count assumes ~packed-core density;
  a sparser floorplan scales down linearly.

## References

- [ADR 0012](../adr/0012-tile-decomposition.md) Amendments 1–3 —
  guide-constrained sweep, bbox finding, track-pitch finding.
- [`../results.md`](../results.md) Phase 3.3 — guide-region size
  distribution at 200 vs 1120 DBU.
- [GPU vs DRT throughput spike](gpu-vs-drt-throughput.md) — DRT baseline
  and the search-space framing.
- [Tier B spike](tier-b-envelope-throughput.md) — why K-batching on one
  big grid is dead (and why small-grid batching is a *different* bet).
- [ADR 0001](../adr/0001-pytorch-mps-host.md) — MPS host; CUDA is the
  production scaling target.
