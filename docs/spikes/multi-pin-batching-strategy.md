# Spike — Multi-pin batching strategy for Slice 3

**Status:** Resolved (2026-06-01). Settles the open question in
[`../plans/ws33-tile-router-implementation.md`](../plans/ws33-tile-router-implementation.md)
Slice 3. **Verdict: option (b) round-batching; option (c) "2-pin-only" is
ruled out — it would batch only 8.8% of the sweep work.**

## Question

The batched small-grid kernel
([batched-small-grid-sweep](batched-small-grid-sweep.md)) is single-source
per net. Multi-pin nets need `pin_count − 1` attachment sweeps via the
existing incremental tree-growth. Three candidate strategies for Slice 3:

- **(a)** batch *only* the first attachment sweep; subsequent attachments run
  sequentially.
- **(b)** batch round-*r* attachment sweeps across all nets still growing —
  variable batch size per round.
- **(c)** batch 2-pin nets only; route ≥3-pin nets sequentially via the
  existing path.

(c) is the cheapest. The decision hinges on **how much of the routing work
lives in ≥3-pin nets** — if 2-pin nets dominate work, (c) suffices; if not,
(b) is needed.

## Method

`scripts/measure_pin_count_distribution.py` walks the Hazard3 fixture at the
track pitch (1120 DBU), applies the same routable + in-cap filter as the
sweep prototypes (2–20 M1 pins, has `guide_region`, ≤256 axis, pins inside
region), and histograms three metrics per pin count:

- **nets**: count of nets in that bucket.
- **cells**: total `region.cell_count` (sub-grid size) in that bucket.
- **work = cells × (pin_count − 1)**: total sweep-cell-work, since each
  multi-pin net costs `pin_count − 1` attachment sweeps on the same sub-grid.

Pure measurement, no routing. ~5 s wall-clock.

## Result

Hazard3, track pitch, in-cap population = 19,230 nets (93.7% of the 20,524
routable):

| pins | nets | %nets | cum% | cells (Σ) | %cells | %work | cum%work |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 12,572 | 65.4% | 65.4% | 60.8M | 35.6% | **8.8%** | 8.8% |
| 3 | 4,032 | 21.0% | 86.3% | 32.4M | 19.0% | 9.3% | 18.1% |
| 4 | 1,324 | 6.9% | 93.2% | 20.7M | 12.1% | 9.0% | 27.0% |
| 5 | 377 | 2.0% | 95.2% | 8.6M | 5.0% | 5.0% | 32.0% |
| 6–9 | 405 | 2.1% | 97.3% | 17.2M | 10.1% | 15.6% | 47.5% |
| 10–13 | 393 | 2.0% | 99.3% | 21.7M | 12.7% | 33.0% | 80.5% |
| 14–20 | 127 | 0.7% | 100.0% | 9.3M | 5.5% | 19.5% | 100.0% |

(full table in the script output and `docs/results.md` if folded later.)

### Headlines

- **2-pin share of in-cap nets: 65.4%**
- **2-pin share of sweep-work: 8.8%**
- **≥3-pin share of in-cap nets: 34.6%**
- **≥3-pin share of sweep-work: 91.2%**

Two compounding effects make work concentrate in multi-pin nets:
1. ≥3-pin nets have larger guide regions (they span more area).
2. Each one costs `pin_count − 1` sweeps, so a 12-pin net costs 11 sweeps.

The distribution is unintuitive: a single 12-pin net's work ≈ all 3-pin
nets combined.

## Decision

**Slice 3 implements option (b): round-batching.** Each attachment round
batches the nets still growing; their sub-grids stay the same, their
sources change (the round's tree-seed per net). Cost:

- Extend `sweep_sssp_3d_batched` to support `extra_sources` per net (the
  existing single-source kernel already takes a per-call `extra_sources`
  sequence; the batched form needs a per-batch-slice list). This is the
  one kernel change.
- Per round, group still-attaching nets and dispatch one batched sweep;
  backtrace each slice against its sub-grid; commit; advance round.
- A net's attachments may convergence-mask out earlier than the slowest in
  its round — same "slowest bounds the batch" trade-off the
  [batched-small-grid-sweep spike](batched-small-grid-sweep.md) already
  measured, with the same answer (still wins on MPS).

**Option (c) is ruled out.** End-to-end batched speedup under (c) would be
bounded by ~1.1× (8.8% of work × ~4× kernel speedup + 91.2% unchanged),
losing essentially all of Amendment 4's win.

**Option (a) is dominated by (b).** (a) does round-1 only; (b) does
round-1, round-2, … as long as enough nets remain. Batch sizes shrink each
round (19k → 6.7k → 2.6k → 1.3k → …), but stay far above the GPU's
occupancy floor (~10²–10³ per the slot-scale spike) for the rounds that
carry meaningful work.

## What this does NOT cover

- **Convergence-masking** — letting a net drop out of the batch once its
  distances converge, so faster nets don't wait for the slowest. Already
  named as a deferred lever (ADR 0012 Amendment 4); orthogonal to the
  round-batching decision.
- **Diminishing returns past round 4–5.** When the batch shrinks below the
  GPU's useful occupancy, falling back to sequential for the tail might
  win — measure in Slice 3.
- **Actual end-to-end ms/net at Hazard3 scale** — that lands when Slice 3
  is built and run.

## References

- [`../plans/ws33-tile-router-implementation.md`](../plans/ws33-tile-router-implementation.md)
  Slice 3 — strategy choice updated to (b) per this spike.
- [`batched-small-grid-sweep.md`](batched-small-grid-sweep.md) — the kernel
  that round-batching dispatches into.
- [ADR 0012](../adr/0012-tile-decomposition.md) Amendments 1 & 4 —
  guide-constrained model + batched-kernel decision this builds on.
- `scripts/measure_pin_count_distribution.py` — the measurement.
