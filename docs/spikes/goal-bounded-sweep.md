# Spike — goal-bounded sweep

**Status:** Resolved — **modest win, not the drt-gap closer** (2026-06-03).
Bounding each net's sweep region to its pin bbox (instead of the full GR guide)
cuts cells ~1.6× and time ~1.17× with **zero routability or wirelength loss** —
but it does not close the ~3.8× gap to drt, and the *why* is the useful finding:
**bounding the region ≠ goal-directing the search.** The real lever is
goal-biased *expansion* (A*-style f-band), the harder follow-on.

## Question

`gpu-astar-evaluation.md` ruled out GPU A* (wrong parallelism shape) but kept
its idea: A*'s goal-direction is drt's search-space advantage. The cheapest
version that preserves our batchable regular sweep is to **shrink the region** —
route each net on its pin-bbox corridor (+ margin) instead of the full GR-guide
bbox. Does that cut search-space and time while preserving routability?

## Method

`scripts/goal_bounded_sweep_prototype.py`. Each in-cap Hazard3 net (track pitch)
is routed two ways on its **own** sub-grid (independent — the only variable is
region size): full `guide_region` vs **goal-bounded** (full layer stack — vias
need it — with row/col tightened to `net_bbox(pins) ± margin`, a subset of the
guide since pins lie inside it). Reports cells, ms/net, routability, wirelength.
Sample 500, seed 0, CPU (single-net routing is overhead-bound on MPS; the
full-vs-bounded *ratio* is device-independent and cleaner on CPU).

## Result

| margin 4, sample 500 | median | mean | p90 |
|---|---:|---:|---:|
| full cells | 4,332 | 10,105 | 19,992 |
| bounded cells | 1,728 | 6,443 | 11,759 |
| full ms/net | 2.50 | 20.04 | 15.79 |
| bounded ms/net | 2.07 | 17.16 | 11.19 |

- **Cell reduction: 1.57× mean (2.5× median); speedup: 1.17×.**
- **Routability lost by bounding: 0/500.** Wirelength bounded/full: **1.000×**
  (median, mean, p90) — bounding forces no detours, no quality change.
- Tighter margin barely helps: margin 2 → 1.60× cells, still 1.17× speedup, 0
  loss. So **margin is not the lever** — the guide-vs-bbox *extent* is, and it's
  modest because GR guides already roughly hug the pin bbox (median only 2.5×).

## The finding: region-size ≠ expansion-strategy

The speedup (1.17×) is far below the cell reduction (1.57×). Two reasons, and
the second is the important one:

1. **Sweep time is diameter-bound, not area-bound.** Bellman-Ford/sweep
   converges in ~grid-diameter iterations. Bounding shrinks *area* more than
   *diameter* (pins span the box), so fewer cells per iteration but ~the same
   iteration count → speedup < cell-reduction.
2. **A bounded box still runs *full* SSSP inside it.** drt's A* advantage is not
   "a smaller box" — it's **goal-directed expansion**: A* only touches cells
   whose `f = g + h` is promising (a goal-cone toward the sink), not every cell
   in the box. Our sweep computes distance to *every* cell in the bounded
   region regardless. So bounding the region captures only a sliver of drt's
   search-space cut; the bulk is the expansion strategy, which a box doesn't
   change.

## Decision

- **Adopt goal-bounding as a cheap, safe, modest lever** (~1.2× single-net, 0
  quality loss). Worth wiring into the router region builder (bound to pin bbox
  when the guide is much larger), but it is **not** the drt-gap closer.
- **The real search-space lever is goal-biased expansion** — a bounded relax
  that only updates cells with `f = g + h` in a band (A*-like pruning, kept
  dense/regular so it still batches). That is the next, harder spike; it
  attacks the expansion strategy, not just the region.
- **Quantify the batched-router benefit before committing.** This measured
  *independent single-net* routing; in the bucketed batched router, bounding
  shrinks bucket-max sub-grids → less padding, which may compound — but
  bucketing already cut padding to ~1.0× median, so the end-to-end gain is
  likely also ~1.2×. Measure with a `bounded_region` flag on `GuideRouter`
  before banking it.

## What this does NOT cover

- **Shared-grid routability.** Independent sub-grids carry pin-access obstacles
  but not other nets' committed wires. The 0/500 loss is necessary, not
  sufficient: a tight box may fail when a committed wire blocks the direct
  corridor. drt handles this with **adaptive box growth** (re-route in a larger
  box on failure) — the natural mitigation if shared-grid loss appears.
- **End-to-end batched-router speedup** (padding interaction) — untested.
- **Goal-biased expansion** — the actual lever, deferred to its own spike.

## References

- [`gpu-astar-evaluation.md`](gpu-astar-evaluation.md) — why bounding (not GA*)
  is the fitting form of A*'s idea for our batch model.
- [`size-bucketed-batching.md`](size-bucketed-batching.md) — the drt
  like-for-like (~3.8×) this tries to chip at.
- `scripts/goal_bounded_sweep_prototype.py` — the harness.
