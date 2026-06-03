# Spike — GPU A* (Zhou & Zeng GA*) evaluation for detailed routing

**Status:** Resolved — **evaluated, not adopted** (2026-06-03). GA*'s
within-search parallelism is the wrong shape for our many-small-nets workload
(by the paper's own criteria); but the *idea* it points at — goal-directed
search instead of full-grid SSSP — is exactly our ~3.8× throughput gap vs drt
(`size-bucketed-batching.md` → drt like-for-like). The follow-on
[`goal-bounded-sweep.md`](goal-bounded-sweep.md) tests the part that fits us.

## Question

Our bucketed router is ~3.8× slower than drt's comparable (initial-route) pass,
and the cause is search-space: drt's guided A* touches ~1–5k cells/net; our SSSP
sweep touches the whole sub-grid. Would the GPU A* of Zhou & Zeng — "Massively
Parallel A* Search on a GPU" (AAAI 2015), implemented at
[`jbujak/A-star-CUDA`](https://github.com/jbujak/A-star-CUDA) — close it?

## What GA* is

A method to parallelize **one** A* search across a GPU: each iteration extracts
**K best nodes from K parallel priority queues** and expands them simultaneously
(their Algorithm 1), with **parallel cuckoo hashing** (or hashing-with-
replacement) for closed-list duplicate detection, and a memory-bounded frontier
variant for huge closed lists. Reported up to **45× over sequential CPU A***, on
**exponential-state** problems — sliding puzzles, Rubik's cube, protein design.

## Finding — poor fit for the bulk, by the paper's own criteria

Three points, each grounded in the paper's text:

1. **Wrong parallelism axis.** GA* parallelizes *within* one giant search. We
   have ~20k **independent small** searches (one per net, ~4k cells). The paper
   explicitly sets our model aside: it notes prior GPU A* (NVIDIA/Bleiweiss)
   "able to solve multiple small A* search problems simultaneously" but says
   they "cannot parallelize an individual A* search" — and that
   batched-across-searches model *is* ours. We don't lack within-net
   parallelism; we have abundant across-net parallelism (our batched sweep).
   GA* solves a problem we don't have.
2. **Grid pathfinding is its weakest case — stated outright.** The paper: in
   grid pathfinding "the degree of a node is usually less than ten, which
   limits the degree of parallelism." Our 3D grid is **degree 6** (4 in-layer +
   2 via). GA*'s within-search parallelism is starved exactly here.
3. **K-best extraction wastes work.** Expanding K-best instead of 1-best
   generates extra node expansions (the paper flags the trade-off). On a
   ~4k-cell net there is nothing to amortise that overhead against.

## The part that *does* fit — and why not GA*

A* goal-direction is the right lever for our gap, but the open question is how to
get it **without losing the batch-across-nets model that is our edge**:

- Our **sweep batches well precisely because it is branch-free** (fixed
  `cumsum`/`cummin` passes over a dense tensor).
- **A* is irregular** (data-dependent priority queue, ragged per-net frontier) →
  SIMD **divergence** the moment it's batched across nets. Naive "K A* searches
  in lockstep" trades our biggest advantage for A*'s efficiency.

So the reconciliation keeps the regular, batchable structure and adds goal-
direction by **bounding the region**, not by porting A*:

1. **Goal-bounded sweep (next spike).** Shrink each attachment's region to a
   source→sink bbox + margin (goal corridor) instead of the full guide bbox —
   most of A*'s search-space cut, zero irregularity, composes with bucketing.
2. **Goal-biased wavefront.** A bounded relax that only touches cells with
   `f = g + h` in a band (A*-like pruning, kept dense). More of A*'s efficiency,
   more work to build.
3. **GA* per *tail* net only.** The few over-cap clock/power nets (Slice 5 tail)
   are the one place we have "few, large searches" — GA*'s actual regime. A
   niche; not the bulk.

## Decision

- **Do not adopt GA* / `jbujak/A-star-CUDA` for the bulk router.** Wrong
  parallelism shape; the repo is MIT but archived (19 commits, 77★), single-
  search, **2D unit-cost** grid + puzzle demos — no batching, no 3D, no vias, no
  weighted obstacles. Not a usable library here.
- **Pursue goal-direction via goal-bounded sweep** (keeps our batch model) —
  tested in [`goal-bounded-sweep.md`](goal-bounded-sweep.md).
- **Keep GA* as a reference** only for a future per-net GA* on the over-cap
  tail (its parallel-PQ + cuckoo-hash dedup), the lowest-priority path.
- The more relevant external prior art for our shape is the **batched
  many-small-searches** line (NVIDIA/Bleiweiss GPU multi-agent pathfinding) the
  paper cites and sets aside — worth a look if/when goal-bounded sweep plateaus.

## References

- Zhou & Zeng, "Massively Parallel A* Search on a GPU," AAAI 2015
  (<https://cdn.aaai.org/ojs/9367/9367-13-12895-1-2-20201228.pdf>).
- [`jbujak/A-star-CUDA`](https://github.com/jbujak/A-star-CUDA) — MIT
  implementation (single-search; 2D grid + puzzle demos).
- [`size-bucketed-batching.md`](size-bucketed-batching.md) — the drt
  like-for-like that motivates closing the search-space gap.
- [`gpu-vs-drt-throughput.md`](gpu-vs-drt-throughput.md) — drt's guided-search
  search-space advantage (the original framing).
- [`goal-bounded-sweep.md`](goal-bounded-sweep.md) — the follow-on test.
