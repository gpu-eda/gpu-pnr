# Spike — size-bucketing the round-batched router

**Status:** Resolved — **positive** (2026-06-03). Size-bucketing is the
padding-poison fix; it makes the round-batched MPS router **22–33× faster
end-to-end** and **overturns the throughput basis of
[ADR 0013](../adr/0013-pause-e5-detailed-routing-on-mps.md)**. See ADR 0013
Amendment 1.

## Question

WS3.3 Slice 3's round-batched router collapsed to ~262–391 ms/net at sample
1000 (9–11× slower than sequential and than OpenROAD drt), which drove the
ADR 0013 decision to pause E5-on-MPS. The collapse was traced to
**padding-to-max waste**: a round pads every net to the batch's largest
sub-grid. Does **size-bucketing** — group similar-sized nets, pad within a
bucket only — recover the win, and does it carry end-to-end through the
backtrace, or is the router backtrace-bound (in which case bucketing wouldn't
help and the pause stands)?

## Method

Two levels, both on the Hazard3 gf180mcuD fixture at track pitch, M4 Pro MPS,
sample 1000, seed 0:

1. **Sweep-level** (`scripts/batched_sweep_prototype.py`, no router): batched
   sweep with `--batch 1000` (one giant batch = what the router did) vs
   `--batch 16 --sort-by-size` (size-bucketed) vs sequential. Isolates the
   sweep (backtrace excluded).
2. **End-to-end** (`scripts/guide_router_hazard3.py --bucket K`): the full
   `GuideRouter.route` with the new `bucket_size` knob — giant batch
   (`bucket_size=None`) vs `--bucket 16`.

## Result

### Sweep-level — padding is the whole disease

| config | batched ms/net | padding waste | whole-chip sweep |
|---|---:|---:|---:|
| size-bucketed (K=16, sorted) | **4.42** | 1.0× median, 2.3× mean | 91s |
| giant batch (K=1000) | 310.88 | **31×** | 6380s |
| sequential | 21.76 | — | 447s |

Bucketing turns the sweep **70× faster** (310.88 → 4.42 ms/net) and beats
sequential **4.9×** — confirming [ADR 0012](../adr/0012-tile-decomposition.md)
Amendment 4's size-sorted 4.05× at full scale. The 31× padding waste in the
giant batch is the entire cost.

### End-to-end — the win carries through the backtrace

| `GuideRouter.route` config | ms/net | whole-chip | routed | conflicts |
|---|---:|---:|---:|---:|
| giant batch (`bucket_size=None`) | 262–391 | ~134 min | 853/935 | 1587 |
| **size-bucketed (`--bucket 16`)** | **11.76** | **~4 min** | 853/935 | 1587 |

**22–33× faster end-to-end** (giant batch shows run-to-run variance from the
~1 GB tensor thrash, 262–391; bucketed is stable at 11.76). Crucially the
routing is **bit-identical** — same 853/935 routed, same 1587 deferred
conflicts — pinned by `test_bucketed_route_identical_to_single_batch`. The
bucketed router is now **3.4× faster than sequential** (~40 ms/net). vs drt it
is still **~3.8× slower at comparable work** — bucketing closed a ~100× gap, it
did not pull ahead (see "vs OpenROAD drt — the like-for-like" below).

### What this corrects

The ADR 0013 read — "GPU ~5%, 71% pipeline bubbles, backtrace-bound, the fixes
are CUDA-shaped" — was from a **35 s window of the 365 s run** and mis-attributed
the cause. The breakdown is now measured: the giant-batch *sweep* alone was
~290 s of the 365 s (the 1 GB padded tensor thrashing MPS reads as "bubbles" in
a short window). Bucketing drops the sweep to ~4 s; the remaining ~7 s is
backtrace/commit. So the router was **~80% padding, ~20% backtrace** — and
padding has a cheap, pure-MPS fix (no kernel change, no CUDA).

## Decision

1. **Size-bucketing is adopted** (`bucket_size` on `GuideRouter`, default knob
   to be set when WS3.3 resumes; K=16 measured here). It is **not** an optional
   lever — for the end-to-end router it is the difference between 4 min and 134
   min on MPS.
2. **ADR 0013's throughput pause is reversed** (ADR 0013 Amendment 1). The
   collapse was a fixable padding bug, not an MPS ceiling — the basis for "MPS
   can't win" is gone. (E5-on-MPS is **not** faster than drt; see below.)
3. **The next bottleneck is the per-net CPU backtrace** (~7 s / ~60% of the
   bucketed router's time). With buckets uniform-shaped, the best-pin argmin
   vectorises to one gather + `torch.min` per bucket; on-GPU backtrace is the
   lever beyond that. This is now the top WS3.3 throughput follow-up.

## vs OpenROAD drt — the like-for-like (we are still slower)

Comparing our single dirty pass to drt's *full* DRC-clean run (~673s) flatters
us; the honest comparison is against drt's **initial route** (0th iteration,
also dirty). From the run-05-08 detailed-routing log (drt is multi-threaded,
~5–10 cores):

| | ms/net | hardware | nets | state |
|---|---:|---|---:|---|
| drt initial route (0th iter, 74s) | **3.07** | ~5–10 CPU cores | 24,124 (all) | 11,788 viols |
| us, size-bucketed | **11.76** | 1 GPU stream | ~20.5k in-cap | 1,587 conflicts |

**drt is ~3.8× faster at comparable work**, routing more nets, on CPU, likely on
slower hardware. Bucketing closed a ~100× gap (giant-batch 262–391 ms/net) to
~3.8×, i.e. "hopeless → same ballpark," not "ahead." Per-compute-unit it's
closer (drt ~16.6 cpu-ms/net single-thread-equiv vs our 11.76 gpu-ms/net); drt
wins wall-clock by using its cores. Closing the remaining gap is a *search-space*
problem (drt: guided A\* + pattern routing over ~1–5k cells/net; us: full SSSP
sweep), not a bucketing one. Full decomposition: `docs/results.md` "drt
performance — the honest like-for-like."

## What this does NOT cover

- **Rip-up (Slice 4) / tail (Slice 5) / DRC.** Still 853/935 (91.2%) routed
  with 1587 unresolved conflicts; drt is 100% + DRC-clean. The throughput
  comparison is in-cap-only; the *complete* competitiveness claim still needs
  Slices 4–6.
- **Convergence-masking** (slowest-net-bounds-the-batch). Untested; a further
  lever on top of bucketing, smaller than the padding fix.
- **Bucket-size tuning.** K=16 measured; the optimum (padding waste vs GPU
  occupancy) is unswept.
- **Hardware-matched wall-clock vs drt.** drt's ~12 min is on unknown hardware;
  per-net throughput is the fair axis.

## References

- [ADR 0013](../adr/0013-pause-e5-detailed-routing-on-mps.md) Amendment 1 — the
  reversal this spike drives.
- [ADR 0012](../adr/0012-tile-decomposition.md) Amendments 4 (size-sorted 4.05×)
  & 5 (the giant-batch walk-back this corrects).
- [`gpu-vs-drt-throughput.md`](gpu-vs-drt-throughput.md) — the throughput
  comparison framing.
- `scripts/batched_sweep_prototype.py`, `scripts/guide_router_hazard3.py
  --bucket` — the harnesses.
- `docs/results.md` Phase 3.3 — folded measurement tables.
