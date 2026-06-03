# Handoff — WS3.3 GuideRouter: Slice 3 + size-bucketing landed, Slice 4 (rip-up) next

**Created:** 2026-06-03
**Working tree:** clean; all work pushed to `main` (CI green).
**Branch:** main

<!--
Ephemeral. At resolution every load-bearing piece migrates to docs/adr,
docs/plans, docs/spikes; then `git rm` this file. See docs/handoff-discipline.md.
-->

## Goal & next-up

**Goal of this session:** Resume Slice 3, build the batched router, and measure
it honestly against drt. It turned into a measure → pause → *reverse* arc: Slice
3 round-batching collapsed at scale, we paused E5-on-MPS (ADR 0013), then found
the cause was a padding bug fixed by **size-bucketing** (22–33×), reversed the
pause (ADR 0013 Am1), corrected an over-claimed drt comparison, and evaluated
two search-space levers.

**Next session should pick up:** **WS3.3 Slice 4 — cross-net conflict detect +
rip-up/reroute** (`docs/plans/ws33-tile-router-implementation.md` Slice 4 section
has the full sketch + tests). The bucketed router leaves ~1587 deferred conflicts
at sample 1000; Slice 4 resolves them (lowest-HPWL wins, requeue losers, bounded
rounds ≤3). Unlocks ADR 0008.

**Verification command:**

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/        # Expect: 113 passed
uv run pytest tests/test_guide_router.py -q      # Expect: 22 passed
# Size-bucketing A/B (22-33× faster, bit-identical routes):
uv run python scripts/guide_router_hazard3.py --device mps --sample 1000 --bucket 16
```

## Done this session

| Commit | Subject | Notes |
|---|---|---|
| `07411cb` | Slice 3 round-batched routing | kernel `extra_sources`, `_NetWork`, 5 tests |
| `da4d9c5` | mark Slice 3 done | (later superseded by the arc below) |
| `44106fe` | Slice 3 throughput walk-back | ADR 0012 Am5: collapse at scale |
| `30a2e4b` | pause E5-on-MPS | ADR 0013 (later reversed) |
| `859c703` | resolve old WS3.3 handoff | folded into ADR 0013 |
| `2f00e51` | **size-bucketing** | 22–33× faster, routes bit-identical, `bucket_size` |
| `3797d2b` | reverse the pause | ADR 0013 Am1; WS3.3 resumed |
| `3611e16` | correct drt comparison | we are ~3.8× *slower*, not faster |
| `8aea444` | GA* + goal-bounded spikes | search-space levers evaluated |

## Open follow-ups (priority-ordered)

### 1. Slice 4 — rip-up / reroute (the next build)

The terminal-ish correctness slice. Plan has tests + the ≤3-round bound. **Key
watch-out (verified in code):** the commit step sets `committed[cells]=True`
(`src/gpu_pnr/guide_router.py` commit block); rip-up's un-commit **must clear
those bits** or a rerouted net is wrongly re-blocked. Bucketing/prep path is
already correct — reroute batches go through the same `_attach_batch`.

### 2. Throughput levers (post-Slice-4, all in the plan's Slice 3 follow-ups)

- **Backtrace is now the top cost** (~60% of the bucketed router). Within
  uniform-shaped buckets the best-pin argmin vectorises (gather + `torch.min`);
  on-GPU backtrace beyond. Inline NOTE at `_attach_batch`'s backtrace loop.
- **Search-space (the drt ~3.8× gap):** goal-bounding the region is a *modest*
  ~1.2× safe lever (`docs/spikes/goal-bounded-sweep.md`); the **real** lever is
  **goal-biased expansion** (A*-style f-band, kept dense/batchable — NOT GA*,
  see `docs/spikes/gpu-astar-evaluation.md`). Its own future spike.
- Pick a default `bucket_size` and drop the `bucket_size=None` A/B path (+ its
  equivalence test). Convergence-masking. Consolidate the 5×-duplicated
  net-sampling loop into a `sample_nets` helper in `_hazard3_io.py`.

## Critical context

- **We are NOT faster than drt.** The like-for-like (our dirty pass vs drt's
  dirty *initial* route): drt 3.07 ms/net vs us 11.76 → **drt ~3.8× faster**,
  multi-threaded CPU. An earlier draft claimed "~2.5× faster" by comparing our
  pass to drt's *full DRC-clean* run — corrected in `docs/results.md` "drt
  performance — the honest like-for-like." Bucketing closed a ~100× gap to
  ~3.8×: "hopeless → same ballpark," not "ahead."
- **Two `verify-don't-assume` catches this session, both on my own claims**
  (the 35 s profile window mis-attributing the collapse; the drt full-run
  divide). Decompose measurements before drawing end-to-end conclusions.
- **The drt gap is search-space, not tuning:** drt's guided A*/pattern routing
  touches ~1–5k cells/net; our full SSSP sweep touches the whole sub-grid.
  Bounding the *box* ≠ goal-directing the *search* — `goal-bounded-sweep.md`
  separates these cleanly.
- **CUDA stays the endgame** (ADR 0013 Am1, ADR 0001): graphs kill the eager
  dispatch syncs, custom kernels move backtrace on-GPU. E5-on-MPS is now
  competitive-ballpark, which *raises* the bar for what CUDA must beat.

## References

- [`../plans/ws33-tile-router-implementation.md`](../plans/ws33-tile-router-implementation.md) — WS3.3 build status (Active); Slice 4 sketch.
- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md) — WS3.3 in the Phase 3 plan.
- [`../adr/0013-pause-e5-detailed-routing-on-mps.md`](../adr/0013-pause-e5-detailed-routing-on-mps.md) — the pause + Amendment 1 reversal (the session's spine).
- [`../adr/0012-tile-decomposition.md`](../adr/0012-tile-decomposition.md) Am5 — the throughput walk-back.
- [`../results.md`](../results.md) Phase 3.3 — all measurement tables + the honest drt comparison.
- `docs/spikes/size-bucketed-batching.md`, `gpu-astar-evaluation.md`, `goal-bounded-sweep.md` — the three spikes.

## Migration note

When WS3.3 ships (Slice 6): everything load-bearing already lives in its
permanent home (plan slice-status, ADR 0012 Am1–5, ADR 0013 + Am1, results.md
Phase 3.3, the spikes). `git rm` this file in the same commit that flips the
WS3.3 boxes in `phase3-detailed-routing.md`. Follow-up 2's levers, if pursued,
land as plan Slice 3 follow-up updates or new spikes.
