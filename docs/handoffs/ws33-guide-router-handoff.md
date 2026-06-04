# Handoff — WS3.3 GuideRouter: Slice 4 (rip-up) landed, Slice 5 (coarsened tail) next

**Created:** 2026-06-04
**Working tree:** clean; Slice 4 committed (`ba44d82`), pushed to `main`.
**Branch:** main

<!--
Ephemeral. At WS3.3 resolution (Slice 6) every load-bearing piece migrates to
docs/adr, docs/plans, docs/spikes, results.md; then `git rm` this file. See
docs/handoff-discipline.md.
-->

## Goal & next-up

**Done this session:** **WS3.3 Slice 4 — cross-net conflict detect + rip-up /
reroute** (`ba44d82`). Same-round nets share one `w_cur` snapshot so two can
claim a cell; Slice 4 detects ≥2-claimant cells, the lowest-HPWL net keeps each
(ADR 0007), losers are ripped up + rerouted (≤3 rounds, then left unrouted).
Rip-up restores each loser's loser-exclusive cells from the original
`w_chip`/`w_v_chip` (full restore, not clear-to-finite — the handoff watch-out)
and clears the `committed` bits. Suite 113 → 116. **Hazard3 sample-1000: 0
cross-net conflicts** (was ~1587 deferred) — exit criterion met.

**Next session should pick up:** **WS3.3 Slice 5 — coarsened-pass fallback for
the over-cap / no-guide tail** (`docs/plans/ws33-tile-router-implementation.md`
Slice 5 section has the full sketch + tests). The ~6% over-cap (Amendment 3) +
no-guide nets route on a 4× coarsened grid, pinned as obstacles before the
in-cap batched passes.

**But first, a Slice 4 follow-up worth a quick look (walk-back signal):** at
sample-1000, in-cap routability is **77.1%** (214/935 unrouted). That's the
honest cost of resolving conflicts (a Slice-3 net counted "routed" while
overlapping is now forced to reroute-or-fail), but 22.9% unrouted trips the
plan's ">1% fail to converge" walk-back. **The current run doesn't separate
rip-up non-convergence from initial cross-net contention** — instrument that
split, then decide whether to raise `MAX_RIPUPS` to 5 (one-line + re-measure)
or leave the tail to Slice 5. See the plan's Slice 4 carried follow-up.

**Verification command:**

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/        # Expect: 116 passed
uv run pytest tests/test_guide_router.py -q      # Expect: 25 passed
# Slice 4 — 0 cross-net conflicts on Hazard3 sample-1000:
uv run python scripts/guide_router_hazard3.py --device mps --sample 1000 --bucket 16
#   → "cross-net conflicts: 0  (none)"; routed 721/935 (77.1%)
```

## Critical context

- **The conflict-resolution invariant is the gate, not routability.** 77.1%
  routed + 0 conflicts is *more* correct than a higher routed-fraction with
  illegal overlaps. Don't read the routability drop as a regression — read it
  as the count going honest.
- **The un-commit watch-out is handled** (`_ripup_net`): loser-exclusive cells
  full-restore from `w_chip`; cells shared with a surviving winner stay
  committed to the winner; `committed` bits cleared exactly where restored.
- **`route` is now layered cleanly:** `_route_population` is a pure "drain to
  completion against the current `w_cur`" primitive; `route` wraps it in the
  rip-up loop. Population keyed by one `net_to_work` dict (HPWL-ascending
  insertion order preserved across requeue). `_cell_index_tensors` shares the
  batched index-assignment (no per-cell MPS kernel launches).
- **We are NOT faster than drt** (carried, unchanged): like-for-like drt 3.07
  ms/net vs us 11.76 → drt ~3.8× faster. The search-space gap (drt's guided
  A*/pattern routing vs our full SSSP sweep) is the real lever, not tuning —
  see `docs/spikes/goal-bounded-sweep.md` + `gpu-astar-evaluation.md`.

## Open follow-ups (priority-ordered)

1. **Slice 4 routability / walk-back** (above) — instrument rip-up-fail vs
   contention-fail, decide `MAX_RIPUPS`. Quick, do before/with Slice 5.
2. **Slice 5 — coarsened-pass tail** (the next build; plan has the sketch).
3. **Throughput levers** (post-Slice-5, all in the plan's Slice 3 follow-ups):
   backtrace is the top cost (~60%); goal-biased expansion is the real
   search-space lever; pick a default `bucket_size` + drop the A/B path;
   consolidate the 5×-duplicated net-sampling loop into `sample_nets`.

## References

- [`../plans/ws33-tile-router-implementation.md`](../plans/ws33-tile-router-implementation.md) — Slices 1–4 done; Slice 5 sketch + Slice 4 carried follow-up.
- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md) — WS3.3 in the Phase 3 plan.
- [`../adr/0013-pause-e5-detailed-routing-on-mps.md`](../adr/0013-pause-e5-detailed-routing-on-mps.md) — the pause + Am1 reversal.
- [`../adr/0008-defer-route-nets-batched.md`](../adr/0008-defer-route-nets-batched.md) — the rip-up unlock Slice 4 delivers.
- [`../results.md`](../results.md) Phase 3.3 — Slice 2/3/4 tables + the honest drt comparison.

## Migration note

When WS3.3 ships (Slice 6): everything load-bearing already lives in its
permanent home (plan slice-status, ADR 0012 Am1–5, ADR 0013 + Am1, results.md
Phase 3.3, the spikes). `git rm` this file in the same commit that flips the
WS3.3 boxes in `phase3-detailed-routing.md`.
