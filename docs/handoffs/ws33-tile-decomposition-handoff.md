# Handoff — WS3.3 pivot: guide-constrained sweep replaces fixed-tile K-batching

**Created:** 2026-05-28 (replaces 2026-05-15 handoff after Tier B + DRT comparison)
**Working tree:** clean
**Branch:** main

<!--
Reminder: a handoff is ephemeral. At resolution, every load-bearing piece
below migrates into a docs/adr/, docs/plans/, docs/spikes/, or design-doc
home, and this file is then `git rm`'d in the same commit as the migration.

See docs/handoff-discipline.md for the migration table.
-->

## Goal & next-up

**Goal of this session:** Establish CI golden benchmarks, measure CPU vs
MPS throughput, compare against TritonRoute DRT, and determine whether
the fixed-tile K-batching approach from ADR 0012 is viable.

**Outcome:** K-batching is dead (confirmed on both local M4 Pro and CI
M2). GPU sweep is 7-13× faster than CPU — but DRT is faster per-net
because it searches 65-328× fewer cells via guide-constrained A\*.
ADR 0012 amended to pivot toward guide-constrained adaptive sweep.

**Next session should pick up:** Design and prototype the
guide-constrained sweep — the new core of WS3.3 per ADR 0012
Amendment 1. Concrete first step: write a GRT guide parser that reads
OpenROAD's `*.guide` output and maps guide regions to sub-grid
bounding boxes in our coordinate system.

**Verification command:**

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/
# Expect: 77 passed

# Confirm CI bench workflow runs
gh run list --repo gpu-eda/gpu-pnr --limit 2
# Expect: 2 completed Bench runs (MPS-only and MPS+CPU)

# Confirm ADR 0012 amendment exists
grep "Amendment 1" docs/adr/0012-tile-decomposition.md
# Expect: "## Amendment 1 (2026-05-28): guide-constrained sweep..."
```

## Done this session

| Commit | Subject | Notes |
|---|---|---|
| `ae40ec7` | ci: add bench workflow for golden GPU perf measurements | Self-hosted M2 runner, pipefail, concurrency, timeouts |
| `abf98e5` | ci: add CPU vs MPS device matrix to bench workflow | 6-job matrix (3 sizes × 2 devices), argparse choices validation |
| `2a79f6d` | docs: spike — GPU sweep vs TritonRoute DRT throughput comparison | Load-bearing finding: 65-328× search space gap |
| (pending) | docs: ADR 0012 Amendment 1 + handoff | Guide-constrained sweep pivot |

## Open follow-ups (priority-ordered)

### 1. GRT guide parser (small-medium)

Read OpenROAD's `*.guide` file format. Map guide GCell regions to
(layer, row_start, row_end, col_start, col_end) sub-grid bounding
boxes in our coordinate system. The Hazard3 fixture at
`~/Code/Apitronix/hazard-test` should have guide output from
LibreLane's GRT step.

### 2. Sub-grid sweep prototype (medium)

Prove that sweeping a small sub-grid (e.g., 50×30×2 = 3,000 cells)
extracted from the chip-scale cost tensor produces correct routes.
Key questions:
- Can we index into `w_cur` with a bbox slice without copying?
- What's the actual ms/net at sub-grid scale on MPS?
- Does the 0.16 ms/net estimate from the spike hold?

### 3. Batched small-grid sweep kernel (medium-large)

The GPU parallelism win comes from batching many independent small
sub-grids in one kernel call. Design options:
- Pad all sub-grids to a common size and batch as a 4D tensor
- Use a scatter/gather approach with per-net offset indices
- Investigate `torch.vmap` or `torch.compile` for variable-size batching

This is the hardest piece and may need its own spike.

### 4. Update WS3.3 plan (small)

Revise `docs/plans/ws33-tile-router-implementation.md` to reflect the
guide-constrained architecture. The 8-slice plan is obsolete — Slices
1-2 shipped (tile geometry + Hazard3 measurement), Slices 3-8 need
rewriting around the new approach.

### 5. Resolve CI bench baseline question (small)

The CI M2 numbers show 0 K-batching benefit and ~3× slower than local
M4 Pro, as expected. Consider running the bench at `e5dd5be` (Tier A
commit) via `workflow_dispatch` with `ref: e5dd5be` to definitively
confirm whether Tier A's 4× ever existed on M2. Low priority — the
Tier B spike already concluded environmental regression.

## Critical context

**ADR 0012 Amendment 1 is the load-bearing document.** It captures the
full pivot rationale and the new design. The original ADR sections
(§1-§7) remain for historical context but are largely superseded.

**The DRT comparison changes the success metric.** The original target
was "22-44 min for Hazard3 on MPS." The new target is: competitive
with DRT's 2-minute 14-threaded detailed routing on the same design.
Guide-constrained sweep at 0.16 ms/net × 20k nets = 3.2s sweep time
makes this plausible, but backtrace + conflict detection + rip-up
iterations will add significant overhead.

**GPU acceleration is validated but the value is architectural, not
raw speed.** GPU is 7-13× faster than CPU on the same grid (M4 Pro).
The path to beating DRT is combining GPU speed with DRT-scale search
spaces (guide-constrained), not brute-forcing larger grids.

**CI infrastructure is live.** Self-hosted M2 Mac Mini runner
(`macos-runner-1`) at `gpu-eda` org. Bench workflow triggers on push
to main (when src/ or scripts/bench* change) and on `workflow_dispatch`.
Golden baselines captured in GitHub Actions artifacts.

**The tile_router.py module (Slice 1) still has useful code.** The
`partition_chip`, `net_bbox`, `classify_nets` functions and tests may
be repurposed for guide-region partitioning. Don't delete — refactor.

## References

- [ADR 0012 Amendment 1](../adr/0012-tile-decomposition.md) — the
  guide-constrained sweep pivot
- [GPU vs DRT spike](../spikes/gpu-vs-drt-throughput.md) — throughput
  comparison and DRT architecture summary
- [Tier B spike](../spikes/tier-b-envelope-throughput.md) — K-batching
  dead, sequential is the design parameter
- [`../plans/ws33-tile-router-implementation.md`](../plans/ws33-tile-router-implementation.md)
  — 8-slice plan (obsolete; needs rewrite per follow-up 4)
- [`../plans/phase3-detailed-routing.md`](../plans/phase3-detailed-routing.md)
  — Phase 3 plan (WS3.3 section needs update)

## Migration note

When the guide-constrained prototype lands and this handoff resolves:

- Follow-up 1 (guide parser) → code in `src/gpu_pnr/` + tests
- Follow-up 2 (sub-grid prototype) → results in `docs/results.md`
- Follow-up 3 (batched kernel) → new ADR if design is non-obvious
- Follow-up 4 (plan rewrite) → updated `docs/plans/ws33-tile-router-implementation.md`
- Then `git rm docs/handoffs/ws33-tile-decomposition-handoff.md`
  in the migration commit. Commit message: `docs: resolve WS3.3
  handoff — fold into plan + guide-constrained prototype`.
