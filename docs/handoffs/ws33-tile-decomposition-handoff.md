# Handoff — WS3.3 GuideRouter: Slices 1–2 landed, Slice 3 (batched) next

**Created:** 2026-05-28; refreshed 2026-06-02 (GuideRouter Slices 1–2 landed).
**Working tree:** clean; all work pushed to `main` (CI green).
**Branch:** main

<!--
A handoff is ephemeral: it captures only what's in flight + the next pickup.
The durable record already lives in its permanent homes — don't duplicate it
here. At resolution, `git rm` this file (see Migration note).
-->

## Where things are

WS3.3 is mid-build: the guide-constrained router is being implemented in
6 slices per
[`ws33-tile-router-implementation.md`](../plans/ws33-tile-router-implementation.md)
(the plan is the source of truth for slice status + design). Design decisions
all live in [ADR 0012](../adr/0012-tile-decomposition.md) Amendments 1–4.

**Done (folded into their homes — listed for orientation, not as a log):**

- **Substrate**: track-pitch grid, `guide_region` mapper, `sweep_sssp_3d_batched`
  (batched kernel, 2.46–4.05× over sequential on MPS — ADR 0012 Am4 +
  `docs/spikes/batched-small-grid-sweep.md`).
- **Slice 1** — `gpu_pnr.guide_router`: `classify_nets` (in-cap vs over-cap/
  no-guide tail), `NetPlan`, HPWL ordering. Replaced the deleted `tile_router.py`.
- **Slice 2** — `GuideRouter.route`: single-stream routing on the shared
  `w_cur`. Hazard3-validated: **0 cross-net conflicts (CPU+MPS)**, 96% routed.
  `docs/results.md` Phase 3.3 "GuideRouter Slice 2".

## Next up: Slice 3 — batched routing via `sweep_sssp_3d_batched`

Replace Slice 2's sequential per-net sweep with batched groups (the plan's
Slice 3 section has the full sketch). Multi-pin batching strategy is **settled
= round-batching** (`docs/spikes/multi-pin-batching-strategy.md`: 2-pin nets are
only 8.8% of sweep-work, so batching them alone caps the win at ~1.1×). Needs a
per-net `extra_sources` extension on `sweep_sssp_3d_batched`.

**Watch-outs carried from Slice 2 (these will bite Slice 3/4 if ignored):**

- **`prep_subgrid` is per-sub-grid.** The batched path must apply it per-net
  *before* padding/stacking into the `(K,L,H,W)` tensor — it can't run on the
  stacked tensor.
- **The `committed` bool-mask re-block must carry into the batched path.** The
  pin-access prep rewrites landing-pad cells to finite, which resurrects prior
  nets' committed wires (this caused 4 cross-net conflicts in Slice 2 before the
  fix). Any batched commit must re-block committed cells after prep too.
- **Slice 4 rip-up must clear `committed` bits on un-commit** — noted in
  `guide_router.py`. A rerouted net wrongly re-blocked otherwise.

## Verification command

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/
# Expect: 106 passed

uv run pytest tests/test_guide_router.py -q          # Expect: 18 passed
# Slice 2 shared-grid validation: 0 cross-net conflicts, ~96% routed
uv run python scripts/guide_router_hazard3.py --device cpu --sample 100
```

## Loose ends (not blocking Slice 3)

- **Carried cleanup:** the net-sampling loop (shuffle/filter/build pins) is
  duplicated across `track_pitch_sweep_prototype`, `batched_sweep_prototype`,
  and `guide_router_hazard3`. Consolidate a `sample_nets` helper + the shared
  constants (`MIN_PINS`, `MAX_PINS`, `_net_pins`, `_pct`, `HAZARD3_ROUTABLE_NETS`)
  into `_hazard3_io.py` before another script needs them.
- **Deferred throughput levers** (post-router): convergence-masking and option-B
  size-bucketing — see `docs/spikes/batched-small-grid-sweep.md` "next levers".
- **CI bench baseline** (follow-up 5, optional): Tier B already concluded the
  Tier-A 4× erosion is environmental; confirming on M2 at `e5dd5be` is optional.
- **Pin-access ADR amendment** (blocked): the snap-vs-local-fine-region decision
  needs DEF pin geometry (the guide fixture is GCell-granular). ADR 0012 Am3
  open Q#1. Not on the WS3.3 critical path.

## Critical context for Slice 3

- **The router is PDK-agnostic by design.** It routes on whatever `w_chip` it's
  given; PDK rules (pin-access) are injected via the `prep_subgrid` hook, wired
  up only in `scripts/guide_router_hazard3.py`. Keep it that way.
- **CPU beats MPS at this grain** (single net ~4k cells, overhead-bound) — the
  track-pitch prototype and Slice 2 both confirm it. Slice 3's batched kernel is
  precisely the fix; the win to reproduce end-to-end is the spike's 2.46–4.05×.
- **`net_bbox` survives** in `guide_router.py` as the no-guide fallback region
  builder + HPWL source. The rest of the old tile machinery is gone.

## Migration note

This handoff resolves when WS3.3 ships (Slice 6). Everything load-bearing is
already in its permanent home — plan (slice status), ADR 0012 (Am 1–4),
`docs/results.md` Phase 3.3, and the spikes. At that point `git rm` this file in
the same commit that flips the WS3.3 boxes in `phase3-detailed-routing.md`:
`docs: resolve WS3.3 handoff — guide-constrained router shipped`.
