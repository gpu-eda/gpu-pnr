# Handoff — WS3.3 GuideRouter: Slices 1–3 landed, Slice 4 (ripup) next

**Created:** 2026-05-28; refreshed 2026-06-02 (GuideRouter Slice 3 landed).
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
- **Slice 3** — round-batched routing (`_NetWork` + per-net `extra_sources` on
  `sweep_sssp_3d_batched`). **MPS 1.34× faster than Slice 2** (32.1 vs 43.0
  ms/net); CPU regresses (padding waste, no parallelism). Rough vs OpenROAD drt:
  wirelength 1.003× on the in-cap subset. `docs/results.md` Phase 3.3 "Slice 3".

## Next up: Slice 4 — cross-net conflict detect + rip-up / reroute

Round-batching deliberately defers same-round conflicts (Slice 3 leaves ~19/99
in the Hazard3 sample). Slice 4 detects cells claimed by ≥2 nets, keeps the
lowest-HPWL net (ADR 0007), requeues losers, reroutes against the updated
`w_cur` in bounded rounds (≤3). Plan's Slice 4 section has the sketch + tests.

**Watch-outs for Slice 4 (verified live in `guide_router.py`):**

- **Rip-up must clear `committed` bits on un-commit.** The commit step sets
  `committed[cells] = True` (guide_router.py ~line 488); un-committing a
  rerouted net must clear them, or the net is wrongly re-blocked. Comment is in
  place at the `committed` block.
- **`prep_subgrid` is per-sub-grid + the `committed` re-block** — both correctly
  handled in the round loop (per-net clone → prep → `w_sub[sub_committed]=inf`
  before stacking). Slice 4's reroute batches go through the same path; keep it.

## Carried follow-ups from Slice 3 (not blocking Slice 4)

- **Device-aware dispatch** — round-batching wins on MPS, loses 11.4× on CPU
  (padding-to-max waste, no parallelism). A production router should route
  sequential on CPU, round-batched on MPS.
- **Backtrace is the walk-back watch** — per-net CPU backtrace (per-pin `.item()`
  argmin + per-net `.cpu()`) is the unamortised cost; vectorise + hoist, or push
  onto the GPU. Profile in Slice 6 (noted inline in `guide_router.py`).
- **Size-bucketing (deferred Am4 lever)** — the 1.34× MPS win is below the
  kernel spike's 2.46–4.05× because one batch pads every net to the largest
  sub-grid (max 131k cells). Option-B bucketing recovers it; post-router.
- **Extract `attach_nearest_pin`** — the per-attachment kernel (sweep→pick→
  backtrace→grow) is duplicated between `route_multipin_nets_3d` and the round
  loop; share it once Slice 4 settles whether the attachment step changes.
- **`drt_compare.py` adds a 4th net-sampling copy** — fold into the `sample_nets`
  helper below when it lands.

## OpenROAD drt reference (the Slice 6 quality gate)

`drt` = TritonRoute = the WS3.3 exit criterion (≤1.2× wire, ≤1.2× vias). The
reference is pre-computed in the LibreLane fixture (`RUN_2026-05-08_22-32-24/
44-openroad-detailedrouting`, the same run our GUIDE + FINAL_DEF read), so the
Slice 6 gate is a metrics-parse, not an OpenROAD run: **24,124 nets, 1,234,353 µm
wire, 181,514 vias, 0 DRC, ~12 min**. `scripts/drt_compare.py` is the harness.

## Verification command

```sh
cd ~/Code/gpu-pnr && uv run pytest tests/
# Expect: 111 passed

uv run pytest tests/test_guide_router.py -q          # Expect: 20 passed
# Slice 3 round-batched A/B (MPS 1.34× vs Slice 2) + drt comparison:
uv run python scripts/guide_router_hazard3.py --device mps --sample 100
uv run python scripts/drt_compare.py --device mps --sample 300
```

## Loose ends (not blocking Slice 4)

- **Carried cleanup:** the net-sampling loop (shuffle/filter/build pins) is now
  duplicated across `track_pitch_sweep_prototype`, `batched_sweep_prototype`,
  `guide_router_hazard3`, and `drt_compare`. Consolidate a `sample_nets` helper +
  the shared constants (`MIN_PINS`, `MAX_PINS`, `_net_pins`, `_pct`,
  `HAZARD3_ROUTABLE_NETS`) into `_hazard3_io.py` before another script needs them.
- **Deferred throughput levers** (post-router): convergence-masking and option-B
  size-bucketing — see `docs/spikes/batched-small-grid-sweep.md` "next levers".
- **CI bench baseline** (follow-up 5, optional): Tier B already concluded the
  Tier-A 4× erosion is environmental; confirming on M2 at `e5dd5be` is optional.
- **Pin-access ADR amendment** (blocked): the snap-vs-local-fine-region decision
  needs DEF pin geometry (the guide fixture is GCell-granular). ADR 0012 Am3
  open Q#1. Not on the WS3.3 critical path.

## Critical context for Slice 4

- **The router is PDK-agnostic by design.** It routes on whatever `w_chip` it's
  given; PDK rules (pin-access) are injected via the `prep_subgrid` hook, wired
  up only in `scripts/guide_router_hazard3.py`. Keep it that way.
- **Round-batching shares one `w_cur` snapshot per round** — Slice 4's conflict
  detect runs on the committed set *after* a round, and reroute losers go into a
  subsequent round's batch (not mid-round). HPWL order is preserved through the
  `in_cap` → `work` → `active` chain.
- **`net_bbox` survives** in `guide_router.py` as the no-guide fallback region
  builder + HPWL source. The rest of the old tile machinery is gone.

## Migration note

This handoff resolves when WS3.3 ships (Slice 6). Everything load-bearing is
already in its permanent home — plan (slice status), ADR 0012 (Am 1–4),
`docs/results.md` Phase 3.3, and the spikes. At that point `git rm` this file in
the same commit that flips the WS3.3 boxes in `phase3-detailed-routing.md`:
`docs: resolve WS3.3 handoff — guide-constrained router shipped`.
