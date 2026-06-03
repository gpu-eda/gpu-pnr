# ADR 0013 — Pause E5 detailed-routing throughput work on MPS

**Status:** Accepted (2026-06-02); **REVERSED by Amendment 1 (2026-06-03)** —
the pause rested on the *un-bucketed* router; size-bucketing makes the MPS
router 22–33× faster, narrowing the gap to drt from ~100× to ~3.8× (still
slower, but no longer an MPS ceiling). Read the original decision below as
superseded; the critical-learnings retrospective stands (with the correction
in Amendment 1).

## Context

E5 (sweep-based detailed routing, [`../architecture.md`](../architecture.md))
reached its first end-to-end real-fixture throughput measurement in WS3.3
Slice 3: the guide-constrained round-batched router, run whole-sample on the
Hazard3 gf180mcuD fixture and compared against OpenROAD's detailed router
(`drt` / TritonRoute). The result is a **measured negative**, and it
falsifies the core projection this whole workstream was built on.

The [`gpu-vs-drt-throughput.md`](../spikes/gpu-vs-drt-throughput.md) spike
(2026-05-28) projected that guide-constrained sweep would route at **~0.16
ms/net, ~112× faster than DRT's single-threaded ~10.7 ms/net**, and that
"the real GPU win comes from batching many small sub-grids." Every step of
WS3.3 (ADR 0012 Amendments 1–4) was justified by that projection. Slice 3
measured it. It was wrong on three independent counts — see below. This is
not a bug to fix; it is a falsified hypothesis, and the honest response is to
stop building Slices 4–6 for an MPS throughput win that the measurements say
cannot materialise on this hardware at this design size.

## The three falsified assumptions (the critical learning)

The spike's ~112× projection rested on three premises. Slice 3 + a Metal
System Trace (`docs/results.md` Phase 3.3 "execution-time reckoning";
ADR 0012 Amendment 5) refute each:

1. **"A net sweeps ~3,000 cells" — wrong by 1–2 orders of magnitude.**
   The spike's arithmetic used a 50×30×2 ≈ 3,000-cell guide bbox. The real
   Hazard3 guide regions at track pitch are **median 4,332, p90 18,237, max
   251,720 cells**. The per-net search space the projection divided by was
   optimistic by 1–80×, so the "~0.16 ms/net" never had a basis. Measured
   per-net cost is **32–390 ms/net**, not 0.16.

2. **"The GPU's 7–13× speedup compensates" — invisible end-to-end.**
   That speedup is real for the *sweep primitive in isolation* (ADR 0012
   Am4: 2.46–4.05× batched). But the end-to-end router is **GPU ~5%
   utilised, 71% pipeline bubbles, 0% GPU-bound** (Metal System Trace of the
   sample-1000 route). The per-net CPU backtrace (`.cpu()` + `.item()`
   serialisation) and the eager per-iteration dispatch sync starve the GPU.
   The primitive's win never reaches the router; the workload is
   latency/bubble-bound, not compute-bound.

3. **"Batching many small sub-grids parallelises well" — counterproductive.**
   Round-batching pads every net in a round to the batch's largest sub-grid,
   so one 251k-cell net inflates the batch tensor to ~1 GB and taxes every
   host↔device transfer. Per-net cost is **non-monotonic and rises with
   batch size** (69 → 32 → 391 ms/net at sample 20/100/1000). At realistic
   scale round-batching is **~9.8× slower than plain sequential routing** and
   **~11× slower than drt** whole-chip (~134 min vs ~12 min). Batching, the
   projected source of the win, is the source of the regression.

The deeper lesson: ADR 0012 Amendment 4 validated the **sweep primitive** and
explicitly bounded the claim ("the decision is about the sweep primitive, not
end-to-end router throughput"). We treated that caveat as a formality and let
a primitive-level microbenchmark stand in for an end-to-end projection. The
caveat was the whole story.

## Decision

1. **Pause E5 detailed-routing throughput work on MPS at the end of Slice 3.**
   Do not build Slices 4 (rip-up), 5 (tail), 6 (chip-scale gate) to chase an
   MPS throughput win. On a design this size MPS cannot beat a mature
   multithreaded CPU router (drt: ~12 min, DRC-clean, 0 violations), and the
   profile shows it cannot even beat our own *sequential* MPS routing.

2. **Keep all correctness-validated work.** Slices 1–3 code stays (tests
   green); the sweep primitive, guide-constrained model, track pitch, and
   `drt_compare.py` harness are all sound and are the substrate for any future
   CUDA attempt. This is a pause, not a revert.

3. **Record E5-on-MPS's positive results honestly alongside the negative.**
   The *quality* story is good: on the nets it routes, wirelength is ~1.00×
   drt and vias ~0.45× (`docs/results.md` Phase 3.3). E5 routes competitively;
   it just can't do so *fast* on MPS. The thesis "GPU detailed routing is
   viable" is not refuted — only "MPS delivers it on a small core" is.

## What E5 would need to win (and why that means CUDA, not MPS)

The two named fixes both point off MPS:

- **Kill the pipeline bubbles** (the larger lever — GPU idle 95%): on-GPU
  backtrace + convergence-masking, and a non-eager dispatch model (CUDA
  graphs) to remove the per-iteration sync. PyTorch-MPS eager execution makes
  this hard; CUDA graphs make it natural.
- **Size-bucketing** to stop padding small nets to the batch max.

Both are CUDA-shaped. ADR 0001 always kept MPS as the *development* host and
CUDA as the production target; this measurement makes that split concrete:
**E5's throughput case is a CUDA experiment, not an MPS one.**

## Follow-up experiments (against the original E1–E5 list)

The [`../architecture.md`](../architecture.md) open question was "E1 (cuOpt)
vs E5-on-CUDA when CUDA returns." Slice 3 informs it:

- **E5-on-CUDA** — the cleanest test of whether the *one untested variable*
  (CUDA bandwidth + CUDA-graph dispatch + on-GPU backtrace) actually delivers
  the projected win. Highest information value *if* CUDA hardware is the
  priority; but note the projection has now been wrong once, so this should be
  gated on a **primitive-level CUDA bubble measurement** (does CUDA-graph
  dispatch get GPU utilisation above ~50% on the round structure?) *before*
  re-committing to Slices 4–6. Don't repeat the "microbenchmark → end-to-end
  projection" mistake.
- **E1 (cuOpt MILP track assignment)** — a fundamentally different formulation
  that does **not** have the per-net-backtrace bubble problem (it is a batched
  optimisation solve, not a per-net maze with CPU reconstruction). Given that
  the bubble, not the sweep, is what sank E5-on-MPS, E1's structure is
  arguably a better fit for GPU throughput. Also gates on CUDA.
- **MPS-viable now (no CUDA):** the highest-value MPS work is **not** more E5
  routing throughput. Options that are MPS-shaped:
  - **E2 (GPU pin-access analysis)** — pin access is a per-cell stencil over
    fixed geometry (drt spends ~2 min/2m07s CPU on it); embarrassingly
    parallel, no per-net backtrace bubble, plausibly a real MPS win.
  - **Finish E5's *quality* story** (not throughput): use the existing router
    + `drt_compare.py` to land a defensible "wire/via competitive with drt"
    claim on the full in-cap set, as a correctness result, explicitly
    divorced from the speed claim.
- **E3 (differentiable DR) / E4 (RL ordering + GPU maze)** — unchanged;
  longer-horizon, not informed either way by this measurement.

**Recommendation:** when CUDA returns, run a **primitive-level CUDA dispatch
probe** first (1 day) to decide E5-on-CUDA vs E1 on evidence rather than
preference. Until then, if E-series work continues on MPS, prefer **E2** over
more E5 throughput. Do not resume WS3.3 Slices 4–6 on MPS.

## Consequences

- WS3.3 (`docs/plans/phase3-detailed-routing.md`) moves to **Paused**; the
  Slice 4–6 build (`ws33-tile-router-implementation.md`) is **Closed
  (paused)**, not deleted. Its exit criteria are explicitly not met and not
  being pursued on MPS.
- Phase 3 cannot reach its current exit criteria (WS3.3 whole-chip drt
  competitiveness *on MPS*) as written; the throughput half of that criterion
  is reframed as a CUDA-gated successor experiment.
- The `gpu-vs-drt-throughput.md` spike is **Resolved (negative)** — its
  projection is superseded by measurement.
- The WS3.3 handoff is folded into this ADR + ADR 0012 Am5 + the plans and
  removed (its work is paused, not in-flight).
- ADR 0012 (the guide-constrained design) stays valid as the *design* of E5;
  Amendment 5 already records the throughput walk-back. Nothing in ADR 0012 is
  reverted.

## Amendment 1 (2026-06-03): pause reversed — the collapse was padding, and padding has a cheap MPS fix

The pause above rested on the round-batched router measuring 262–391 ms/net at
sample 1000. That measurement was on the **un-bucketed** router — one giant
batch per round, padding every net to the round's largest sub-grid.
[`../spikes/size-bucketed-batching.md`](../spikes/size-bucketed-batching.md)
size-buckets the round (group similar-sized nets, pad within a bucket only) and
measures the result:

- **End-to-end: 262–391 → 11.76 ms/net (22–33× faster), routing bit-identical**
  (same 853/935 routed, same 1587 deferred conflicts; pinned by a test). The
  bucketed MPS router is **3.4× faster than sequential**. **vs drt: still ~3.8×
  *slower* at comparable work** — bucketing closed a ~100× gap, it did not make
  us faster than drt. (An earlier draft claimed "~2.5× faster than drt"; that
  divided drt's full DRC-clean run by net count vs our single dirty pass —
  corrected in `docs/results.md` "drt performance — the honest like-for-like":
  drt's *initial route* is 3.07 ms/net vs our 11.76, multi-threaded CPU.)
- Sweep-level: bucketing cuts padding waste 31× → 1.0× and the sweep 70×
  (310.88 → 4.42 ms/net), beating sequential 4.9×.

**This reverses the pause.** The decision rests below on three "falsified
assumptions"; the spike shows two of the three were **measurement artifacts of
the giant batch**, not properties of the approach:

1. *"Real guide regions are 4k–252k cells, not ~3k"* — still true, but **not
   fatal**: bucketing means a 252k-cell net no longer pads the 4k-cell nets
   beside it. Heterogeneity is handled, not poisonous.
2. *"GPU ~5%, 71% pipeline bubbles, backtrace-bound"* — **wrong cause.** That
   profile was a 35 s window of the 365 s run; the giant-batch *sweep* alone was
   ~290 s of it (a ~1 GB padded tensor thrashing MPS reads as "bubbles"). The
   router was ~80% padding, ~20% backtrace.
3. *"Batching is counterproductive"* — **inverted.** Batching is the win *when
   bucketed*; the giant all-nets batch was the disease, not batching.

**The honest meta-lesson:** ADR 0013 committed the very error it accused the
gpu-vs-drt spike of — drawing an end-to-end conclusion from an incomplete
measurement (a 35 s profile window + the un-bucketed router). "Verify, don't
assume" applies to walk-backs too. The decision to *pause and write up* was
cheap and correct; the *conclusion* ("MPS can't win, fixes are CUDA-shaped")
was premature by one optimization.

### What changes

- **WS3.3 resumes on MPS.** The throughput basis for pausing is gone — the
  collapse was a fixable padding bug, not an MPS ceiling. Slices 4 (rip-up), 5
  (tail), 6 (chip-scale gate) proceed as correctness work. (Note: the bucketed
  router is ~3.8× *slower* than drt's comparable initial pass — see
  `docs/results.md` — so "resume," not "we win on speed.")
- **Size-bucketing is a WS3.3 deliverable**, not a deferred lever (it is the
  difference between 4 min and 134 min). The `bucket_size=None` giant-batch path
  stays only as A/B scaffolding until a default is chosen.
- **The next throughput lever is the per-net CPU backtrace** (~60% of the
  bucketed router's time). Within uniform-shaped buckets the best-pin argmin
  vectorises (gather + `torch.min`); on-GPU backtrace is the step beyond — *that*
  part of the "CUDA-shaped" framing survives, scoped to backtrace only.

### What still stands from the original decision

- The **critical-learnings retrospective** (the gpu-vs-drt projection was wrong
  on per-net search-space arithmetic) — bucketing fixes the *consequence*, but
  the projection's ~0.16 ms/net was still baseless; the real number is ~12.
- The **CUDA follow-up experiments** (E5-on-CUDA / E1 cuOpt, gated on a dispatch
  probe; E2 pin-access as MPS-viable) — unchanged; CUDA remains the path to a
  *decisive* win and to the backtrace fix. E5-on-MPS is now "competitive," which
  raises the bar for what CUDA must beat.

## Links

- [`../spikes/size-bucketed-batching.md`](../spikes/size-bucketed-batching.md)
  — the reversal measurement (Amendment 1).
- [`../spikes/gpu-vs-drt-throughput.md`](../spikes/gpu-vs-drt-throughput.md)
  — the falsified projection (now Resolved-negative).
- [ADR 0012](0012-tile-decomposition.md) Amendment 5 — the throughput
  walk-back measurement this decision rests on.
- [`../results.md`](../results.md) Phase 3.3 — execution-time + profiling data.
- [ADR 0001](0001-pytorch-mps-host.md) — MPS-as-host / CUDA-as-production; this
  ADR makes the throughput half of that split concrete for E5.
- [`../architecture.md`](../architecture.md) — the E1–E5 experiment list.
