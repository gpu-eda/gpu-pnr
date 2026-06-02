# ADR 0013 — Pause E5 detailed-routing throughput work on MPS

**Status:** Accepted (2026-06-02).

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

## Links

- [`../spikes/gpu-vs-drt-throughput.md`](../spikes/gpu-vs-drt-throughput.md)
  — the falsified projection (now Resolved-negative).
- [ADR 0012](0012-tile-decomposition.md) Amendment 5 — the throughput
  walk-back measurement this decision rests on.
- [`../results.md`](../results.md) Phase 3.3 — execution-time + profiling data.
- [ADR 0001](0001-pytorch-mps-host.md) — MPS-as-host / CUDA-as-production; this
  ADR makes the throughput half of that split concrete for E5.
- [`../architecture.md`](../architecture.md) — the E1–E5 experiment list.
