# Spike — Batched small-grid sweep: does packing K independent sub-grids amortise per-net overhead?

**Status:** Resolved YES (2026-05-29). Batching K independent per-net
sub-grids into one kernel call is **2.46–4.05× faster than sequential on
MPS** at track pitch. This resolves the load-bearing open hypothesis of the
[slot-scale-parallelism spike](slot-scale-parallelism.md) and of
[ADR 0012](../adr/0012-tile-decomposition.md) Amendment 1 §3.

## Question

The [track-pitch prototype](../results.md) (Phase 3.3) measured single-stream
per-net routing as **overhead-bound**: MPS 16.8 ms/net median vs CPU 2.57,
because a ~4k-cell sub-grid starves a 2,560-ALU GPU — fixed kernel-launch +
convergence-sync cost swamps the tiny compute. The slot-scale spike named the
fix and flagged it as the one unproven, load-bearing claim:

> *Can we batch the tiny sweeps to amortise launch overhead and reach the
> bandwidth ceiling?*

Concretely: batch K **independent** small sub-grids (one per net) into a single
kernel call, vs K sequential single-grid sweeps. Does the GPU win?

This is a different bet from the dead Tier-B K-batching
([tier-b-envelope-throughput](tier-b-envelope-throughput.md)): that batched K
*sources* over one *shared* large grid and lost past 256². This batches K
*different small* grids — each independent, each cache-resident.

## Method

`src/gpu_pnr/sweep.py::sweep_sssp_3d_batched` — mirrors `sweep_sssp_3d_multi`
but takes a true per-net `w` of shape `(K, L, H, W)` (not one grid broadcast
across K sources) with one source per net and a single batch-wide
`seg_barrier`. Correctness: 4 tests in `tests/test_sweep_3d.py` assert the
batched result equals K separate `sweep_sssp_3d` runs (same-size, variable-size
padded, anisotropic) and that inf padding stays inf.

`scripts/batched_sweep_prototype.py` — over a sample of routable, in-cap
Hazard3 nets at the track pitch (1120 DBU): map guides → `guide_region`
sub-grid, slice an independent sub-grid, then time the **single-source distance
sweep** two ways:

- **Sequential** — K separate `sweep_sssp_3d` calls.
- **Batched** — pad the K sub-grids to a common shape (inf pad = wall), stack to
  `(K, L, H, W)`, one `sweep_sssp_3d_batched` call.

H2D transfer and pad/stack host cost are **excluded** from the timed window, so
the measurement isolates GPU compute + launch + sync. The single-source sweep
(no backtrace, no multi-pin tree growth) is the unit, matching the slot-scale
spike's framing. `--sort-by-size` batches similar-sized nets (low padding
waste); the default shuffled order is the realistic-variance case.

Sample: 128 in-cap nets, seed 0, K=16, M4 Pro. Sub-grid cells median 4,332,
p90 26,478, max 162,192 (matches the Phase 3.3 size distribution).

## Result — batching wins on GPU, loses on CPU

| Device | Batch order | sequential ms/net | batched ms/net | speedup | padding waste (mean) |
|---|---|---:|---:|---:|---:|
| **MPS** | shuffled | 27.4 | **11.1** | **2.46×** | 8.3× |
| **MPS** | size-sorted | 28.1 | **6.9** | **4.05×** | 3.2× |
| CPU | shuffled | 5.5 | 36.3 | 0.15× | 8.3× |
| CPU | size-sorted | 6.4 | 15.7 | 0.41× | 3.2× |

(ms/net is the overall total ÷ net count; medians track the same direction.
Correctness check on real data: max |Δdist| between batched and sequential = 0.)

Three findings:

1. **The hypothesis holds: batching amortises the per-net overhead.** MPS
   batched is 2.46× (shuffled) to 4.05× (sorted) faster than sequential. The
   per-net kernel-launch + convergence-sync cost — the binding constraint the
   track-pitch prototype found — *is* amortisable by packing many tiny sweeps
   into few launches. Batched MPS (3.8 ms/net median, sorted) is now
   competitive with CPU sequential, closing most of the GPU's overhead gap.

2. **The win is GPU-specific — which confirms the diagnosis.** CPU is
   *slower* batched (0.15–0.41×): with no kernel-launch overhead to amortise,
   batching only adds cost. Note batching does **more total work** than
   sequential (padding cells + every net runs to the slowest net's iteration
   count — all batches ran 16 iters regardless of per-net diameter). That MPS
   wins *despite* doing more work is direct evidence the bottleneck was
   overhead, not compute.

3. **Padding waste is the lever, and it sets the option-A/B verdict.**
   Size-sorting cut padding waste 8.3× → 3.2× and lifted the MPS win 2.46× →
   4.05×. So **option A (plain padded stack) already wins**, and **option B
   (size bucketing) is a worthwhile ~1.6× further gain, not a prerequisite** —
   exactly the "A now, B/C deferred as tail-optimisation" framing of ADR 0012
   Amendment 3.

## Verdict

**Resolved YES.** The batched small-grid sweep is the real GPU-parallelism win
for guide-constrained routing, as the slot-scale spike projected. On MPS it
reverses the track-pitch prototype's "CPU beats GPU at this grain" result for
the sweep step. The whole-chip single-source extrapolation drops from ~560 s
sequential to ~143 s batched (sorted) on M4 Pro — a real 2.5–4× before any of
the further levers below, and CUDA's ~10× bandwidth makes batching essential
just to feed the ALUs there.

## What this does NOT yet show / next levers

- **Still overhead-bound, not bandwidth-bound.** ~143 s batched is far above
  the 31 s linear ideal. The "slowest net bounds the batch" (uniform 16 iters)
  and residual padding waste remain. Two levers: **convergence-masking** (let
  converged nets drop out of the batch instead of iterating to the slowest) and
  **option-B size bucketing**.
- **Sweep only — no backtrace / multi-pin / rip-up.** The unit is one
  single-source distance sweep. Real routing adds per-net backtrace (CPU),
  multi-pin tree growth (multiple sweeps), and cross-net conflict + rip-up on
  the shared `w_cur`. Nets here are independent by construction; conflict
  handling is upstream and unmodelled.
- **Cross-prototype number caveat.** The sequential single-source figure here
  (~27 ms/net MPS) is *not* directly comparable to the track-pitch prototype's
  full-route 16.8 ms/net — different unit, and each single sweep pays the
  `seg_barrier` autotune's ~3 syncs (which the batched kernel amortises across
  K too). The load-bearing result is the **ratio** within this one harness.
- **Small sample.** 128 nets / 8 batches; means are tail-sensitive (the
  162k-cell max net), medians robust. Direction is consistent across all four
  cells of the matrix.

## Reproduce

```sh
uv run pytest tests/test_sweep_3d.py -k batched           # 4 correctness tests
uv run python scripts/batched_sweep_prototype.py --device mps --sample 128 --batch 16
uv run python scripts/batched_sweep_prototype.py --device mps --sample 128 --batch 16 --sort-by-size
uv run python scripts/batched_sweep_prototype.py --device cpu --sample 128 --batch 16   # GPU-specificity
```

## References

- [slot-scale-parallelism spike](slot-scale-parallelism.md) — posed this
  hypothesis as its load-bearing open question; now resolved here.
- [ADR 0012](../adr/0012-tile-decomposition.md) Amendment 1 §3 (batched
  small-grid sweep) and Amendment 3 (track pitch; A-now/B-C-deferred framing).
- [tier-b-envelope-throughput spike](tier-b-envelope-throughput.md) — the
  *different*, dead bet (K sources on one big grid) this contrasts with.
- [`../results.md`](../results.md) Phase 3.3 — track-pitch prototype that found
  single-stream overhead-bound and motivated this spike.
