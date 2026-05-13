#!/usr/bin/env python3
"""Profile one sweep_sssp_3d call on a sub-region of the Hazard3
chip-scale grid using PyTorch's native profiler.

The full chip-scale grid (5 x 8823 x 8644) requires ~25-30 GB of MPS
memory for sweep state (12 grid-sized tensors in the precomputed
_ScanState, plus the cost grid, distance map, and per-axis masks).
That OOMs on machines with the standard 30 GB MPS cap. This script
defaults to a 4400 x 4300 sub-region (half-chip extent) which fits
comfortably and has the same per-iter cost characteristics for
profiling purposes.

Uses `torch.profiler.profile` with CPU + MPS activities so we get a
per-operator breakdown of where time goes inside one outer iteration.
xctrace's "Metal System Trace" template has a buffer-size limit that
truncates traces to ~1s, which doesn't fit our ~15-second sweep runs.
The PyTorch profiler doesn't have that limit and produces a clean
operator-level table.

Run: uv run python scripts/profile_chip_sweep.py [MAX_ITERS] [H] [W]
  MAX_ITERS defaults to 8 (each outer iter = 4 axis sweeps + via relax).
  H and W default to 4400 / 4300 (half-chip).
"""

from __future__ import annotations

import sys
import time

import torch
import torch.profiler

from _hazard3_io import (
    FINAL_DEF,
    GUIDE,
    build_chip_grid,
    parse_def_diearea,
    parse_guides,
)
from gpu_pnr.sweep import sweep_sssp_3d


def main() -> None:
    max_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sub_h = int(sys.argv[2]) if len(sys.argv) > 2 else 4400
    sub_w = int(sys.argv[3]) if len(sys.argv) > 3 else 4300

    print("[setup] loading guides...", flush=True)
    t0 = time.perf_counter()
    all_nets = parse_guides(GUIDE)
    print(f"[setup]   {len(all_nets)} nets parsed in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    print("[setup] parsing DIEAREA...", flush=True)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    print(f"[setup]   DIEAREA ({xlo},{ylo})-({xhi},{yhi})", flush=True)

    print("[setup] building chip-scale cost grid (CPU)...", flush=True)
    t0 = time.perf_counter()
    w_full = build_chip_grid(all_nets, xlo, ylo, xhi, yhi)
    print(f"[setup]   full shape={tuple(w_full.shape)} built in "
          f"{time.perf_counter() - t0:.1f}s", flush=True)

    sub_h = min(sub_h, w_full.shape[1])
    sub_w = min(sub_w, w_full.shape[2])
    w_cpu = w_full[:, :sub_h, :sub_w].contiguous()
    print(f"[setup]   sliced to shape={tuple(w_cpu.shape)} "
          f"({w_cpu.numel() * 4 / 1e9:.2f} GB)", flush=True)
    del w_full

    print("[setup] transferring to MPS...", flush=True)
    t0 = time.perf_counter()
    w = w_cpu.to("mps")
    torch.mps.synchronize()
    print(f"[setup]   on {w.device} in {time.perf_counter() - t0:.1f}s",
          flush=True)
    del w_cpu

    source = (0, w.shape[1] // 2, w.shape[2] // 2)
    if not torch.isfinite(w[source]).item():
        finite_mask = torch.isfinite(w[0]).cpu()
        idx = finite_mask.nonzero()[0]
        source = (0, int(idx[0]), int(idx[1]))
        print(f"[setup] center not finite; using {source}", flush=True)

    print("[warmup] one sweep with max_iters=1 to compile MPS kernels...",
          flush=True)
    t0 = time.perf_counter()
    _, _ = sweep_sssp_3d(w, source, via_cost=5.0, max_iters=1,
                          check_every=1)
    torch.mps.synchronize()
    print(f"[warmup]   done in {time.perf_counter() - t0:.1f}s", flush=True)

    print(f"\n[profile] sweep_sssp_3d max_iters={max_iters} source={source}",
          flush=True)
    t0 = time.perf_counter()
    activities = [torch.profiler.ProfilerActivity.CPU]
    # MPS support in torch.profiler varies by torch version; gracefully
    # fall back to CPU-only if it isn't available.
    if hasattr(torch.profiler.ProfilerActivity, "MPS"):
        activities.append(torch.profiler.ProfilerActivity.MPS)
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        with_stack=False,
    ) as prof:
        d, n_iters = sweep_sssp_3d(
            w, source, via_cost=5.0, max_iters=max_iters,
            check_every=max(1, max_iters),
        )
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    finite_count = int(torch.isfinite(d).sum().cpu().item())
    print(
        f"[profile] done in {elapsed:.2f}s "
        f"({1000 * elapsed / max(1, n_iters):.0f}ms/iter), "
        f"n_iters={n_iters}, finite_cells={finite_count}",
        flush=True,
    )

    print("\n=== Profiler: top 30 ops by self CPU time ===", flush=True)
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=30,
    ))


if __name__ == "__main__":
    main()
