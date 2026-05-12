"""Compare K sequential SSSP calls vs one K-batched multi-source SSSP.

Same grid, same K random source positions. Reports wall-clock for each
and the speedup. Demonstrates the GPU throughput potential of
sweep-sharing. Supports both 2D (sweep_sssp_multi) and 3D
(sweep_sssp_3d_multi) modes; the 3D mode is the one Phase 3.3 tile
decomposition will use.
"""

from __future__ import annotations

import argparse
import random
import time

import torch

from gpu_pnr.sweep import (
    sweep_sssp,
    sweep_sssp_3d,
    sweep_sssp_3d_multi,
    sweep_sssp_multi,
)


def _sync(device: str) -> None:
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument(
        "--mode", choices=("2d", "3d"), default="2d",
        help="2d: sweep_sssp_multi; 3d: sweep_sssp_3d_multi at L layers"
    )
    p.add_argument(
        "--layers", type=int, default=5,
        help="number of metal layers for 3d mode (gf180mcuD = 5)"
    )
    p.add_argument(
        "--via-cost", type=float, default=5.0,
        help="via cost for 3d mode"
    )
    args = p.parse_args()

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    if args.mode == "3d":
        print(
            f"3D grid: {args.layers} layers x {args.size}x{args.size}  "
            f"via_cost={args.via_cost}  device={device}"
        )
    else:
        print(f"2D grid: {args.size}x{args.size}  device={device}")
    print()

    # Warm-up.
    if args.mode == "3d":
        w_warm = torch.ones(args.layers, 64, 64, device=device)
        _ = sweep_sssp_3d(w_warm, (0, 0, 0), via_cost=args.via_cost)
        _ = sweep_sssp_3d_multi(
            w_warm, [(0, 0, 0), (0, 10, 10)], via_cost=args.via_cost
        )
    else:
        w_warm = torch.ones(64, 64, device=device)
        _ = sweep_sssp(w_warm, (0, 0))
        _ = sweep_sssp_multi(w_warm, [(0, 0), (10, 10)])
    _sync(device)

    rng = random.Random(args.seed)
    if args.mode == "3d":
        w = torch.ones(args.layers, args.size, args.size, device=device)
        all_sources = [
            (
                rng.randrange(args.layers),
                rng.randrange(args.size),
                rng.randrange(args.size),
            )
            for _ in range(max(args.ks))
        ]
    else:
        w = torch.ones(args.size, args.size, device=device)
        all_sources = [
            (rng.randrange(args.size), rng.randrange(args.size))
            for _ in range(max(args.ks))
        ]

    print(
        f"{'K':>4}  {'sequential_ms':>14}  {'multi_ms':>10}  "
        f"{'speedup':>8}  {'ms/source_seq':>14}  {'ms/source_mul':>14}"
    )
    print("-" * 80)
    for K in args.ks:
        sources = all_sources[:K]

        _sync(device)
        t0 = time.perf_counter()
        if args.mode == "3d":
            for src in sources:
                _ = sweep_sssp_3d(w, src, via_cost=args.via_cost)
        else:
            for src in sources:
                _ = sweep_sssp(w, src)
        _sync(device)
        t_seq = (time.perf_counter() - t0) * 1000.0

        _sync(device)
        t0 = time.perf_counter()
        if args.mode == "3d":
            _ = sweep_sssp_3d_multi(w, sources, via_cost=args.via_cost)
        else:
            _ = sweep_sssp_multi(w, sources)
        _sync(device)
        t_mul = (time.perf_counter() - t0) * 1000.0

        speedup = t_seq / t_mul if t_mul > 0 else float("inf")
        print(
            f"{K:>4}  {t_seq:>14.1f}  {t_mul:>10.1f}  "
            f"{speedup:>7.2f}x  {t_seq/K:>14.2f}  {t_mul/K:>14.2f}"
        )


if __name__ == "__main__":
    main()
