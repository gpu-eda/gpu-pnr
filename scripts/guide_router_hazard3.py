#!/usr/bin/env python3
"""WS3.3 GuideRouter Slice 2 validation — single-stream route on Hazard3.

The Slice 2 exit criterion (`docs/plans/ws33-tile-router-implementation.md`):
run the guide-constrained single-stream router on a Hazard3 sample and confirm
it reproduces the track-pitch prototype's ms/net and **0 cross-net conflicts**.

Unlike `track_pitch_sweep_prototype.py` (which routes each net on an
*independent* clone), this drives `gpu_pnr.guide_router.GuideRouter`, which
routes the sampled nets sequentially on **one shared** chip cost grid in
HPWL-ascending order: each routed net's cells become obstacles for later nets.
The cross-net-conflict-free property is the substrate invariant this validates
(it is what the tile prototype first showed; here it must hold on the
guide-constrained shared grid). Pin-access rules are injected per sub-grid via
the router's `prep_subgrid` hook — the same `apply_pin_access_rules` the
prototype applied, now plumbed through the PDK-agnostic router.

Run:
  uv run python scripts/guide_router_hazard3.py --device cpu --sample 100
  uv run python scripts/guide_router_hazard3.py --device mps --sample 200
  uv run python scripts/guide_router_hazard3.py --pitch 200   # A/B over-sampled
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import Counter

import torch

from _hazard3_io import (
    FINAL_DEF,
    GF180MCUD,
    GUIDE,
    LAYER_ORDER,
    apply_pin_access_rules,
    build_chip_grid,
    parse_def_diearea,
    parse_guides,
)

from gpu_pnr.guide_router import GuideRouter

# Reuse the prototype's net filter, pin mapper, percentile helper, and the
# whole-chip extrapolation constant so this samples an identical population.
from track_pitch_sweep_prototype import (
    HAZARD3_ROUTABLE_NETS,
    MAX_PINS,
    MIN_PINS,
    TRACK_PITCH_DBU,
    _net_pins,
    _pct,
)

PDK = GF180MCUD
VIA_COST = 5.0  # match the track-pitch prototype


def _conflicts(results) -> int:
    """Count cells claimed by more than one routed net (must be 0)."""
    cell_owners: Counter[tuple[int, int, int]] = Counter()
    for res in results:
        for cell in res.cells:
            cell_owners[cell] += 1
    return sum(1 for n in cell_owners.values() if n > 1)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pitch", type=int, default=TRACK_PITCH_DBU,
                   help="grid pitch in DBU (default 1120 = track pitch)")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | mps | cpu (default auto: mps if available)")
    p.add_argument("--sample", type=int, default=100,
                   help="number of routable nets to route on the shared grid")
    p.add_argument("--margin", type=int, default=4, help="guide_region margin (cells)")
    p.add_argument("--seed", type=int, default=0, help="sample shuffle seed")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    device = ("mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device

    all_nets = parse_guides(GUIDE)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    chip_origin = (xlo, ylo)
    chip_h = (yhi - ylo) // args.pitch + 1
    chip_w = (xhi - xlo) // args.pitch + 1
    chip_shape = (len(LAYER_ORDER), chip_h, chip_w)

    print(f"GuideRouter Slice 2 — pitch {args.pitch} DBU, device {device}",
          flush=True)
    print("  building chip-scale cost grid...", flush=True)
    t0 = time.perf_counter()
    w_chip = build_chip_grid(all_nets, xlo, ylo, xhi, yhi, pitch_dbu=args.pitch)
    w_chip = w_chip.to(device)
    print(f"    shape {tuple(w_chip.shape)} in {time.perf_counter() - t0:.1f}s",
          flush=True)

    # Build a deterministic sample of routable nets: (pins, guide_rects) pairs.
    names = list(all_nets.keys())
    random.Random(args.seed).shuffle(names)
    nets: list[list[tuple[int, int, int]]] = []
    guides: list[list[tuple[int, int, int, int, str]]] = []
    for name in names:
        if len(nets) >= args.sample:
            break
        rects = all_nets[name]
        pins = _net_pins(rects, chip_origin, args.pitch)
        if not (MIN_PINS <= len(pins) <= MAX_PINS):
            continue
        nets.append(pins)
        guides.append(rects)

    def prep(w_sub: torch.Tensor, local_pins: list[tuple[int, int, int]]) -> None:
        apply_pin_access_rules(w_sub, PDK, local_pins)

    router = GuideRouter(
        w_chip, chip_origin=chip_origin, layer_order=LAYER_ORDER,
        pitch_dbu=args.pitch, margin=args.margin, chip_shape=chip_shape,
        prep_subgrid=prep,
    )

    # Warm up the device so the first timed route doesn't eat shader compile.
    if device == "mps":
        warm = torch.full((2, 8, 8), 1.0, device=device)
        GuideRouter(
            warm, chip_origin=(0, 0), layer_order=LAYER_ORDER,
            pitch_dbu=args.pitch,
        ).route([[(0, 0, 0), (0, 7, 7)]], [[]])
        torch.mps.synchronize()

    in_cap, tail = router.classify(nets, guides)
    print(f"\n  sample: {len(nets)} routable nets — "
          f"{len(in_cap)} in-cap, {len(tail)} tail (over-cap/no-guide)",
          flush=True)

    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    results = router.route(nets, guides, via_cost=VIA_COST)
    if device == "mps":
        torch.mps.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    routed = [r for r in results if r.routed]
    conflicts = _conflicts(results)
    ms_per_incap = elapsed_ms / max(len(in_cap), 1)

    print(f"\n  routed: {len(routed)}/{len(in_cap)} in-cap "
          f"({100*len(routed)/max(len(in_cap),1):.1f}%); "
          f"tail unrouted: {len(tail)}", flush=True)
    print(f"  CROSS-NET CONFLICTS: {conflicts}  "
          f"({'✓ PASS' if conflicts == 0 else '✗ FAIL'})", flush=True)
    print(f"  ms/net (in-cap, aggregate mean): {ms_per_incap:.2f} "
          f"(total {elapsed_ms/1000:.1f}s for {len(in_cap)} in-cap nets)",
          flush=True)
    lengths = sorted(r.length for r in routed)
    if lengths:
        print(f"  routed wirelength cells: median={_pct(lengths, 0.5):.0f} "
              f"p90={_pct(lengths, 0.9):.0f} max={lengths[-1]}", flush=True)
    total_s = ms_per_incap * HAZARD3_ROUTABLE_NETS / 1000.0
    print(f"  → whole-chip single-stream extrapolation "
          f"({HAZARD3_ROUTABLE_NETS} routable): {total_s:.0f}s", flush=True)
    print("\n  Reading: 0 cross-net conflicts is the substrate invariant "
          "(shared-grid commit works).\n  The aggregate ms/net is a mean "
          "(total / in-cap count); compare to the track-pitch prototype's\n  "
          "MEAN (CPU ~16, MPS ~37 — docs/results.md Phase 3.3), not its "
          "tail-light median.\n  Routability < 100% is honest cross-net "
          "contention (no rip-up yet — Slice 4 recovers it).", flush=True)


if __name__ == "__main__":
    main()
