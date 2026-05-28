#!/usr/bin/env python3
"""WS3.3 track-pitch sweep prototype — ADR 0012 Amendment 3 validation.

Amendment 3 found our cost tensor is sampled at 200 DBU while gf180mcuD
routing tracks sit at 1120 DBU (5.6×/axis over-sampling). Re-measuring
guide-region *sizes* at the track pitch validated Amendment 1's throughput
model on paper (median 4,332 cells, 0.24 ms/net *linear*). This prototype
turns the paper estimate into a measurement and forces the one open
decision the size measurement could not: pin access on a coarse grid.

It does two independent things:

  PART A — pin-access geometry (measure-first; Amendment 3 open question #1).
    Quantize every routable net's M1 pins at both 200 DBU and the track
    pitch and count where coarsening breaks pin identity:
      - intra-net merge: ≥2 distinct pins of one net collapse to the same
        (layer,row,col) cell — that net silently loses a terminal.
      - cross-net collision: one grid cell is claimed by pins of ≥2 nets —
        exactly the failure that sank 21/27 nets in the 200 DBU tile
        prototype (ADR 0012 §Prototype findings). A 5.6× coarser grid can
        only make this worse; this quantifies by how much, so the
        snapping-vs-local-fine-region decision (a future ADR 0012
        amendment) is data-driven, not guessed.

  PART B — real per-net throughput at track pitch.
    Build the chip-scale cost grid at the track pitch, then for a random
    sample of routable nets: map guides → `guide_region` sub-grid, slice an
    independent sub-grid, and route it with `route_multipin_nets_3d`.
    Reports *measured* ms/net (kernel launch + convergence sync included),
    which the 0.24 ms/net linear extrapolation omits — on ~4k-cell grids
    that fixed overhead is expected to dominate. Compares against the 54
    ms/net 200 DBU densest-tile baseline (ADR 0012 §Prototype findings).

Each net routes on its *own* freshly-sliced sub-grid (the guide-constrained
model: per-net search space, no shared obstacle tensor). Cross-net conflict
handling and the batched small-grid kernel are later work
(slot-scale-parallelism spike); this prototype isolates the per-net cost.

Run:
  uv run python scripts/track_pitch_sweep_prototype.py            # pitch 1120, auto device
  uv run python scripts/track_pitch_sweep_prototype.py --device cpu --sample 200
  uv run python scripts/track_pitch_sweep_prototype.py --pitch 200   # A/B the over-sampled grid
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

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
    rect_center_to_grid,
)

from gpu_pnr.guides import guide_region
from gpu_pnr.router import MultiPin3DResult, route_multipin_nets_3d

logger = logging.getLogger(__name__)

PDK = GF180MCUD
# Same routable filter as the tile prototype / size measurement: 2..20 M1
# pins (drops single-pin nets and >20-pin clock/power nets).
MIN_PINS = 2
MAX_PINS = 20
# Nets whose H or W exceed this need the coarsened-pass fallback
# (ADR 0012 Amendment 1 §7), not a direct sub-grid sweep — out of scope here.
SUBGRID_AXIS_CAP = 256
# gf180mcuD M1-M4 routing-track pitch (ADR 0012 Amendment 3).
TRACK_PITCH_DBU = 1120
# 200 DBU densest-tile sequential baseline (ADR 0012 §Prototype findings).
TILE_BASELINE_MS = 54.0
# Routable-net population at track pitch (docs/results.md Phase 3.3), for the
# whole-chip total-time extrapolation.
HAZARD3_ROUTABLE_NETS = 20524


def _net_pins(
    rects: list[tuple[int, int, int, int, str]],
    chip_origin: tuple[int, int],
    pitch: int,
) -> list[tuple[int, int, int]]:
    """Metal1 pin cells of a net at the given grid pitch.

    Unlike `tile_decomp_prototype._net_chip_pins` (hardcoded 200 DBU), this
    honours `pitch` so pins land on the same grid the sub-grid is sliced from.
    """
    return [
        rect_center_to_grid(r, chip_origin, pitch_dbu=pitch)
        for r in rects
        if r[4] == "Metal1"
    ]


@dataclass
class SampleResult:
    """PART B per-net routing outcome over the sampled nets."""

    attempted: int = 0
    routed: int = 0
    skipped_none: int = 0
    over_cap: int = 0
    off_region: int = 0
    pin_on_inf: int = 0
    pin_merge_fail: int = 0
    route_fail: int = 0
    per_net_ms: list[float] = field(default_factory=list)
    cell_counts: list[int] = field(default_factory=list)


def _pct(sorted_vals: Sequence[float], q: float) -> float:
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def measure_pin_access(
    all_nets: dict[str, list[tuple[int, int, int, int, str]]],
    chip_origin: tuple[int, int],
    pitch: int,
) -> dict[str, int]:
    """PART A: pin-identity breakage when M1 pins are quantized at `pitch`."""
    cell_owners: dict[tuple[int, int, int], set[str]] = defaultdict(set)
    n_routable = 0
    n_intra_merge = 0
    for name, rects in all_nets.items():
        pins = _net_pins(rects, chip_origin, pitch)
        if not (MIN_PINS <= len(pins) <= MAX_PINS):
            continue
        n_routable += 1
        if len(set(pins)) < len(pins):
            n_intra_merge += 1
        for cell in set(pins):
            cell_owners[cell].add(name)
    contested = {c: o for c, o in cell_owners.items() if len(o) > 1}
    nets_colliding = {name for owners in contested.values() for name in owners}
    return {
        "pitch": pitch,
        "n_routable": n_routable,
        "intra_merge": n_intra_merge,
        "contested_cells": len(contested),
        "nets_colliding": len(nets_colliding),
    }


def route_sample(
    all_nets: dict[str, list[tuple[int, int, int, int, str]]],
    chip_origin: tuple[int, int],
    chip_shape: tuple[int, int, int],
    w_chip: torch.Tensor,
    pitch: int,
    device: str,
    margin: int,
    sample: int,
    seed: int,
) -> SampleResult:
    """PART B: route a random sample of nets, each on its own guide sub-grid."""
    names = list(all_nets.keys())
    random.Random(seed).shuffle(names)

    out = SampleResult()

    # Warm up the device once so the first timed route doesn't eat one-time
    # MPS shader compilation / allocator setup.
    if device == "mps":
        warm = torch.full((2, 8, 8), 1.0).to(device)
        route_multipin_nets_3d(warm, [[(0, 0, 0), (0, 7, 7)]], via_cost=5.0)
        torch.mps.synchronize()

    for name in names:
        if out.attempted >= sample:
            break
        rects = all_nets[name]
        pins = _net_pins(rects, chip_origin, pitch)
        if not (MIN_PINS <= len(pins) <= MAX_PINS):
            continue
        reg = guide_region(
            rects, chip_origin, LAYER_ORDER, pitch,
            margin=margin, chip_shape=chip_shape,
        )
        if reg is None:
            out.skipped_none += 1
            continue
        _, nh, nw = reg.shape
        if nh > SUBGRID_AXIS_CAP or nw > SUBGRID_AXIS_CAP:
            out.over_cap += 1
            continue
        if not all(reg.contains(p) for p in pins):
            out.off_region += 1
            continue

        out.attempted += 1
        out.cell_counts.append(reg.cell_count)
        local = [reg.rebase(p) for p in pins]

        # Routable nets have M1 pins, so the region is M1-anchored (l0==0) and
        # sub-grid layer 0 == M1 — the layer apply_pin_access_rules forces
        # pin-access-only. The rebased pins below rely on this.
        assert reg.l0 == 0, f"expected M1-anchored region, got l0={reg.l0}"

        # Independent sub-grid: clone so PDK mutation / routing never touches
        # the shared chip tensor (this prototype routes each net in isolation).
        w_sub = w_chip[reg.l0:reg.l1, reg.r0:reg.r1, reg.c0:reg.c1].clone()
        apply_pin_access_rules(w_sub, PDK, local)
        w_dev = w_sub.to(device)

        if device == "mps":
            torch.mps.synchronize()
        t0 = time.perf_counter()
        res: MultiPin3DResult = route_multipin_nets_3d(
            w_dev, [local], via_cost=5.0, net_timeout_s=30.0,
        )[0]
        if device == "mps":
            torch.mps.synchronize()
        out.per_net_ms.append((time.perf_counter() - t0) * 1000.0)

        if res.routed:
            out.routed += 1
        elif not all(torch.isfinite(w_dev[p]) for p in local):
            out.pin_on_inf += 1
        elif len(set(local)) < len(local):
            out.pin_merge_fail += 1
        else:
            out.route_fail += 1

    return out


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pitch", type=int, default=TRACK_PITCH_DBU,
        help="grid pitch in DBU (default 1120 = gf180mcuD track pitch)",
    )
    p.add_argument(
        "--device", type=str, default="auto",
        help="auto | mps | cpu (default auto: mps if available)",
    )
    p.add_argument("--sample", type=int, default=100,
                   help="number of in-cap nets to route for timing")
    p.add_argument("--margin", type=int, default=4, help="guide_region margin (cells)")
    p.add_argument("--seed", type=int, default=0, help="sample shuffle seed")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if args.device == "auto":
        device = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device = args.device

    all_nets = parse_guides(GUIDE)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    chip_origin = (xlo, ylo)
    chip_h = (yhi - ylo) // args.pitch + 1
    chip_w = (xhi - xlo) // args.pitch + 1
    chip_shape = (len(LAYER_ORDER), chip_h, chip_w)

    print(
        f"Track-pitch sweep prototype — pitch {args.pitch} DBU, device {device}",
        flush=True,
    )
    print(
        f"  chip {chip_shape[0]}×{chip_h}×{chip_w} cells; "
        f"{len(all_nets)} total nets", flush=True,
    )

    # --- PART A: pin-access geometry, 200 DBU vs the chosen pitch ---
    print("\n=== PART A — pin access (geometry only) ===", flush=True)
    print("  NB: the guide fixture gives GCell-granular M1 rects "
          "(16,800 DBU = 1 GCell), not", flush=True)
    print("  DEF pin shapes; we use rect centers as pin proxies (as the tile "
          "prototype did).", flush=True)
    pitches = [200] if args.pitch == 200 else [200, args.pitch]
    pa = [measure_pin_access(all_nets, chip_origin, p) for p in pitches]
    hdr = f"  {'metric':<24}"
    for r in pa:
        hdr += f"{str(r['pitch']) + ' DBU':>12}"
    print(hdr, flush=True)
    for label, key in [
        ("routable nets", "n_routable"),
        ("intra-net pin merge", "intra_merge"),
        ("cross-net contested cells", "contested_cells"),
        ("nets w/ contested pin", "nets_colliding"),
    ]:
        row = f"  {label:<24}"
        for r in pa:
            row += f"{r[key]:>12}"
        print(row, flush=True)
    print(
        "\n  Reading: intra-net pin merge (a net's own pins collapsing) is the\n"
        "  pitch-sensitive, routing-relevant metric — coarsening to the track\n"
        "  pitch is safe iff it stays ~0. The cross-net collision count is\n"
        "  pitch-INVARIANT here (two nets sharing a GCell get the same center\n"
        "  at any pitch), so it reflects the GCell-proxy artifact, not a\n"
        "  track-pitch effect. True off-track pin access needs DEF pin\n"
        "  geometry (deferred); the per-net sweep in PART B routes each net on\n"
        "  its own sub-grid, so cross-net pin collisions cannot occur there.",
        flush=True,
    )

    # --- PART B: measured per-net throughput at the chosen pitch ---
    print(f"\n=== PART B — measured throughput at {args.pitch} DBU "
          f"({device}) ===", flush=True)
    t0 = time.perf_counter()
    print("  building chip-scale cost grid...", flush=True)
    w_chip = build_chip_grid(all_nets, xlo, ylo, xhi, yhi, pitch_dbu=args.pitch)
    print(f"    shape {tuple(w_chip.shape)} in {time.perf_counter() - t0:.1f}s",
          flush=True)

    res = route_sample(
        all_nets, chip_origin, chip_shape, w_chip, args.pitch,
        device, args.margin, args.sample, args.seed,
    )
    ms = sorted(res.per_net_ms)
    cells = sorted(res.cell_counts)
    attempted = res.attempted

    print(f"\n  sample: {attempted} in-cap nets routed "
          f"(skipped: {res.over_cap} over-cap, {res.skipped_none} no-guide, "
          f"{res.off_region} off-region)", flush=True)
    if attempted:
        print(f"  routed: {res.routed}/{attempted} "
              f"({100*res.routed/attempted:.1f}%); failures — "
              f"pin_on_inf={res.pin_on_inf} pin_merge={res.pin_merge_fail} "
              f"route_fail={res.route_fail}", flush=True)
        print(f"  sub-grid cells: median={_pct(cells, 0.5):.0f} "
              f"p90={_pct(cells, 0.9):.0f} max={cells[-1]}", flush=True)
        print(f"  measured ms/net: median={_pct(ms, 0.5):.2f} "
              f"mean={sum(ms)/len(ms):.2f} p90={_pct(ms, 0.9):.2f} "
              f"max={ms[-1]:.1f}", flush=True)
        total_s = sum(ms) / len(ms) * HAZARD3_ROUTABLE_NETS / 1000.0
        print(f"  → vs 54 ms/net 200 DBU tile baseline: "
              f"{TILE_BASELINE_MS / (sum(ms)/len(ms)):.1f}× "
              f"({'faster' if sum(ms)/len(ms) < TILE_BASELINE_MS else 'slower'})",
              flush=True)
        print(f"  → whole-chip extrapolation (mean × {HAZARD3_ROUTABLE_NETS} "
              f"routable nets): {total_s:.0f}s single-stream", flush=True)


if __name__ == "__main__":
    main()
