#!/usr/bin/env python3
"""Spike — goal-bounded sweep.

Does shrinking each net's sweep region from the full GR-guide bbox to its
**pin bbox + margin** (a goal corridor) cut search-space and time while
preserving routability? This is the search-space lever from
`docs/spikes/gpu-astar-evaluation.md`: A*'s goal-direction is what gives drt its
~3.8× edge (`docs/spikes/size-bucketed-batching.md`), but A* is irregular and
breaks our batch model — bounding the *region* keeps the regular sweep and adds
goal-direction for free.

Each in-cap net is routed two ways on its own sub-grid (independent, so the
only variable is region size): the full `guide_region` vs a goal-bounded region
(full layer stack — vias need it — but row/col tightened to the pin bbox +
margin). Reports cells, ms/net, routability, and wirelength delta.

Caveat: independent sub-grids carry pin-access obstacles but NOT other nets'
committed wires. The routability risk of bounding (a net needing to detour
outside its pin bbox around a committed wire) only shows on the shared grid —
noted as a follow-up if bounding looks promising here.

Run:
  uv run python scripts/goal_bounded_sweep_prototype.py --device mps --sample 300
  uv run python scripts/goal_bounded_sweep_prototype.py --device mps --sample 300 --margin 2
"""

from __future__ import annotations

import argparse
import random
import sys
import time

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

from gpu_pnr.guide_router import net_bbox
from gpu_pnr.guides import GuideRegion, clamp_region_bounds, guide_region
from gpu_pnr.router import route_multipin_nets_3d
from track_pitch_sweep_prototype import (
    MAX_PINS,
    MIN_PINS,
    TRACK_PITCH_DBU,
    _net_pins,
    _pct,
)

PDK = GF180MCUD
VIA_COST = 5.0
PITCH = TRACK_PITCH_DBU
AXIS_CAP = 256


def goal_bounded(
    full: GuideRegion,
    pins: list[tuple[int, int, int]],
    margin: int,
    chip_shape: tuple[int, int, int],
) -> GuideRegion:
    """Full layer stack (vias need it) with row/col tightened to the pin bbox +
    margin — the goal corridor. A subset of `full`, since pins lie inside the
    guide."""
    rmin, cmin, rmax, cmax = net_bbox(pins)
    l0, l1, r0, r1, c0, c1 = clamp_region_bounds(
        full.l0, full.l1,
        rmin - margin, rmax + 1 + margin,
        cmin - margin, cmax + 1 + margin,
        chip_shape,
    )
    return GuideRegion(l0=l0, l1=l1, r0=r0, r1=r1, c0=c0, c1=c1)


def route_on(
    region: GuideRegion,
    pins: list[tuple[int, int, int]],
    w_chip: torch.Tensor,
    device: str,
) -> tuple[float, bool, int, int]:
    """Route the net on its own sub-grid sliced for `region`. Returns
    (ms, routed, wire_cells, region_cells)."""
    rl = (
        slice(region.l0, region.l1),
        slice(region.r0, region.r1),
        slice(region.c0, region.c1),
    )
    w_sub = w_chip[rl].clone()
    local = [region.rebase(p) for p in pins]
    apply_pin_access_rules(w_sub, PDK, local)
    w_dev = w_sub.to(device)
    if device == "mps":
        torch.mps.synchronize()
    t0 = time.perf_counter()
    [res] = route_multipin_nets_3d(w_dev, [local], via_cost=VIA_COST)
    if device == "mps":
        torch.mps.synchronize()
    ms = (time.perf_counter() - t0) * 1000.0
    return ms, res.routed, res.length, region.cell_count


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="auto", help="auto | mps | cpu")
    p.add_argument("--sample", type=int, default=300)
    p.add_argument("--margin", type=int, default=4,
                   help="margin (cells) for BOTH regions, so the comparison is "
                        "guide-extent vs pin-extent at the same margin")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    device = ("mps" if torch.backends.mps.is_available() else "cpu") \
        if args.device == "auto" else args.device

    all_nets = parse_guides(GUIDE)
    xlo, ylo, xhi, yhi = parse_def_diearea(FINAL_DEF)
    chip_origin = (xlo, ylo)
    chip_h = (yhi - ylo) // PITCH + 1
    chip_w = (xhi - xlo) // PITCH + 1
    chip_shape = (len(LAYER_ORDER), chip_h, chip_w)
    print(f"goal-bounded sweep — device {device}, margin {args.margin}", flush=True)
    w_chip = build_chip_grid(all_nets, xlo, ylo, xhi, yhi, pitch_dbu=PITCH).to(device)

    # Sample in-cap nets, keeping (pins, full guide region).
    names = list(all_nets.keys())
    random.Random(args.seed).shuffle(names)
    work: list[tuple[list[tuple[int, int, int]], GuideRegion]] = []
    for nm in names:
        if len(work) >= args.sample:
            break
        rects = all_nets[nm]
        pins = _net_pins(rects, chip_origin, PITCH)
        if not (MIN_PINS <= len(pins) <= MAX_PINS):
            continue
        full = guide_region(
            rects, chip_origin, LAYER_ORDER, PITCH,
            margin=args.margin, chip_shape=chip_shape,
        )
        if full is None:
            continue
        _, nh, nw = full.shape
        if nh > AXIS_CAP or nw > AXIS_CAP:
            continue
        if not all(full.contains(p) for p in pins):
            continue
        work.append((pins, full))

    # Warm up MPS.
    if device == "mps":
        warm = torch.full((2, 8, 8), 1.0, device=device)
        route_multipin_nets_3d(warm, [[(0, 0, 0), (0, 7, 7)]], via_cost=VIA_COST)
        torch.mps.synchronize()

    full_cells: list[int] = []
    bnd_cells: list[int] = []
    full_ms: list[float] = []
    bnd_ms: list[float] = []
    full_routed = bnd_routed = 0
    lost = 0  # full routes but bounded fails
    wl_ratio: list[float] = []
    for pins, full in work:
        bnd = goal_bounded(full, pins, args.margin, chip_shape)
        f_ms, f_ok, f_wl, f_cells = route_on(full, pins, w_chip, device)
        b_ms, b_ok, b_wl, b_cells = route_on(bnd, pins, w_chip, device)
        full_cells.append(f_cells)
        bnd_cells.append(b_cells)
        full_ms.append(f_ms)
        bnd_ms.append(b_ms)
        full_routed += f_ok
        bnd_routed += b_ok
        if f_ok and not b_ok:
            lost += 1
        if f_ok and b_ok and f_wl > 0:
            wl_ratio.append(b_wl / f_wl)

    n = len(work)
    fc, bc = sorted(full_cells), sorted(bnd_cells)
    fm, bm = sorted(full_ms), sorted(bnd_ms)
    wq = sorted(wl_ratio)
    tot_f = sum(full_ms)
    tot_b = sum(bnd_ms)
    print(f"\n  {n} in-cap nets routed two ways ({device})", flush=True)
    print(f"  {'':<20}{'median':>10}{'mean':>10}{'p90':>10}", flush=True)
    print(f"  {'full cells':<20}{_pct(fc,0.5):>10.0f}{sum(fc)/n:>10.0f}"
          f"{_pct(fc,0.9):>10.0f}", flush=True)
    print(f"  {'bounded cells':<20}{_pct(bc,0.5):>10.0f}{sum(bc)/n:>10.0f}"
          f"{_pct(bc,0.9):>10.0f}", flush=True)
    print(f"  → cell reduction (mean): {sum(fc)/max(sum(bc),1):.2f}×", flush=True)
    print(f"  {'full ms/net':<20}{_pct(fm,0.5):>10.2f}{sum(fm)/n:>10.2f}"
          f"{_pct(fm,0.9):>10.2f}", flush=True)
    print(f"  {'bounded ms/net':<20}{_pct(bm,0.5):>10.2f}{sum(bm)/n:>10.2f}"
          f"{_pct(bm,0.9):>10.2f}", flush=True)
    print(f"  → speedup (total): {tot_f/max(tot_b,1e-9):.2f}× "
          f"({tot_f/n:.2f} → {tot_b/n:.2f} ms/net)", flush=True)
    print(f"\n  routed: full {full_routed}/{n} ({100*full_routed/n:.1f}%), "
          f"bounded {bnd_routed}/{n} ({100*bnd_routed/n:.1f}%)", flush=True)
    print(f"  routability LOST by bounding (full ok, bounded fail): {lost} "
          f"({100*lost/max(full_routed,1):.1f}% of full's routes)", flush=True)
    if wq:
        print(f"  wirelength bounded/full (both routed): median={_pct(wq,0.5):.3f}× "
              f"mean={sum(wq)/len(wq):.3f}× p90={_pct(wq,0.9):.3f}×", flush=True)
    print("\n  Reading: bounding wins iff cell-reduction → speedup AND "
          "routability loss is ~0. Non-zero loss = nets need to detour outside "
          "the pin bbox; the fix is adaptive box growth (drt does this).",
          flush=True)


if __name__ == "__main__":
    main()
